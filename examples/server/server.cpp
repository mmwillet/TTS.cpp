#include "httplib.h"
#include "ggml.h"
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"
// mime type for sending response
#define MIMETYPE_WAV "audio/wav"
#define MIMETYPE_AIFF "audio/aiff"
#define MIMETYPE_JSON "application/json; charset=utf-8"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cinttypes>
#include <deque>
#include <memory>
#include <mutex>
#include <signal.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include "tts.h"
#include "audio_file.h"
#include "args.h"
#include "common.h"

enum server_state {
    LOADING,  // Server is starting up / model loading
    READY,    // Server is ready
};

// These are form copied from llama.cpp which copied them from openAI chat:
// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
// In testing, openAI TTS endpoints make use of the same behavior.
enum error_type {
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_AUTHENTICATION, // not currently supported as auth keys are not built in yet
    ERROR_TYPE_SERVER,
    ERROR_TYPE_NOT_FOUND,
    ERROR_TYPE_PERMISSION, // not currently supported as auth keys are not built in yet
    ERROR_TYPE_UNAVAILABLE, // custom error
    ERROR_TYPE_NOT_SUPPORTED, // custom error
};

enum task_type {
    TTS,
    CONDITIONAL_PROMPT,
};

using json = nlohmann::ordered_json;

template <typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    // Fallback null to default value
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key);
        } catch (NLOHMANN_JSON_NAMESPACE::detail::type_error const &) {
            fprintf(stderr, "Wrong type supplied for parameter '%s'. Expected '%s', using default value\n", key.c_str(), json(default_value).type_name());
            return default_value;
        }
    } else {
        return default_value;
    }
}

bool write_audio_data(float * data, size_t length, std::vector<uint8_t> & output, AudioFileFormat format = AudioFileFormat::Wave, float sample_rate = 44100.f, float frequency = 440.f, int channels = 1) {
    AudioFile<float> file;
    file.setBitDepth(16);
    file.setSampleRate(sample_rate);
    file.setNumChannels(channels);
    int samples = (int) (length / channels);
    file.setNumSamplesPerChannel(samples);
    for (int channel = 0; channel < channels; channel++) {
        for (int i = 0; i < samples; i++) {
            file.samples[channel][i] = data[i];
        }
    }
    return file.writeData(output, format);
}

static void log_server_request(const httplib::Request & req, const httplib::Response & res) {
    if (req.path == "/v1/health") {
        return;
    }

    fprintf(stdout, "request: %s %s %s %d\n", req.method.c_str(), req.path.c_str(), req.remote_addr.c_str(), res.status);
}

struct simple_text_prompt_task {
    simple_text_prompt_task(task_type task, std::string prompt): task(task), prompt(prompt) {
        id = rand();
        time = std::chrono::steady_clock::now();
    }

    task_type task;
    int id;
    std::string prompt;
    generation_configuration * gen_config;
    void * response;
    size_t length;
    bool success = false;
    std::string message;
    std::chrono::time_point<std::chrono::steady_clock> time;
    float sample_rate = 44100.0f;

    bool timed_out(int t) {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::ratio<1>> duration = now - time;
        return (int) duration.count() > t;
    }
};

struct simple_task_queue {
    std::mutex rw_mutex;
    std::condition_variable condition;
    std::deque<simple_text_prompt_task*> queue;
    bool running = true;

    struct simple_text_prompt_task * get_next() {
        struct simple_text_prompt_task * resp;
        std::unique_lock<std::mutex> lock(rw_mutex);
        condition.wait(lock, [&]{ 
            return !queue.empty() || !running; 
        });
        if (!running) {
            return nullptr;
        }
        resp = queue.front();
        queue.pop_front();
        lock.unlock();
        return resp;
    }

    void terminate() {
        std::lock_guard<std::mutex> lock(rw_mutex);
        running = false; 
        condition.notify_all();
    }

    void push(struct simple_text_prompt_task * task) {
        std::lock_guard<std::mutex> lock(rw_mutex);
        queue.push_back(task);
        condition.notify_one();
    }
};

struct simple_response_map {
    std::mutex rw_mutex;
    std::condition_variable updated;
    int cleanup_timeout = 300;
    std::atomic<bool> running = true;
    std::thread * cleanup_thread;

    std::map<int, simple_text_prompt_task*> completed;

    void cleanup_routine() {
        std::unique_lock<std::mutex> lock(rw_mutex);
        while(true) {
            updated.wait(lock, [&]{
                return completed.size() > 100 || !running;
            });
            if (!running) {
                return;
            }
            auto now = std::chrono::steady_clock::now();
            std::vector<int> deletable;
            for (auto const& [key, task] : completed) {
                if (task->timed_out(cleanup_timeout)) {
                    deletable.push_back(key);
                }
            }
            for (auto const id : deletable) {
                completed.erase(id);
            }
        }
    }

    void terminate() {
        std::lock_guard<std::mutex> lock(rw_mutex);
        running = false; 
        updated.notify_all();
    }

    void push(struct simple_text_prompt_task * task) {
        std::unique_lock<std::mutex> lock(rw_mutex);
        completed[task->id] = task;
        lock.unlock();
        updated.notify_all();
    }

    struct simple_text_prompt_task * get(int id) {
        std::unique_lock<std::mutex> lock(rw_mutex);
        struct simple_text_prompt_task * resp = nullptr;
        try {
            return completed.at(id);
        } catch (const std::out_of_range& e) {
            updated.wait(lock, [&]{
                return completed.find(id) != completed.end() || !running;
            });
            if (!running) {
                return nullptr;
            }
            return completed.at(id);
        }
    }
};

void init_response_map(simple_response_map * rmap) {
    rmap->cleanup_routine();
}

struct worker {
    worker(struct simple_task_queue * task_queue, struct simple_response_map * response_map, std::string text_encoder_path = "", int task_timeout = 300): task_queue(task_queue), response_map(response_map), text_encoder_path(text_encoder_path), task_timeout(task_timeout) {};
    ~worker() {
        delete runner;
    }
    struct simple_task_queue * task_queue;
    struct simple_response_map * response_map;

    struct tts_runner * runner = nullptr;
    std::string text_encoder_path;
    std::atomic<bool> running = true;
    std::thread * thread = nullptr;

    int task_timeout;

    void terminate() {
        running = false;
    }

    void loop() {
        while (running) {
            struct simple_text_prompt_task * task = task_queue->get_next();
            if (task) {
                process_task(task);
            }
        }
    }

    void process_task(struct simple_text_prompt_task * task) {
        if (task->timed_out(task_timeout)) {
            return;
        }
        int outcome;
        tts_response * data = nullptr;
        switch(task->task) {
            case TTS:
                data = new tts_response;
                outcome = generate(runner, task->prompt, data, task->gen_config);
                task->response = (void*) data->data;
                task->length = data->n_outputs;
                task->sample_rate = runner->sampling_rate;
                task->success = outcome == 0;
                response_map->push(task);
                break;
            case CONDITIONAL_PROMPT:
                if (text_encoder_path.size() == 0) {
                    task->message = "A text encoder path must be specified on server initialization in order to support conditional prompting.";
                    response_map->push(task);
                    break;
                }
                update_conditional_prompt(runner, text_encoder_path, task->prompt);
                task->success = true;
                response_map->push(task);
                break;
        }
    }
};

void init_worker(std::string model_path, int n_threads, bool cpu_only, generation_configuration * config, worker * w) {
    w->runner = runner_from_file(model_path, n_threads, config, cpu_only);
    w->loop();
}

typedef std::vector<worker*> worker_pool;

void terminate(worker_pool * pool) {
    for (auto w : *pool) {
        w->terminate();
    }
    if (pool->size() > 0) {
        (*pool)[0]->task_queue->terminate();
        (*pool)[0]->response_map->terminate();
    }
    for (auto w : *pool) {
        if (w->thread) {
            w->thread->join();
        }
        delete w;
    }
}

static std::string safe_json_to_str(json data) {
    return data.dump(-1, ' ', false, json::error_handler_t::replace);
}

// this function maybe used outside of server_task_result_error
static json format_error_response(const std::string & message, const enum error_type type) {
    std::string type_str;
    int code = 500;
    switch (type) {
        case ERROR_TYPE_INVALID_REQUEST:
            type_str = "invalid_request_error";
            code = 400;
            break;
        case ERROR_TYPE_AUTHENTICATION:
            type_str = "authentication_error";
            code = 401;
            break;
        case ERROR_TYPE_NOT_FOUND:
            type_str = "not_found_error";
            code = 404;
            break;
        case ERROR_TYPE_SERVER:
            type_str = "server_error";
            code = 500;
            break;
        case ERROR_TYPE_PERMISSION:
            type_str = "permission_error";
            code = 403;
            break;
        case ERROR_TYPE_NOT_SUPPORTED:
            type_str = "not_supported_error";
            code = 501;
            break;
        case ERROR_TYPE_UNAVAILABLE:
            type_str = "unavailable_error";
            code = 503;
            break;
    }
    return json {
        {"code", code},
        {"message", message},
        {"type", type_str},
    };
}

std::function<void(int)> shutdown_handler;
std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

int main(int argc, const char ** argv) {
    int default_n_threads = std::min((int)std::thread::hardware_concurrency(), 1);
    int default_http_threads = std::min((int)std::thread::hardware_concurrency() - 1, 3);
    int default_n_parallel = 1;
    int default_port = 8080;
    int default_timeout = 300;
    std::string default_host = "127.0.0.1";
    float default_temperature = 0.9f;
    int default_top_k = 50;
    float default_repetition_penalty = 1.1f;

    arg_list args;
    args.add_argument(float_arg("--temperature", "(OPTIONAL) The temperature to use when generating outputs. Defaults to 0.9.", "-t", false, &default_temperature));
    args.add_argument(int_arg("--topk", "(OPTIONAL) when set to an integer value greater than 0 generation uses nucleus sampling over topk nucleaus size. Defaults to 50.", "-tk", false, &default_top_k));
    args.add_argument(float_arg("--repetition-penalty", "The by channel repetition penalty to be applied the sampled output of the model. defaults to 1.0.", "-r", false, &default_repetition_penalty));
    args.add_argument(string_arg("--model-path", "(REQUIRED) The local path of the gguf model file for Parler TTS mini or large v1 or Kokoro.", "-mp", true));
    args.add_argument(int_arg("--n-threads", "The number of cpu threads to run generation with. Defaults to hardware concurrency.", "-nt", false, &default_n_threads));
    args.add_argument(bool_arg("--use-metal", "(OPTIONAL) Whether to use metal acceleration", "-m"));
    args.add_argument(bool_arg("--no-cross-attn", "(OPTIONAL) Whether to not include cross attention", "-ca"));
    args.add_argument(string_arg("--text-encoder-path", "(OPTIONAL) The local path of the text encoder gguf model for conditional generaiton.", "-tep", false));
    args.add_argument(string_arg("--ssl-file-cert", "(OPTIONAL) The local path to the PEM encoded ssl cert.", "-sfc", false));
    args.add_argument(string_arg("--ssl-file-key", "(OPTIONAL) The local path to the PEM encoded ssl private key.", "-sfk", false));
    args.add_argument(int_arg("--port", "(OPTIONAL) The port to use. Defaults to 8080.", "-p", false, &default_port));
    args.add_argument(string_arg("--host", "(OPTIONAL) the hostname of the server. Defaults to '127.0.0.1'.", "-h", false, default_host));
    args.add_argument(int_arg("--n-http-threads", "(OPTIONAL) The number of http threads to use. Defaults to hardware concurrency minus 1.", "-ht", false, &default_http_threads));
    args.add_argument(int_arg("--timeout", "(OPTIONAL) The server side timeout on http calls in seconds. Defaults to 300 seconds.", "-t", false, &default_timeout));
    args.add_argument(int_arg("--n-parallelism", "(OPTIONAL) the number of parallel models to run asynchronously. Deafults to 1.", "-np", false, &default_n_parallel));
    args.add_argument(string_arg("--voice", "(OPTIONAL) the default voice to use when generating audio. Only used with applicable models.", "-v", false, "af_alloy"));
    args.parse(argc, argv);
    if (args.for_help) {
        args.help();
        return 0;
    }
    args.validate();

    generation_configuration * default_generation_config = new generation_configuration(args.get_string_param("--voice"), *args.get_int_param("--topk"), *args.get_float_param("--temperature"), *args.get_float_param("--repetition-penalty"), !args.get_bool_param("--no-cross-attn"));

    worker_pool * pool = nullptr;
    struct simple_task_queue * tqueue = new simple_task_queue;
    struct simple_response_map * rmap  = new simple_response_map;

    bool conditional_prompt_viable = args.get_string_param("--text-encoder-path").size() > 0 && *args.get_int_param("--n-parallelism") <= 1;

    std::unique_ptr<httplib::Server> svr;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (args.get_string_param("--ssl-file-cert") != "" && args.get_string_param("--ssl-file-key") != "") {
        fprintf(stdout, "Running with SSL: key = %s, cert = %s\n", args.get_string_param("--ssl-file-key").c_str(), args.get_string_param("--ssl-file-cert").c_str());
        svr.reset(new httplib::SSLServer(args.get_string_param("--ssl-file-key").c_str(), args.get_string_param("--ssl-file-cert").c_str()));
    } else {
        fprintf(stdout, "Running without SSL\n");
        svr.reset(new httplib::Server());
    }
#else
    if (args.get_string_param("--ssl-file-cert") != "" && args.get_string_param("--ssl-file-key") != "") {
        fprintf(stderr, "Server is built without SSL support\n");
        return 1;
    }
    svr.reset(new httplib::Server());
#endif

    std::atomic<server_state> state{LOADING};

    svr->set_logger(log_server_request);

    auto res_error = [](httplib::Response & res, const json & error_data) {
        json final_response {{"error", error_data}};
        res.set_content(safe_json_to_str(final_response), MIMETYPE_JSON);
        res.status = json_value(error_data, "code", 500);
    };

    auto res_ok_json = [](httplib::Response & res, const json & data) {
        res.set_content(safe_json_to_str(data), MIMETYPE_JSON);
        res.status = 200;
    };

    auto res_ok_audio = [](httplib::Response & res, const std::vector<uint8_t> & audio, std::string mime_type) {
        res.set_content((char*)audio.data(), audio.size(), mime_type);
        res.status = 200;
    };

    svr->set_exception_handler([&res_error](const httplib::Request &, httplib::Response & res, const std::exception_ptr & ep) {
        std::string message;
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception & e) {
            message = e.what();
        } catch (...) {
            message = "Unknown Exception";
        }

        json formatted_error = format_error_response(message, ERROR_TYPE_SERVER);
        fprintf(stderr, "got exception: %s\n", formatted_error.dump().c_str());
        res_error(res, formatted_error);
    });

    svr->set_error_handler([&res_error](const httplib::Request &, httplib::Response & res) {
        if (res.status == 404) {
            res_error(res, format_error_response("File Not Found", ERROR_TYPE_NOT_FOUND));
        }
    });

    // set timeouts and change hostname and port
    svr->set_read_timeout(*args.get_int_param("--timeout"));
    svr->set_write_timeout(*args.get_int_param("--timeout"));

    auto middleware_server_state = [&res_error, &state](const httplib::Request & req, httplib::Response & res) {
        server_state current_state = state.load();
        if (current_state == LOADING) {
            res_error(res, format_error_response("Loading model", ERROR_TYPE_UNAVAILABLE));
            return false;
        }
        return true;
    };

    // register server middlewares
    svr->set_pre_routing_handler([&middleware_server_state](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        // If this is OPTIONS request, skip validation because browsers don't include Authorization header
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Credentials", "true");
            res.set_header("Access-Control-Allow-Methods",     "GET, POST");
            res.set_header("Access-Control-Allow-Headers",     "*");
            res.set_content("", "text/html"); // blank response, no data
            return httplib::Server::HandlerResponse::Handled; // skip further processing
        }
        if (!middleware_server_state(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    const auto handle_health = [&](const httplib::Request &, httplib::Response & res) {
        json health = {{"status", "ok"}};
        res_ok_json(res, health);
    };

    const auto handle_tts = [&tqueue, &rmap, &res_error, &res_ok_audio, &default_generation_config](const httplib::Request &req, httplib::Response & res) {
        json data = json::parse(req.body);
        if (!data.contains("input") || !data.at("input").is_string()) {
            json formatted_error = format_error_response("the 'input' field is required for tts generation and must be passed as a string.", ERROR_TYPE_INVALID_REQUEST);
            res_error(res, formatted_error);
            return;
        }

        std::string mime_type = MIMETYPE_WAV;
        AudioFileFormat audio_type = AudioFileFormat::Wave;
        if (data.contains("response_format") && data.at("response_format").is_string()) {
            std::string format = data.at("response_format").get<std::string>();
            if (format != "wav" && format != "wave" && format != "aiff") {
                json formatted_error = format_error_response("Currently 'wav' and 'aiff' are the only supported formats for the 'response_format' field.", ERROR_TYPE_NOT_SUPPORTED);
                res_error(res, formatted_error);
                return;
            } else if (format == "aiff") {
                mime_type = MIMETYPE_AIFF;
                audio_type = AudioFileFormat::Aiff;
            }
        }

        std::string prompt = data.at("input").get<std::string>();
        struct simple_text_prompt_task * task = new simple_text_prompt_task(TTS, prompt);
        int id = task->id;
        generation_configuration * conf = new generation_configuration();
        std::memcpy(conf, default_generation_config, sizeof(generation_configuration));
        float temp;
        float rep_pen;
        int top_k;
        if (data.contains("temperature") && data.at("temperature").is_number()) {
            temp = data.at("temperature").get<float>();
            conf->temperature = temp;
        }

        if (data.contains("top_k") && data.at("top_k").is_number()) {
            top_k = data.at("top_k").get<int>();
            conf->top_k = top_k;
        }

        if (data.contains("repetition_penalty") && data.at("repetition_penalty").is_number()) {
            rep_pen = data.at("repetition_penalty").get<float>();
            conf->repetition_penalty = rep_pen;
        }

        if (data.contains("voice") && data.at("voice").is_string()) {
            conf->voice = data.at("voice").get<std::string>();
        }

        task->gen_config = conf;
        tqueue->push(task);
        struct simple_text_prompt_task * rtask = rmap->get(id);
        if (!rtask->success) {
            json formatted_error = format_error_response(rtask->message, ERROR_TYPE_SERVER);
            res_error(res, formatted_error);
            return;
        }

        std::vector<uint8_t> audio;
        bool success = write_audio_data((float *)rtask->response, rtask->length, audio, audio_type, rtask->sample_rate);
        if (!success) {
            json formatted_error = format_error_response("failed to write audio data", ERROR_TYPE_SERVER);
            res_error(res, formatted_error);
            return;
        }

        res_ok_audio(res, audio, mime_type);
    };

    const auto handle_conditional = [&args, &tqueue, &rmap, &res_error, &res_ok_json](const httplib::Request & req, httplib::Response & res) {
        if (args.get_string_param("--text-encoder-path").size() == 0) {
            json formatted_error = format_error_response("A '--text-encoder-path' must be specified for conditional generation.", ERROR_TYPE_NOT_SUPPORTED);
            res_error(res, formatted_error);
            return;
        }
        if (*args.get_int_param("--n-parallelism") > 1) {
            json formatted_error = format_error_response("Conditional prompting is not supported for parallelism greater than 1.", ERROR_TYPE_NOT_SUPPORTED);
            res_error(res, formatted_error);
            return;
        }
        json data = json::parse(req.body);
        if (!data.contains("input") || !data.at("input").is_string()) {
            json formatted_error = format_error_response("the 'input' field is required for conditional prompting.", ERROR_TYPE_INVALID_REQUEST);
            res_error(res, formatted_error);
            return;
        }
        std::string prompt = data.at("input").get<std::string>();
        struct simple_text_prompt_task * task = new simple_text_prompt_task(CONDITIONAL_PROMPT, prompt);
        int id = task->id;
        tqueue->push(task);
        struct simple_text_prompt_task * rtask = rmap->get(id);
        if (!rtask->success) {
            json formatted_error = format_error_response(rtask->message, ERROR_TYPE_SERVER);
            res_error(res, formatted_error);
            return;
        }
        json health = {{"status", "ok"}};
        res_ok_json(res, health);
    };

    // register API routes
    svr->Get("/health", handle_health);
    svr->Post("/v1/audio/speech", handle_tts);
    svr->Post("/v1/audio/conditional-prompt", handle_conditional);

    // Start the server
    svr->new_task_queue = [&args] { 
        return new httplib::ThreadPool(*args.get_int_param("--n-http-threads")); 
    };

    // clean up function, to be called before exit
    auto clean_up = [&svr]() {
        svr->stop();
    };

    // bind HTTP listen port
    bool bound = svr->bind_to_port(args.get_string_param("--host"), *args.get_int_param("--port"));

    if (!bound) {
        fprintf(stderr, "%s: couldn't bind HTTP server socket, hostname: %s, port: %d\n", __func__, args.get_string_param("--host").c_str(), *args.get_int_param("--port"));
        clean_up();
        return 1;
    }

    rmap->cleanup_timeout = *args.get_int_param("--timeout");
    rmap->cleanup_thread = new std::thread(init_response_map, rmap);

    // run the HTTP server in a thread
    std::thread t([&]() { svr->listen_after_bind(); });
    svr->wait_until_ready();
    fprintf(stdout, "%s: HTTP server is listening, hostname: %s, port: %d, http threads: %d\n", __func__, args.get_string_param("--host").c_str(), *args.get_int_param("--port"), *args.get_int_param("--n-http-threads"));

    // load the model
    fprintf(stdout, "%s: loading model and initializing main loop\n", __func__);

    // It might make sense in the long run to have the primary thread run clean up on the response map and keep the model workers parallel.
    // pool = initialize_workers(args, tqueue, rmap);
    pool = new worker_pool;
    for (int i = *args.get_int_param("--n-parallelism"); i > 0; i--) {
        if (i == 1) {
            fprintf(stdout, "%s: server is listening on http://%s:%d\n", __func__, args.get_string_param("--host").c_str(), *args.get_int_param("--port"));
            worker * w = new worker(tqueue, rmap, args.get_string_param("--text-encoder-path"), *args.get_int_param("--timeout"));
            state.store(READY);
            pool->push_back(w);
            init_worker(args.get_string_param("--model-path"), *args.get_int_param("--n-threads"), !args.get_bool_param("--use-metal"), default_generation_config, w);
        } else {
            worker * w = new worker(tqueue, rmap, args.get_string_param("--text-encoder-path"), *args.get_int_param("--timeout"));
            w->thread = new std::thread(init_worker, args.get_string_param("--model-path"), *args.get_int_param("--n-threads"), !args.get_bool_param("--use-metal"), default_generation_config, w);
            pool->push_back(w);
        }
    }

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    clean_up();
    t.join();
    terminate(pool);
    rmap->cleanup_thread->join();

    return 0;
}
