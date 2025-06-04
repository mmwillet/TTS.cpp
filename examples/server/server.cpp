#include "httplib.h"
#include "ggml.h"
#include "util.h"
#include <cstdio>
#include <string>
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"
// mime type for sending response
#define MIMETYPE_WAV "audio/wav"
#define MIMETYPE_AIFF "audio/aiff"
#define MIMETYPE_JSON "application/json; charset=utf-8"
#define MIMETYPE_HTML "text/html; charset=utf-8"
#define MIMETYPE_TXT "text/plain"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <csignal>
#include <thread>
#include <unordered_map>
#include <filesystem>
#include <chrono>
#include "tts.h"
#include "audio_file.h"
#include "args_common.h"
#include "tts_server_thread_osx.h"

#include "index.html.hpp"

namespace {

using json = nlohmann::ordered_json;

void res_ok_json_str(httplib::Response & res, str output) {
    res.set_content(output, MIMETYPE_JSON);
    res.status = 200;
}

string safe_json_to_str(const json & data) {
    return data.dump(-1, ' ', false, json::error_handler_t::replace);
}

void res_ok_audio(httplib::Response & res, const vector<uint8_t> & audio, str mime_type) {
    res.set_content(reinterpret_cast<const char *>(audio.data()), audio.size(), mime_type);
    res.status = 200;
}

void res_error(httplib::Response & res, str err) {
    res.set_content(err, MIMETYPE_TXT);
    res.status = 500;
}

class simple_task_queue;

class simple_text_prompt_task {
    mutex condition_mutex{};
    condition_variable condition{};
    friend simple_task_queue;
public:
    str prompt{""};
    str conditional_prompt{""};
    str model{""};
    AudioFileFormat format{};
    generation_configuration gen_config{};
    atomic<chrono::time_point<chrono::steady_clock>> time{};

    vector<uint8_t> response;
    bool success{};
    atomic<bool> locked_by_worker{};

    bool timed_out(int cleanup_timeout) const {
        const auto start{time.load(memory_order_relaxed)};
        return chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - start).count() > cleanup_timeout;
    }

    void respond() {
        lock_guard lock{condition_mutex};
        locked_by_worker.store(false);
        condition.notify_one();
    }
};

struct worker;

class simple_task_queue {
    mutex rw_mutex{};
    condition_variable condition{};
    deque<weak_ptr<simple_text_prompt_task>> queue{};
public:
    vector<unique_ptr<worker>> workers{};
    atomic<bool> running{true};
    atomic<int> startup_fence{};
    int cleanup_timeout{300};
    str text_encoder_path{""};

    shared_ptr<simple_text_prompt_task> get_next() {
        unique_lock lock(rw_mutex);
        condition.wait(lock, [&] {
            return !queue.empty() || !running.load();
        });
        if (!running.load()) {
            return {};
        }
        do {
            shared_ptr result = queue.front().lock();
            queue.pop_front();
            if (!result) {
                continue;
            }
            if (result->timed_out(cleanup_timeout)) {
                result->respond();
                continue;
            }
            return result;
        } while (!queue.empty());
        return {};
    }

    void terminate();

    void request(shared_ptr<simple_text_prompt_task> & task) {
        unique_lock lock{task->condition_mutex};
        task->time.store(chrono::steady_clock::now(), memory_order_relaxed);
        {
            unique_lock lock2{rw_mutex};
            task->response.clear();
            task->success = false;
        }
        task->locked_by_worker.store(true, memory_order_relaxed);
        {
            lock_guard lock2(rw_mutex);
            queue.emplace_back(task);
            condition.notify_one();
        }
        do {
            if (condition.wait_for(lock, chrono::seconds(1), [&] {
                return !task->locked_by_worker.load() || !running.load();
            })) {
                return;
            }
        } while (!task->timed_out(cleanup_timeout));
    }
};

struct worker {
    worker(simple_task_queue & q, const arg_list & args, const unordered_map<string, string> & model_map)
        : q{q}, args{args}, model_map{model_map} {
    }
    reference_wrapper<simple_task_queue> q;
    reference_wrapper<const arg_list> args;
    reference_wrapper<const unordered_map<string, string>> model_map;

    unordered_map<string, unique_ptr<tts_runner>> runners{};
    tts_server_threading::native_thread worker_thread{};

    void loop() {
        const arg_list & args_ = args.get();
        const int n_threads{args_["n-threads"]};
        const generation_configuration startup_config{parse_generation_config(args_)};
        const bool cpu_only{!args_["use-metal"]};

        for (const auto & [id, path]: model_map.get()) {
            runners[id].reset(runner_from_file(path.c_str(), n_threads, startup_config, cpu_only));
        }
        q.get().startup_fence.fetch_sub(1, memory_order_acq_rel);

        while (q.get().running.load()) {
            if (shared_ptr const task{q.get().get_next()}) {
                process_task(*task);
                task->respond();
            }
        }
    }

    void process_task(simple_text_prompt_task & task) {
        tts_runner * runner = &*runners[task.model];
        if (*task.conditional_prompt) {
            TTS_ASSERT(*q.get().text_encoder_path);
            update_conditional_prompt(runner, q.get().text_encoder_path, task.conditional_prompt);
        }
        tts_response data;
        task.success = !generate(runner, task.prompt, data, task.gen_config);
        if (!task.success) {
            return;
        }

        AudioFile<float> file{};
        file.setSampleRate(runner->sampling_rate);
        file.samples[0] = vector(data.data, data.data + data.n_outputs);
        const bool write_audio_data_result{file.writeData(task.response, task.format)};
        TTS_ASSERT(write_audio_data_result);
    }
};

void simple_task_queue::terminate() {
    if (workers.empty()) {
        return;
    }
    {
        lock_guard lock{rw_mutex};
        running.store(false);
        condition.notify_all();
    }
    for (const auto & w : workers) {
        w->worker_thread.join();
    }
    workers.clear();
}

std::function<void()> shutdown_handler;

void signal_handler(int /*signal*/) {
    static atomic_flag is_terminating{};
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }
    shutdown_handler();
}
}

int main(int argc, const char ** argv) {
    simple_task_queue q{};
    arg_list args{};
    add_common_args(args);
    args.add({
        "", "default-model", "dm",
        "The default model to use when multiple models (a directory with multiple GGUF files) are provided. "
        "This can be set by giving the path to the model (./models/Kokoro_no_espeak.gguf), "
        "the filename (Kokoro_no_espeak.gguf), or the model ID itself (Kokoro_no_espeak)"
    });
    add_text_encoder_arg(args);
    args.add({8080, "port", "p", "The port to use. Defaults to 8080"});
    args.add({"127.0.0.1", "host", "h", "The hostname of the server. Defaults to 127.0.0.1"});
    args.add({
        max(static_cast<int>(thread::hardware_concurrency()) - 1, 3), "n-http-threads", "ht",
        "The number of http threads to use. Defaults to hardware concurrency minus 1"
    });
    args.add({300, "timeout", "t", "The server side timeout on http calls in seconds. Defaults to 300 seconds"});
    args.add({1, "n-parallelism", "np", "The number of parallel models to run asynchronously. Defaults to 1"});
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    args.add({"", "ssl-file-cert", "sfc", "The local path to the PEM encoded SSL certificate"});
    args.add({"", "ssl-file-key", "sfk", "The local path to the PEM encoded SSL private key"});
#endif
    args.parse(argc, argv);
    q.startup_fence.store(args["n-parallelism"], memory_order_relaxed);
    q.cleanup_timeout = args["timeout"];
    q.text_encoder_path = args["text-encoder-path"];

    unique_ptr<httplib::Server> svr;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    {
        const str cert{args["ssl-file-cert"]};
        const str key{args["ssl-file-key"]};
        if (*cert) {
            TTS_ASSERT(*key);
            fprintf(stdout, "Running with SSL: key = %s, cert = %s\n", key, cert);
            svr = make_unique<httplib::SSLServer>(key, cert);
        } else {
            TTS_ASSERT(!*key);
            fprintf(stdout, "Running without SSL\n");
            svr = make_unique<httplib::Server>();
        }
    }
#else
    svr = make_unique<httplib::Server>();
#endif

    std::unordered_map<std::string, std::string> model_map = {};
    if (const str model_path{args["model-path"]}; filesystem::is_directory(model_path)) {
        for (auto const &entry : std::filesystem::directory_iterator(model_path)) {
            if (!entry.is_directory() && entry.path().extension() == ".gguf") {
                const std::string id = entry.path().stem();
                model_map[id] = entry.path().string();
            }
        }
        if (model_map.empty()) {
            fprintf(stderr, "No model found in directory %s", model_path);
            return 1;
        }
    } else {
        const std::filesystem::path path = model_path;
        model_map[path.stem()] = path;
    }

    str default_model{args["default-model"]};
    if (*default_model) {
        const string model{filesystem::path{default_model}.stem()};
        if (auto found = model_map.find(model); found != model_map.end()) {
            default_model = found->first.c_str();
        } else {
            fprintf(stderr, "Invalid Default Model Provided: %s", model.c_str());
            return 1;
        }
    } else {
        default_model = model_map.begin()->first.c_str();
    }

    const string models_json_output{[&model_map] {
        vector<json> models = {};
        const auto model_creation{chrono::system_clock::now().time_since_epoch().count()};
        for (const auto & id: model_map | views::keys) {
            json model{
                {"id", id},
                {"object", "model"},
                {"created", model_creation},
                {"owned_by", "tts.cpp"}
            };
            models.push_back(model);
        }
        return safe_json_to_str({{"object", "list"}, {"data", models}});
    }()};

    svr->set_logger([](const httplib::Request & req, const httplib::Response & res) {
        if (req.path == "/v1/health") {
            return;
        }

        fprintf(stdout, "request: %s %s %s %d\n", req.method.c_str(), req.path.c_str(), req.remote_addr.c_str(), res.status);
    });

    // set timeouts and change hostname and port
    const int timeout{args["timeout"]};
    svr->set_read_timeout(timeout);
    svr->set_write_timeout(timeout);

    // register server middlewares
    svr->set_pre_routing_handler([&q](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        // If this is OPTIONS request, skip validation because browsers don't include Authorization header
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Credentials", "true");
            res.set_header("Access-Control-Allow-Methods",     "GET, POST");
            res.set_header("Access-Control-Allow-Headers",     "*");
            res.set_content("", "text/html"); // blank response, no data
            return httplib::Server::HandlerResponse::Handled; // skip further processing
        }
        if (q.startup_fence.load(memory_order_relaxed)) {
            res_error(res, "Loading model");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    const generation_configuration startup_config{parse_generation_config(args)};
    const auto handle_tts = [
        &q,
        &model_map,
        default_model,
        &startup_config
    ](const httplib::Request &req, httplib::Response & res) {
        thread_local auto task{make_shared<simple_text_prompt_task>()};
        if (task->locked_by_worker.load()) {
            res_error(res, "Service unavailable");
            return;
        }

        const json data(json::parse(req.body));

        if (!data.contains("input") || !data.at("input").is_string()) {
            res_error(res, "the 'input' field is required for tts generation and must be passed as a string");
            return;
        }
        const string & prompt = data.at("input").get<string>();
        if (prompt.empty()) {
            res_error(res, "the 'input' field must be a non-empty string");
            return;
        }
        task->prompt = prompt.c_str();

        string conditional_prompt;
        if (data.contains("conditional_prompt") && data.at("conditional_prompt").is_string()) {
            if (!*q.text_encoder_path) {
                res_error(res, "A text encoder path must be specified on server initialization "
                "in order to support conditional prompting.");
                return;
            }
            conditional_prompt = data.at("conditional_prompt").get<string>();
        }
        task->conditional_prompt = conditional_prompt.c_str();

        string model;
        if (data.contains("model") && data.at("model").is_string()) {
            model = data.at("model").get<string>();
            if (!model_map.contains(model)) {
                res_error(res, "Invalid Model");
                return;
            }
            task->model = model.c_str();
        } else {
            task->model = default_model;
        }

        str mime_type = MIMETYPE_WAV;
        AudioFileFormat format = AudioFileFormat::Wave;
        if (data.contains("response_format") && data.at("response_format").is_string()) {
            if (const string & requested = data.at("response_format").get<string>(); requested == "aiff") {
                mime_type = MIMETYPE_AIFF;
                format = AudioFileFormat::Aiff;
            } else if (requested != "wav" && requested != "wave") {
                res_error(res,
                    "Currently 'wav' and 'aiff' are the only supported formats for the 'response_format' field");
                return;
            }
        }
        task->format = format;

        task->gen_config = startup_config;
        if (data.contains("temperature") && data.at("temperature").is_number()) {
            task->gen_config.temperature = data.at("temperature").get<float>();
        }
        if (data.contains("top_k") && data.at("top_k").is_number()) {
            task->gen_config.top_k = data.at("top_k").get<int>();
        }
        if (data.contains("top_p") && data.at("top_p").is_number()) {
            task->gen_config.top_p = data.at("top_p").get<float>();
        }
        if (data.contains("repetition_penalty") && data.at("repetition_penalty").is_number()) {
            task->gen_config.repetition_penalty = data.at("repetition_penalty").get<float>();
        }
        string voice;
        if (data.contains("voice") && data.at("voice").is_string()) {
            voice = data.at("voice").get<string>();
            task->gen_config.voice = voice.c_str();
        }

        q.request(task);

        if (task->locked_by_worker.load()) {
            res_error(res, "Timed out");
            return;
        }
        if (!task->success) {
            res_error(res, "Generation failed");
            return;
        }
        if (task->response.empty()) {
            res_error(res, "Model returned an empty response");
            return;
        }

        res_ok_audio(res, task->response, mime_type);
    };

    // register API routes
    svr->Get("/", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(reinterpret_cast<const char*>(index_html), MIMETYPE_HTML);
        res.status = 200;
    });
    svr->Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res_ok_json_str(res, R"({"status":"ok")");
    });
    svr->Post("/v1/audio/speech", handle_tts);
    svr->Get("/v1/models", [output = models_json_output.c_str()](const httplib::Request & _, httplib::Response & res) {
        res_ok_json_str(res, output);
    });

    // Start the server
    const int n_http_threads{args["n-http-threads"]};
    svr->new_task_queue = [n_http_threads] {
        return new httplib::ThreadPool(n_http_threads);
    };

    shutdown_handler = [&svr] {
        svr->stop();
    };

    const str host{args["host"]};
    const int port{args["port"]};
    // bind HTTP listen port
    if (!svr->bind_to_port(host, port)) {
        fprintf(stderr, "%s: couldn't bind HTTP server socket, hostname: %s, port: %d\n", __func__, host, port);
        shutdown_handler();
        return 1;
    }

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action{};
    sigint_action.sa_handler = signal_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, nullptr);
    sigaction(SIGTERM, &sigint_action, nullptr);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    fprintf(stdout, "%s: loading model and initializing main loop\n", __func__);
    for (int i{q.startup_fence.load(memory_order_relaxed)}; i > 0; i--) {
        auto & w = q.workers.emplace_back(make_unique<worker>(q, args, model_map));
        w->worker_thread = tts_server_threading::native_thread(&worker::loop, w.get());//thread(&worker::loop, w.get());
    }
    fprintf(stdout, "%s: HTTP server is listening with %d threads on http://%s:%d/\n",
        __func__, n_http_threads, host, port);
    svr->listen_after_bind();
    fprintf(stdout, "%s: HTTP server listening on hostname: %s and port: %d, is shutting down.\n",
        __func__, host, port);
    q.terminate();

    return 0;
}
