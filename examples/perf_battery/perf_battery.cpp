#include <stdio.h>

#include <chrono>
#include <functional>
#include <thread>

#include "../../src/models/loaders.h"
#include "args.h"
#include "common.h"

using perf_cb = std::function<void()>;

double benchmark_ms(perf_cb func) {
	auto start = std::chrono::steady_clock::now();
	func();
	auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

/*
 * These are the 'Harvard Sentences' (https://en.wikipedia.org/wiki/Harvard_sentences). They are phonetically
 * balanced sentences typically used for standardized testing of voice over cellular and telephone systems.
 */
std::vector<std::string> TEST_SENTENCES = {
	"The birch canoe slid on the smooth planks.",
	"Glue the sheet to the dark blue background.",
	"It's easy to tell the depth of a well.",
	"These days a chicken leg is a rare dish.",
	"Rice is often served in round bowls.",
	"The juice of lemons makes fine punch.",
	"The box was thrown beside the parked truck.",
	"The hogs were fed chopped corn and garbage.",
	"Four hours of steady work faced us.",
	"A large size in stockings is hard to sell.",
	"The boy was there when the sun rose.",
	"A rod is used to catch pink salmon.",
	"The source of the huge river is the clear spring.",
	"Kick the ball straight and follow through."
	"Help the woman get back to her feet.",
	"A pot of tea helps to pass the evening.",
	"Smoky fires lack flame and heat.",
	"The soft cushion broke the man's fall.",
	"The salt breeze came across from the sea.",
	"The girl at the booth sold fifty bonds.",
	"The small pup gnawed a hole in the sock.",
	"The fish twisted and turned on the bent hook.",
	"Press the pants and sew a button on the vest.",
	"The swan dive was far short of perfect.",
	"The beauty of the view stunned the young boy.",
	"Two blue fish swam in the tank.",
	"Her purse was full of useless trash.",
	"The colt reared and threw the tall rider.",
	"It snowed, rained, and hailed the same morning.",
	"Read verse out loud for pleasure."
};

double mean(std::vector<double> series) {
	double sum = 0.0;
	for (double v : series) {
		sum += v;
	}
	return (double) sum / series.size();
}

std::string benchmark_printout(const char * arch, std::vector<double> generation_samples, std::vector<double> output_times) {
	double gen_mean = mean(generation_samples);
	std::vector<double> gen_output;
	for (int i = 0; i < (int) output_times.size(); i++) {
		gen_output.push_back(generation_samples[i]/output_times[i]);
	}
	double gen_out_mean = mean(gen_output);
	std::string printout = (std::string) "Mean Stats for arch " + arch + ":\n\n" + (std::string) "  Generation Time (ms):             " +  std::to_string(gen_mean) + (std::string) "\n";
	printout += (std::string) "  Generation Real Time Factor (ms): " + std::to_string(gen_out_mean) + (std::string) "\n";
	return printout;
}


int main(int argc, const char ** argv) {
    float default_temperature = 1.0f;
    int default_n_threads = std::max((int)std::thread::hardware_concurrency(), 1);
    int default_top_k = 50;
    float default_repetition_penalty = 1.0f;
	arg_list args;
    args.add_argument(string_arg("--model-path", "(REQUIRED) The local path of the gguf model file for Parler TTS mini v1.", "-mp", true));
    args.add_argument(int_arg("--n-threads", "The number of cpu threads to run generation with. Defaults to hardware concurrency. If hardware concurrency cannot be determined it defaults to 1.", "-nt", false, &default_n_threads));
    args.add_argument(float_arg("--temperature", "The temperature to use when generating outputs. Defaults to 1.0.", "-t", false, &default_temperature));
    args.add_argument(int_arg("--topk", "(OPTIONAL) When set to an integer value greater than 0 generation uses nucleus sampling over topk nucleaus size. Defaults to 50.", "-tk", false, &default_top_k));
    args.add_argument(string_arg("--voice", "(OPTIONAL) The voice to use to generate the audio. This is only used for models with voice packs.", "-v", false, ""));
    args.add_argument(float_arg("--repetition-penalty", "The by channel repetition penalty to be applied the sampled output of the model. defaults to 1.0.", "-r", false, &default_repetition_penalty));
    args.add_argument(bool_arg("--use-metal", "(OPTIONAL) whether or not to use metal acceleration.", "-m"));
    args.add_argument(bool_arg("--no-cross-attn", "(OPTIONAL) Whether to not include cross attention", "-ca"));
    args.parse(argc, argv);
    if (args.for_help) {
        args.help();
        return 0;
    }
    args.validate();

    const generation_configuration config{args.get_string_param("--voice"), *args.get_int_param("--topk"), *args.get_float_param("--temperature"), *args.get_float_param("--repetition-penalty"), !args.get_bool_param("--no-cross-attn")};

    unique_ptr<tts_generation_runner> runner{runner_from_file(args.get_string_param("--model-path").c_str(), *args.get_int_param("--n-threads"), config, !args.get_bool_param("--use-metal"))};
    std::vector<double> generation_samples;
    std::vector<double> output_times;
    
    for (std::string sentence : TEST_SENTENCES) {
    	tts_response response;
    	perf_cb cb = [&]{
    		runner->generate(sentence.c_str(), response, config);
    	};
    	double generation_ms = benchmark_ms(cb);
    	output_times.push_back((double)(response.n_outputs / 44.1));
    	generation_samples.push_back(generation_ms);
    }

    fprintf(stdout, "%s", benchmark_printout(runner->loader.get().arch, generation_samples, output_times).c_str());
    static_cast<void>(!runner.release()); // TODO the destructor doesn't work yet
	return 0;
}
