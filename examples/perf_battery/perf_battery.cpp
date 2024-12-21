#include "parler.h"
#include "args.h"
#include "common.h"
#include <stdio.h>
#include <chrono>
#include <functional>

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

std::string benchmark_printout(std::vector<double> generation_samples, std::vector<double> decode_samples, std::vector<double> output_times, std::vector<int> tokens) {
	double gen_mean = mean(generation_samples);
	double dec_mean = mean(decode_samples);
	std::vector<double> gen_output;
	std::vector<double> dec_output;
	std::vector<double> gen_tps;
	std::vector<double> dec_tps;
	for (int i = 0; i < (int) output_times.size(); i++) {
		gen_output.push_back(generation_samples[i]/output_times[i]);
		dec_output.push_back(decode_samples[i]/output_times[i]);
		gen_tps.push_back(tokens[i] / (generation_samples[i] / 1000.0));
		dec_tps.push_back(tokens[i] / (decode_samples[i] / 1000.0));
	}
	double gen_out_mean = mean(gen_output);
	double dec_out_mean = mean(dec_output);
	double mean_gtps = mean(gen_tps);
	double mean_dtps = mean(dec_tps);
	std::string printout = (std::string) "Mean Stats:\n\n" + (std::string) "  Generation Time (ms):             " +  std::to_string(gen_mean) + (std::string) "\n";
	printout += (std::string) "  Decode Time (ms):                 " + std::to_string(dec_mean) + (std::string) "\n";
	printout += (std::string) "  Generation TPS:                   " + std::to_string(mean_gtps) + (std::string) "\n";
	printout += (std::string) "  Decode TPS:                       " + std::to_string(mean_dtps) + (std::string) "\n";
	printout += (std::string) "  Generation Real Time Factor (ms): " + std::to_string(gen_out_mean) + (std::string) "\n";
	printout += (std::string) "  Decode Real Time Factor (ms):     " + std::to_string(dec_out_mean) + (std::string) "\n";
	return printout;
}


int main(int argc, const char ** argv) {
	int default_n_threads = 10;
	arg_list args;
    args.add_argument(string_arg("--model-path", "(REQUIRED) The local path of the gguf model file for Parler TTS mini v1.", "-mp", true));
    args.add_argument(int_arg("--n-threads", "The number of cpu threads to run generation with. Defaults to 10.", "-nt", false, &default_n_threads));
    args.add_argument(bool_arg("--use-metal", "(OPTIONAL) whether or not to use metal acceleration.", "-m"));
    args.add_argument(bool_arg("--no-cross-attn", "(OPTIONAL) Whether to not include cross attention", "-ca"));
    args.parse(argc, argv);
    if (args.for_help) {
        args.help();
        return 0;
    }
    args.validate();

    struct parler_tts_runner * runner = runner_from_file(args.get_string_param("--model-path"), *args.get_int_param("--n-threads"), !args.get_bool_param("--use-metal"), !args.get_bool_param("--no-cross-attn"));
    runner->sampler->temperature = 0.7;
    runner->sampler->repetition_penalty = 1.1;
    std::vector<double> generation_samples;
    std::vector<double> decode_samples;
    std::vector<double> output_times;
    std::vector<int> tokens;
    
    for (std::string sentence : TEST_SENTENCES) {
    	perf_cb cb1 = [&]{
    		runner->generate_audio_tokens(sentence);
    	};
    	double generation_ms = benchmark_ms(cb1);
    	std::vector<uint32_t> audio_tokens;
    	runner->adjust_output_tokens(runner->pctx->output_tokens, audio_tokens);
    	output_times.push_back((double)(((audio_tokens.size() / 9) * 512) / 44.1));
    	tokens.push_back((int)audio_tokens.size());
		perf_cb cb2 = [&]{
    		tts_response outputs;
    		runner->just_audio_token_decode((uint32_t *)audio_tokens.data(), audio_tokens.size() / 9, &outputs);
    	};
    	double decode_ms = benchmark_ms(cb2);
    	generation_samples.push_back(generation_ms);
    	decode_samples.push_back(decode_ms);
    }

    fprintf(stdout, benchmark_printout(generation_samples, decode_samples, output_times, tokens).c_str());
	return 0;
}
