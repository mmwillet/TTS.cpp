#include <chrono>
#include <iostream>

#include "args_common.h"
#include "tts.h"

namespace {
double benchmark_ms(auto lambda) {
	auto start = std::chrono::steady_clock::now();
	lambda();
	auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

/*
 * These are the 'Harvard Sentences' (https://en.wikipedia.org/wiki/Harvard_sentences). They are phonetically
 * balanced sentences typically used for standardized testing of voice over cellular and telephone systems.
 */
constexpr array TEST_SENTENCES = {
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

void benchmark_printout(tts_arch arch, const vector<double> & generation_samples, const vector<double> & output_times) {
	const str arch_name = SUPPORTED_ARCHITECTURES[arch];
	const double gen_mean = mean(generation_samples);
	std::vector<double> gen_output;
	for (size_t i = 0; i < output_times.size(); i++) {
		gen_output.push_back(generation_samples[i]/output_times[i]);
	}
	double gen_out_mean = mean(gen_output);
	cout << "Mean Stats for arch " << arch_name << ":\n\n  Generation Time (ms):             ";
    cout << gen_mean << endl;
	cout << "  Generation Real Time Factor (ms): " << gen_out_mean << endl;
}
}

int main(int argc, const char ** argv) {
    arg_list args{};
    add_common_args(args);
    args.parse(argc, argv);

    const generation_configuration config{parse_generation_config(args)};
    tts_runner * const runner{runner_from_args(args, config)};
    std::vector<double> generation_samples;
    std::vector<double> output_times;
    
    for (const str sentence : TEST_SENTENCES) {
    	tts_response response;
    	const auto cb = [&]{
    		generate(runner, sentence, response, config);
    	};
    	double generation_ms = benchmark_ms(cb);
    	output_times.push_back(response.n_outputs / 44.1);
    	generation_samples.push_back(generation_ms);
    }

    benchmark_printout(runner->arch, generation_samples, output_times);
	return 0;
}
