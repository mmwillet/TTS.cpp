#include "args.h"

#include <iostream>
#include <sstream>

void arg::print_help() const {
    cout << "--" << full_name;
    if (*abbreviation) {
        cout << " (-" << abbreviation << ")";
    }
    if (*description) {
        cout << (required ? ":\n    (REQUIRED) " : ":\n    (OPTIONAL) ") << description << ".\n";
    } else {
        cout << (required ? " is a required parameter.\n" : " is an optional parameter.\n");
    }
}

void arg::parse(span<str> & argv) {
    required = false;
    if (const auto bool_param{get_if<bool>(&value)}) {
        *bool_param = true;
        return;
    }
    if (argv.empty()) {
        fprintf(stderr, "The option '--%s' requires an argument\n", full_name);
        exit(1);
    }
    const str a = argv[0];
    argv = argv.subspan(1);
    if (const auto string_param{get_if<str>(&value)}) {
        *string_param = a;
    } else if (const auto int_param{get_if<int>(&value)}) {
        istringstream{a} >> *int_param;
    } else if (const auto float_param{get_if<float>(&value)}) {
        istringstream{a} >> *float_param;
    }
}

void arg_list::parse(int argc, str argv_[]) {
    TTS_ASSERT(argc);
    span<str> argv{argv_, static_cast<size_t>(argc)};
    argv = argv.subspan(1);
    while (!argv.empty()) {
        str name{argv[0]};
        if (*name != '-') {
            fprintf(stderr, "Only named arguments are supported\n");
            exit(1);
        }
        ++name;
        const map<sv, size_t> * lookup = &abbreviations;
        if (*name == '-') {
            ++name;
            lookup = &full_names;
            if (name == "help"sv) {
                for (const size_t i : full_names | views::values) {
                    args[i].print_help();
                }
                exit(0);
            }
        }
        const auto found = lookup->find(sv{name});
        if (found == lookup->end()) {
            fprintf(stderr, "argument '%s' is not a valid argument. "
                    "Call '--help' for information on all valid arguments.\n", argv[0]);
            exit(1);
        }
        argv = argv.subspan(1);
        args[found->second].parse(argv);
    }
    for (const arg & x : args) {
        if (x.required) {
            fprintf(stderr, "argument '--%s' is required.\n", x.full_name);
            exit(1);
        }
    }
}
