#pragma once

#include <map>
#include <thread>
#include <vector>

#include "imports.h"

/**
 * Holder of one argument.
 */
class arg {
    variant<bool, str, int, float> value;
    bool required;

    void print_help() const;

    void parse(span<str> & argv);

    friend class arg_list;

public:
    const str full_name;
    const str abbreviation;
    const str description;

    template <typename T>
    constexpr arg(T default_value, str full_name, str abbreviation, str description, bool required = false)
        : value{default_value}, required{required},
          full_name{full_name}, abbreviation{abbreviation}, description{description} {
        TTS_ASSERT(full_name[0] != '-');
        TTS_ASSERT(abbreviation[0] != '-');
    }

    template <typename T>
        requires is_same_v<T, bool> || is_same_v<T, str> || is_same_v<T, int> || is_same_v<T, float>
    // ReSharper disable once CppNonExplicitConversionOperator // We want this to automatically cast
    constexpr operator T() const { // NOLINT(*-explicit-constructor)
        return get<T>(value);
    }
};

class arg_list {
    vector<arg> args{};
    map<sv, size_t> full_names{};
    map<sv, size_t> abbreviations{};

public:
    void add(const arg & x) {
        const size_t i{args.size()};
        args.push_back(x);
        TTS_ASSERT(!full_names.contains(args[i].full_name));
        full_names[args[i].full_name] = i;
        if (*args[i].abbreviation) {
            abbreviations[args[i].abbreviation] = i;
        }
    }

    void parse(int argc, str argv_[]);

    constexpr const arg & operator [](sv full_name) const noexcept {
        TTS_ASSERT(full_name[0] != '-');
        return args[full_names.at(full_name)];
    }
};
