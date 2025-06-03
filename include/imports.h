#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string_view>
#include <ranges>
#include <vector>

using namespace std;
using namespace std::string_view_literals;
typedef std::string_view sv;
typedef const char * str;

#define TTS_ABORT(...) tts_abort(__FILE__, __LINE__, __VA_ARGS__)
#define TTS_ASSERT(x) if (!(x)) TTS_ABORT("TTS_ASSERT(%s) failed", #x)
[[noreturn]] void tts_abort(const char * file, int line, const char * fmt, ...);
