#pragma once

// OSX threads other than the main thread are created with a reduced stack size of 512KB by default, this is too low 
// for large GGML graphs in which graph nodes are traversed recursively. To address this we instead use pthreads so that stack
// size can be increased in parity with linux.

#include <thread>

#if defined(__APPLE__)

#include <pthread.h>
#include <functional>

using namespace std;

namespace tts_server_threading {
	// The implementation calls pthread_create() with the stack size parameter equal to the Linux 8MB default, on platforms that support it.
	class native_thread {
	    pthread_t thread;
	    static constexpr size_t THREAD_STACK_SIZE = 8 * 1024 * 1024;
	public:
		native_thread() = default;
		native_thread(const native_thread&) = delete;
	    template<class Function, class... Args>
	    explicit native_thread(Function&& fun, Args&&... args) {
	        auto func = new function<void()>(
	          std::bind(std::forward<Function>(fun), std::forward<Args>(args)...));

	        pthread_attr_t attr_storage, *attr = &attr_storage;
	        pthread_attr_init(attr);
	        pthread_attr_setstacksize(attr, THREAD_STACK_SIZE);

	        auto start_routine = [](void* ptr) -> void* {
	            auto f = reinterpret_cast<function<void()>*>(ptr);
	            // Call the function
	            (*f)();
	            delete f;
	            return nullptr;
	        };

	        pthread_create(&thread, attr, start_routine, func);
	    }

	    void join() { pthread_join(thread, nullptr); }
	};
}

#else

namespace tts_server_threading {
	using native_thread = std::thread;
}

#endif
