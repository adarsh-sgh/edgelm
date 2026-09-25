// Persistent pool for per-op parallel_for. Workers spin briefly between dispatches (decode issues
// a few hundred small ops per token, so a condvar wake per op would dominate), then sleep.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace edgelm {

class ThreadPool {
 public:
  explicit ThreadPool(int threads);  // total, including the calling thread
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return n_; }

  // fn(begin, end, tid) over [0, n) in chunks of `grain`; tid in [0, size()). Dynamic chunking,
  // so fast (P) cores take more chunks than slow (E) cores.
  template <class F>
  void parallel_for(int64_t n, int64_t grain, F&& fn) {
    if (n <= 0) return;
    if (n_ == 1 || n <= grain) {
      fn(int64_t{0}, n, 0);
      return;
    }
    using Fn = std::remove_reference_t<F>;
    auto tramp = [](void* ctx, int64_t b, int64_t e, int tid) { (*static_cast<Fn*>(ctx))(b, e, tid); };
    dispatch(tramp, const_cast<void*>(static_cast<const void*>(&fn)), n, grain);
  }

 private:
  using Task = void (*)(void*, int64_t, int64_t, int);
  void dispatch(Task task, void* ctx, int64_t n, int64_t grain);
  void work(int tid);
  void worker_loop(int tid);

  int n_;
  std::vector<std::thread> threads_;
  Task task_ = nullptr;
  void* ctx_ = nullptr;
  int64_t total_ = 0, grain_ = 1;
  std::atomic<int64_t> next_{0};
  std::atomic<int> pending_{0};
  std::atomic<uint64_t> gen_{0};
  std::atomic<int> sleepers_{0};
  std::atomic<bool> stop_{false};
  std::mutex mu_;
  std::condition_variable cv_;
};

}  // namespace edgelm
