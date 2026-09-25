// Persistent pool for per-op parallel_for. Decode issues ~200 small parallel ops per token, so:
//  * workers spin briefly between jobs before sleeping (a condvar wake per op would dominate);
//  * a job is finished when all of its items are done, not when every worker has checked in, so
//    a descheduled or slow (E-core) worker never stalls the caller. Stragglers that wake late find
//    the job drained and go back to waiting. Job slots are double-buffered and reference-counted
//    so a slot is only refilled once no straggler can still be reading it.
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

  // fn(begin, end, tid) over [0, n) in chunks of `grain`; tid in [0, size()) and unique per
  // thread, so it can index per-thread scratch. Dynamic chunking: fast cores take more chunks.
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
  struct alignas(64) Job {
    std::atomic<uint64_t> gen{0};  // job generation this slot holds; kInvalid while being refilled
    std::atomic<int> refs{0};      // workers currently inside this slot
    std::atomic<int64_t> next{0};  // next item to hand out
    std::atomic<int64_t> done{0};  // items finished
    Task task = nullptr;
    void* ctx = nullptr;
    int64_t total = 0, grain = 1;
  };
  static constexpr uint64_t kInvalid = ~uint64_t{0};

  void dispatch(Task task, void* ctx, int64_t n, int64_t grain);
  static void run_chunks(Job& j, int tid);
  void worker_loop(int tid);

  int n_;
  std::vector<std::thread> threads_;
  Job jobs_[2];
  std::atomic<uint64_t> gen_{0};  // latest published job
  std::atomic<int> sleepers_{0};
  std::atomic<bool> stop_{false};
  std::mutex mu_;
  std::condition_variable cv_;
};

}  // namespace edgelm
