#include "edgelm/thread_pool.hpp"

#include <algorithm>

namespace edgelm {
namespace {
inline void cpu_relax() {
#if defined(__aarch64__) || defined(__arm__)
  __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#endif
}
}  // namespace

ThreadPool::ThreadPool(int threads) : n_(std::max(1, threads)) {
  for (int i = 1; i < n_; ++i) threads_.emplace_back([this, i] { worker_loop(i); });
}

ThreadPool::~ThreadPool() {
  stop_.store(true);
  {
    std::lock_guard<std::mutex> lk(mu_);
    gen_.fetch_add(1);
  }
  cv_.notify_all();
  for (auto& t : threads_) t.join();
}

void ThreadPool::work(int tid) {
  for (;;) {
    const int64_t b = next_.fetch_add(grain_, std::memory_order_relaxed);
    if (b >= total_) break;
    task_(ctx_, b, std::min(total_, b + grain_), tid);
  }
}

void ThreadPool::dispatch(Task task, void* ctx, int64_t n, int64_t grain) {
  task_ = task;
  ctx_ = ctx;
  total_ = n;
  grain_ = std::max<int64_t>(1, grain);
  next_.store(0, std::memory_order_relaxed);
  pending_.store(n_ - 1, std::memory_order_relaxed);
  gen_.fetch_add(1);  // seq_cst: publishes the fields above; pairs with sleepers_ below
  if (sleepers_.load() > 0) {
    std::lock_guard<std::mutex> lk(mu_);
    cv_.notify_all();
  }
  work(0);
  while (pending_.load(std::memory_order_acquire) > 0) cpu_relax();
}

void ThreadPool::worker_loop(int tid) {
  uint64_t seen = 0;
  for (;;) {
    int spins = 0;
    while (gen_.load() == seen && spins < 200000) {
      cpu_relax();
      ++spins;
    }
    if (gen_.load() == seen) {
      std::unique_lock<std::mutex> lk(mu_);
      sleepers_.fetch_add(1);
      cv_.wait(lk, [&] { return gen_.load() != seen; });
      sleepers_.fetch_sub(1);
    }
    seen = gen_.load();
    if (stop_.load()) return;
    work(tid);
    pending_.fetch_sub(1, std::memory_order_release);
  }
}

}  // namespace edgelm
