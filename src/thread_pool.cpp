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
constexpr int kSpins = 200000;  // ~tens of microseconds before a worker parks on the condvar
}  // namespace

ThreadPool::ThreadPool(int threads) : n_(std::max(1, threads)) {
  for (int i = 1; i < n_; ++i) threads_.emplace_back([this, i] { worker_loop(i); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_.store(true);
    gen_.fetch_add(1);
  }
  cv_.notify_all();
  for (auto& t : threads_) t.join();
}

void ThreadPool::run_chunks(Job& j, int tid) {
  for (;;) {
    const int64_t b = j.next.fetch_add(j.grain, std::memory_order_relaxed);
    if (b >= j.total) return;
    const int64_t e = std::min(j.total, b + j.grain);
    j.task(j.ctx, b, e, tid);
    j.done.fetch_add(e - b, std::memory_order_release);
  }
}

void ThreadPool::dispatch(Task task, void* ctx, int64_t n, int64_t grain) {
  const uint64_t g = gen_.load(std::memory_order_relaxed) + 1;  // only this thread publishes
  Job& j = jobs_[g & 1];
  // Retire the slot (used two jobs ago): mark it invalid first, then wait out any straggler that
  // joined before the mark. A worker that joins after it sees kInvalid and backs off (seq_cst on
  // both sides makes these two orders the only possibilities).
  j.gen.store(kInvalid);
  while (j.refs.load() != 0) cpu_relax();
  j.task = task;
  j.ctx = ctx;
  j.total = n;
  j.grain = std::max<int64_t>(1, grain);
  j.next.store(0, std::memory_order_relaxed);
  j.done.store(0, std::memory_order_relaxed);
  j.gen.store(g);
  gen_.store(g);
  if (sleepers_.load() > 0) {
    std::lock_guard<std::mutex> lk(mu_);
    cv_.notify_all();
  }
  run_chunks(j, 0);
  while (j.done.load(std::memory_order_acquire) < n) cpu_relax();
}

void ThreadPool::worker_loop(int tid) {
  uint64_t seen = 0;
  for (;;) {
    int spins = 0;
    while (gen_.load() == seen && spins < kSpins) {
      cpu_relax();
      ++spins;
    }
    if (gen_.load() == seen) {
      std::unique_lock<std::mutex> lk(mu_);
      sleepers_.fetch_add(1);
      cv_.wait(lk, [&] { return gen_.load() != seen || stop_.load(); });
      sleepers_.fetch_sub(1);
    }
    if (stop_.load()) return;
    const uint64_t g = gen_.load();
    seen = g;
    Job& j = jobs_[g & 1];
    j.refs.fetch_add(1);
    if (j.gen.load() == g) run_chunks(j, tid);  // else: slot already retired, skip this job
    j.refs.fetch_sub(1);
  }
}

}  // namespace edgelm
