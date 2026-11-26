// pool.h: Vahana's own thread team (P3), replacing OpenMP so the kernel builds the same way with GCC, Clang and
// MSVC on Linux, Windows and macOS. Which thread computes a row never changes the row's arithmetic, so the team
// only affects speed, never bits.
//
//   vh_parallel([&](int tid, int nthr) { ... vh_barrier(); ... });   // the caller is tid 0
//
// Team size: VAHANA_THREADS, else OMP_NUM_THREADS, else the hardware thread count.
// Waiting: workers spin for VAHANA_SPIN_US microseconds (default from GOMP_SPINCOUNT / 1000, else 300), then sleep
// on the generation word itself (C++20 atomic wait: futex / WaitOnAddress), so a wake-up never queues on a mutex.
// Binding (Linux): with VAHANA_BIND=1 or OMP_PROC_BIND=true, thread t runs on the t-th core of VAHANA_CPUS (else of
// the online CPUs) (all hyperthreads of that core), like OMP_PLACES=cores. Core 0 goes to the kernel's own job thread
// (vh_mark_job_thread); other callers (the Python main thread) are never pinned, since torch's own OpenMP team would
// inherit the pin.
// Nested or concurrent calls: a call from inside a team task runs serially; calls from two threads take turns.
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <cstdio>
#include <algorithm>
#include <vector>
#if defined(__linux__)
#include <sched.h>
#include <pthread.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define VH_RELAX() _mm_pause()
#elif defined(__aarch64__) || defined(_M_ARM64)
#define VH_RELAX() std::this_thread::yield()
#else
#define VH_RELAX() ((void)0)
#endif

// The CPU this thread runs on (Linux), else 0. Used only to pick a node-local task list first.
inline int vh_cpu() {
#if defined(__linux__)
  return sched_getcpu();
#else
  return 0;
#endif
}

namespace vhp {
struct Task { virtual void run(int tid, int n) = 0; virtual ~Task() {} };
template <typename F> struct TaskF : Task { F& f; explicit TaskF(F& f_) : f(f_) {} void run(int t, int n) override { f(t, n); } };

inline int env_int(const char* a, const char* b, int def) {
  for (const char* k : {a, b}) { if (!k) continue; const char* v = std::getenv(k); if (v && std::atoi(v) > 0) return std::atoi(v); }
  return def;
}

class Pool {
 public:
  static Pool& get() { static Pool* p = new Pool(); return *p; }      // never destroyed: no exit-time join
  int size() const { return n_; }

  template <typename F> void run(F& f) {
    if (in_team_) { f(0, 1); return; }                                 // nested: serial, same results
    std::lock_guard<std::mutex> turn(run_mu_);
    if (n_ == 1) { f(0, 1); return; }
    bind_caller();
    TaskF<F> t(f);
    task_ = &t;
    err_ = nullptr;
    left_.store(n_ - 1, std::memory_order_relaxed);
    gen_.fetch_add(1, std::memory_order_seq_cst);
    if (sleepers_.load(std::memory_order_seq_cst) > 0) gen_.notify_all();
    in_team_ = true;
    std::exception_ptr mine = nullptr;
    try { f(0, n_); } catch (...) { mine = std::current_exception(); }
    in_team_ = false;
    wait_until([&] { return left_.load(std::memory_order_acquire) == 0; });
    task_ = nullptr;
    if (mine) std::rethrow_exception(mine);
    if (err_) std::rethrow_exception(err_);
  }
  // Every team member must call it the same number of times (an OpenMP barrier).
  void barrier() {
    if (n_ == 1 || !in_team_) return;
    const int g = bar_gen_.load(std::memory_order_acquire);
    if (bar_cnt_.fetch_add(1, std::memory_order_acq_rel) + 1 == n_) {
      bar_cnt_.store(0, std::memory_order_relaxed);
      bar_gen_.fetch_add(1, std::memory_order_release);
    } else {
      wait_until([&] { return bar_gen_.load(std::memory_order_acquire) != g; });
    }
  }

 private:
  Pool() {
    n_ = env_int("VAHANA_THREADS", "OMP_NUM_THREADS", (int)std::max(1u, std::thread::hardware_concurrency()));
    const char* su = std::getenv("VAHANA_SPIN_US");
    const char* gs = std::getenv("GOMP_SPINCOUNT");
    spin_us_ = su ? std::atol(su) : (gs ? std::atol(gs) / 1000 : 300);
    const char* b1 = std::getenv("VAHANA_BIND"); const char* b2 = std::getenv("OMP_PROC_BIND");
    bind_ = (b1 && b1[0] == '1') || (b2 && (std::strcmp(b2, "true") == 0 || std::strcmp(b2, "TRUE") == 0 ||
                                            std::strcmp(b2, "close") == 0 || std::strcmp(b2, "spread") == 0));
    places();
    for (int t = 1; t < n_; t++) std::thread([this, t] { worker(t); }).detach();
  }
  template <typename P> void wait_until(P done) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; !done(); i++) {
      VH_RELAX();
      if ((i & 255) == 255 && std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(spin_us_ + 50))
        std::this_thread::yield();
    }
  }
  void worker(int tid) {
    bind_self(tid);
    uint32_t seen = 0;
    for (;;) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; gen_.load(std::memory_order_acquire) == seen; i++) {
        VH_RELAX();
        if ((i & 31) == 31 && std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(spin_us_)) {
          sleepers_.fetch_add(1, std::memory_order_seq_cst);
          while (gen_.load(std::memory_order_seq_cst) == seen) gen_.wait(seen, std::memory_order_seq_cst);
          sleepers_.fetch_sub(1, std::memory_order_seq_cst);
          break;
        }
      }
      seen = gen_.load(std::memory_order_acquire);
      in_team_ = true;
      try { task_->run(tid, n_); } catch (...) { std::lock_guard<std::mutex> l(err_mu_); if (!err_) err_ = std::current_exception(); }
      in_team_ = false;
      left_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }
  // The CPUs Vahana may use (Linux). Not the calling thread's mask: an OpenMP runtime with OMP_PROC_BIND (torch's
  // libgomp) pins the main thread to one core at import, and threads created later would inherit that. Source:
  // VAHANA_CPUS ("0-24,32-56", set by the CLI before torch loads), else every online CPU.
  static void parse_list(const char* buf, std::vector<int>& out) {
    for (const char* s = buf; s && *s;) {
      char* e; long a = std::strtol(s, &e, 10); long b = a;
      if (e == s) break;
      if (*e == '-') { const char* t = e + 1; b = std::strtol(t, &e, 10); }
      for (long x = a; x <= b; x++) out.push_back((int)x);
      if (*e != ',') break;
      s = e + 1;
    }
  }
  void places() {
#if defined(__linux__)
    std::vector<int> cpus;
    if (const char* v = std::getenv("VAHANA_CPUS")) parse_list(v, cpus);
    if (cpus.empty()) {
      if (FILE* f = std::fopen("/sys/devices/system/cpu/online", "r")) {
        char buf[512] = {0}; if (std::fgets(buf, sizeof(buf), f)) parse_list(buf, cpus); std::fclose(f);
      }
    }
    std::vector<int> ok;
    for (int c : cpus) if (c >= 0 && c < CPU_SETSIZE) ok.push_back(c);
    allowed_ = ok;
    if (ok.empty()) { bind_ = false; return; }
    std::vector<bool> in(CPU_SETSIZE, false), used(CPU_SETSIZE, false);
    for (int c : ok) in[c] = true;
    for (int c : ok) {                                    // one place per core: its hyperthreads among `ok`
      if (used[c]) continue;
      std::vector<int> pl{c}, sib;
      char path[128]; std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
      if (FILE* f = std::fopen(path, "r")) {
        char buf[256] = {0}; if (std::fgets(buf, sizeof(buf), f)) parse_list(buf, sib); std::fclose(f);
      }
      for (int x : sib) if (x != c && x >= 0 && x < CPU_SETSIZE && in[x]) pl.push_back(x);
      for (int x : pl) used[x] = true;
      places_.push_back(pl);
    }
#endif
  }
  // Workers (and the kernel's job thread) always set their mask: one place each when binding, else every allowed CPU.
  void bind_self(int tid) {
#if defined(__linux__)
    if (allowed_.empty()) return;
    cpu_set_t s; CPU_ZERO(&s);
    if (bind_ && !places_.empty()) { for (int c : places_[tid % places_.size()]) CPU_SET(c, &s); }
    else { for (int c : allowed_) CPU_SET(c, &s); }
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
#else
    (void)tid;
#endif
  }
  void bind_caller() {
    thread_local bool bound = false;
    if (!bound && job_thread_) { bind_self(0); bound = true; }
  }

 public:
  static thread_local bool job_thread_;

 private:

  int n_ = 1;
  long spin_us_ = 300;
  bool bind_ = false;
  std::vector<std::vector<int>> places_;
  std::vector<int> allowed_;
  std::mutex run_mu_, err_mu_;
  // each word on its own cache line: workers spin on gen_ / bar_gen_ while others count down left_ / bar_cnt_
  alignas(64) std::atomic<uint32_t> gen_{0};
  alignas(64) std::atomic<int> left_{0};
  alignas(64) std::atomic<int> sleepers_{0};
  alignas(64) std::atomic<int> bar_cnt_{0};
  alignas(64) std::atomic<int> bar_gen_{0};
  alignas(64) char pad_[64] = {0};
  Task* task_ = nullptr;
  std::exception_ptr err_ = nullptr;
  static thread_local bool in_team_;
};
inline thread_local bool Pool::in_team_ = false;
inline thread_local bool Pool::job_thread_ = false;
}  // namespace vhp

template <typename F> inline void vh_parallel(F&& f) { auto& g = f; vhp::Pool::get().run(g); }
inline void vh_barrier() { vhp::Pool::get().barrier(); }
inline int vh_nthreads() { return vhp::Pool::get().size(); }
inline void vh_mark_job_thread() { vhp::Pool::job_thread_ = true; }
