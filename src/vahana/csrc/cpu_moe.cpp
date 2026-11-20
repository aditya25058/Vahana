#include <cstdio>
#include <string>
// Fused, NUMA-aware, fixed-order CPU expert MLPs (Phase 25b).
//
// y_e = down_e( silu(gate_e x) * up_e x ) for E experts, each with n tokens (n <= 16), weights read
// IN PLACE (page cache, BF16). One OpenMP region per call: phase 1 = (expert, 16-row block) tasks for
// gate+up (+ silu*mul), barrier, phase 2 = (expert, 16-row block) tasks for down.
//
// Reduction order is fully specified so a GPU twin can reproduce the bits (device invariance):
//   row dot over K: 16 fp32 lanes, lane l accumulates k = 16*j + l in increasing j with IEEE fp32 FMA
//   (fma(w, x, acc)), then lanes reduced by a fixed tree: for s = 8,4,2,1: a[i] = a[i] + a[i+s] (i < s).
//   Rounding to bf16 is round-to-nearest-even at the same points as the stock module:
//   g = bf16(dot), u = bf16(dot), s = bf16(silu_f32(g)), h = bf16(s * u), y = bf16(dot(down, h)).
// Each output row is computed entirely by one thread, so results are independent of thread count,
// NUMA placement and of which other tokens/experts share the call (batch invariance).
#include <torch/extension.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <omp.h>
#if defined(__linux__) && defined(__has_include)
#if __has_include(<numaif.h>)
#include <numaif.h>
#define VH_HAVE_NUMA 1
#endif
#endif
#ifndef VH_HAVE_NUMA
#define VH_HAVE_NUMA 0
#define MPOL_MF_MOVE 2
// No NUMA API on this platform: nothing is migrated, every page reports "unknown node" (-1).
static inline long move_pages(int, unsigned long count, void** pages, const int*, int* status, int) {
  (void)pages;
  for (unsigned long i = 0; i < count; i++) status[i] = -1;
  return -1;
}
#endif
#include <sched.h>
#include <cstring>
#include <cmath>
#include <atomic>
#include <vector>
#include <algorithm>

static inline float bf2f(uint16_t h) { uint32_t u = ((uint32_t)h) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static inline uint16_t f2bf(float f) {
  uint32_t u; std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
// ---- V16: 16 fp32 lanes; lane l of a dot product accumulates k = 16j + l in ascending j with a fused multiply-add.
// Every ISA below performs exactly that per-lane sequence, so every build gives the same bits.
#if defined(__AVX512F__)
#define VH_ISA "avx512"
typedef __m512 V16;
static inline V16 v_zero() { return _mm512_setzero_ps(); }
static inline V16 v_loadf(const float* p) { return _mm512_loadu_ps(p); }
static inline V16 v_ldbf16(const uint16_t* p) {
  __m256i v = _mm256_loadu_si256((const __m256i*)p);
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v), 16));
}
static inline V16 v_fma(V16 a, V16 b, V16 c) { return _mm512_fmadd_ps(a, b, c); }
static inline void v_store(float* a, V16 v) { _mm512_storeu_ps(a, v); }
#elif defined(__AVX2__) && defined(__FMA__)
#define VH_ISA "avx2"
struct V16 { __m256 lo, hi; };                      // lanes 0-7, 8-15
static inline V16 v_zero() { return {_mm256_setzero_ps(), _mm256_setzero_ps()}; }
static inline V16 v_loadf(const float* p) { return {_mm256_loadu_ps(p), _mm256_loadu_ps(p + 8)}; }
static inline __m256 bf8(__m128i h) { return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16)); }
static inline V16 v_ldbf16(const uint16_t* p) {
  return {bf8(_mm_loadu_si128((const __m128i*)p)), bf8(_mm_loadu_si128((const __m128i*)(p + 8)))};
}
static inline V16 v_fma(V16 a, V16 b, V16 c) { return {_mm256_fmadd_ps(a.lo, b.lo, c.lo), _mm256_fmadd_ps(a.hi, b.hi, c.hi)}; }
static inline void v_store(float* a, V16 v) { _mm256_storeu_ps(a, v.lo); _mm256_storeu_ps(a + 8, v.hi); }
#elif defined(__ARM_NEON) || defined(__aarch64__)
#define VH_ISA "neon"
struct V16 { float32x4_t q[4]; };
static inline V16 v_zero() { V16 r; for (int i = 0; i < 4; i++) r.q[i] = vdupq_n_f32(0.0f); return r; }
static inline V16 v_loadf(const float* p) { V16 r; for (int i = 0; i < 4; i++) r.q[i] = vld1q_f32(p + 4 * i); return r; }
static inline V16 v_ldbf16(const uint16_t* p) {
  V16 r;
  for (int i = 0; i < 4; i++) r.q[i] = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p + 4 * i), 16));
  return r;
}
static inline V16 v_fma(V16 a, V16 b, V16 c) { V16 r; for (int i = 0; i < 4; i++) r.q[i] = vfmaq_f32(c.q[i], a.q[i], b.q[i]); return r; }
static inline void v_store(float* a, V16 v) { for (int i = 0; i < 4; i++) vst1q_f32(a + 4 * i, v.q[i]); }
#else
#define VH_ISA "scalar"
struct V16 { float v[16]; };
static inline V16 v_zero() { V16 r; for (int i = 0; i < 16; i++) r.v[i] = 0.0f; return r; }
static inline V16 v_loadf(const float* p) { V16 r; std::memcpy(r.v, p, 64); return r; }
static inline V16 v_ldbf16(const uint16_t* p) { V16 r; for (int i = 0; i < 16; i++) r.v[i] = bf2f(p[i]); return r; }
static inline V16 v_fma(V16 a, V16 b, V16 c) { V16 r; for (int i = 0; i < 16; i++) r.v[i] = std::fma(a.v[i], b.v[i], c.v[i]); return r; }
static inline void v_store(float* a, V16 v) { std::memcpy(a, v.v, 64); }
#endif
static inline float tree16(V16 v) {
  float a[16]; v_store(a, v);
  for (int s = 8; s > 0; s >>= 1) for (int i = 0; i < s; i++) a[i] = a[i] + a[i + s];
  return a[0];
}
#if defined(__x86_64__) || defined(_M_X64)
#define VH_PAUSE() _mm_pause()
#elif defined(__aarch64__)
#define VH_PAUSE() __asm__ __volatile__("yield")
#else
#define VH_PAUSE() ((void)0)
#endif
#if defined(__GNUC__) || defined(__clang__)
#define VH_PREFETCH(p) __builtin_prefetch((const void*)(p))
#else
#define VH_PREFETCH(p) ((void)0)
#endif
// dots of one weight row (K bf16) with n fp32 vectors X[t*K ..]
static inline void row_dots(const uint16_t* w, const float* X, int n, int K, float* out) {
  V16 acc[16];
  for (int t = 0; t < n; t++) acc[t] = v_zero();
  for (int k = 0; k < K; k += 16) {
    V16 wv = v_ldbf16(w + k);
    for (int t = 0; t < n; t++) acc[t] = v_fma(wv, v_loadf(X + (size_t)t * K + k), acc[t]);
  }
  for (int t = 0; t < n; t++) out[t] = tree16(acc[t]);
}

// Device-invariant exp: identical bits on CPU and GPU (the Triton twin implements the same ops).
// x clamped to [-87, 87]; n = floor(fma(x, log2e, 0.5)); r = fma(-n, LN2_HI, x); r = fma(-n, LN2_LO, r);
// p = Horner(r) with explicit fmas; result = p * 2^n (exact scaling via exponent bits).
static inline float dexp(float x) {
  x = std::fmin(std::fmax(x, -87.0f), 87.0f);
  float n = std::floor(std::fma(x, 0x1.715476p+0f, 0.5f));
  float r = std::fma(-n, 0x1.63p-1f, x);
  r = std::fma(-n, -0x1.bd0106p-13f, r);
  float p = 0x1.a0d2cep-13f;
  p = std::fma(p, r, 0x1.6e878ap-10f);
  p = std::fma(p, r, 0x1.111210p-7f);
  p = std::fma(p, r, 0x1.555382p-5f);
  p = std::fma(p, r, 0x1.555556p-3f);
  p = std::fma(p, r, 0x1.0p-1f);
  p = std::fma(p, r * r, r);
  p = p + 1.0f;
  uint32_t bits = (uint32_t)((int)n + 127) << 23; float sc; std::memcpy(&sc, &bits, 4);
  return p * sc;
}

#include <chrono>
static double g_ph[5] = {0, 0, 0, 0, 0};  // setup, pass0, pass1, tail, calls
static inline double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
// n == 1: four independent rows at once (same per-row arithmetic as row_dots).
static inline void row_dots4_n1(const uint16_t* w0, const uint16_t* w1, const uint16_t* w2,
                                const uint16_t* w3, const float* X, int K, float* out) {
  V16 a0 = v_zero(), a1 = v_zero(), a2 = v_zero(), a3 = v_zero();
  for (int k = 0; k < K; k += 16) {
    V16 xv = v_loadf(X + k);
    a0 = v_fma(v_ldbf16(w0 + k), xv, a0);
    a1 = v_fma(v_ldbf16(w1 + k), xv, a1);
    a2 = v_fma(v_ldbf16(w2 + k), xv, a2);
    a3 = v_fma(v_ldbf16(w3 + k), xv, a3);
  }
  out[0] = tree16(a0); out[1] = tree16(a1); out[2] = tree16(a2); out[3] = tree16(a3);
}
static int node_of(const void* p) {
  void* page = (void*)((uintptr_t)p & ~(uintptr_t)4095);
  int status = -1;
  if (move_pages(0, 1, &page, nullptr, &status, 0) != 0) return -1;
  return status;
}

// x: [E, n, H] bf16; gate/up: E tensors [I, H]; down: E tensors [H, I]. Returns y [E, n, H] bf16
// and the node of each expert. numa=1: tasks run on threads of the expert's node.
// M2: per-expert activation / bias context (index j = the expert's position in the call).
// nullptr or empty = SiLU-GLU without biases: exactly the original arithmetic.
struct ActCtx { std::vector<int> act; std::vector<const float*> bg, bu, bd; float alpha = 1.702f, limit = 7.0f; };
static inline float act_h(const ActCtx* c, int j, int r, float gdot, float udot) {
  if (!c || c->act.empty()) {
    float gb = bf2f(f2bf(gdot)), ub = bf2f(f2bf(udot));
    float s = bf2f(f2bf(gb / (1.0f + dexp(-gb))));
    return bf2f(f2bf(s * ub));
  }
  float gb = bf2f(f2bf(gdot)), ub = bf2f(f2bf(udot));
  if (c->bg[j]) gb = bf2f(f2bf(gb + c->bg[j][r]));
  if (c->bu[j]) ub = bf2f(f2bf(ub + c->bu[j][r]));
  if (c->act[j] == 1) {                       // gpt-oss: clamp, g * sigmoid(alpha g), (u + 1) * glu
    gb = std::fmin(gb, c->limit);
    ub = std::fmin(std::fmax(ub, -c->limit), c->limit);
    const float z = bf2f(f2bf(gb * c->alpha));
    const float sg = bf2f(f2bf(1.0f / (1.0f + dexp(-z))));
    const float glu = bf2f(f2bf(gb * sg));
    return bf2f(f2bf(bf2f(f2bf(ub + 1.0f)) * glu));
  }
  float s = bf2f(f2bf(gb / (1.0f + dexp(-gb))));
  return bf2f(f2bf(s * ub));
}
static inline uint16_t down_o(const ActCtx* c, int j, int r, float dot) {
  if (!c || c->act.empty() || !c->bd[j]) return f2bf(dot);
  return f2bf(bf2f(f2bf(dot)) + c->bd[j][r]);
}
std::vector<torch::Tensor> expert_mlps_ctx(torch::Tensor x, std::vector<torch::Tensor> gate,
                                       std::vector<torch::Tensor> up, std::vector<torch::Tensor> down,
                                       int64_t numa, const ActCtx* ctx) {
  const double T0 = now_ms();
  // The team size is fixed from OMP_NUM_THREADS at first use: torch ops on this thread (or
  // torch.set_num_threads on the main thread) must not shrink it.
  static const int nthr = [] { const char* v = std::getenv("OMP_NUM_THREADS"); int t = v ? std::atoi(v) : 0;
                               return t > 0 ? t : omp_get_num_procs(); }();
  omp_set_num_threads(nthr);
  const int E = x.size(0), n = x.size(1), H = x.size(2), I = gate[0].size(0);
  TORCH_CHECK(n <= 16 && H % 16 == 0 && I % 16 == 0);
  std::vector<float> xf((size_t)E * n * H), hf((size_t)E * n * I);
  const uint16_t* xp = (const uint16_t*)x.data_ptr();
  for (size_t i = 0; i < xf.size(); i++) xf[i] = bf2f(xp[i]);
  auto y = torch::empty({E, n, H}, x.options());
  uint16_t* yp = (uint16_t*)y.data_ptr();
  std::vector<int> enode(E, 0);
  if (numa == 1) for (int e = 0; e < E; e++) { int nd = node_of(gate[e].data_ptr()); enode[e] = nd < 0 ? 0 : nd; }
  // task lists per node: phase 1 over I/16 blocks, phase 2 over H/16 blocks.
  // numa=2 (Phase 34, row split): rows [0, R/2) of every matrix belong to node 0 and [R/2, R) to node 1
  // (place_rows() migrates the pages to match). Only the thread that computes a row changes, never its arithmetic.
  std::vector<int> t1[2], t2[2];
  for (int e = 0; e < E; e++) {
    int nd = numa == 1 ? (enode[e] & 1) : 0;
    for (int b = 0; b < I / 16; b++) t1[numa == 2 ? (b >= I / 32) : nd].push_back(e * (I / 16) + b);
    for (int b = 0; b < H / 16; b++) t2[numa == 2 ? (b >= H / 32) : nd].push_back(e * (H / 16) + b);
  }
  // numa=1: local-first work stealing. Each thread drains its own node's task list through an
  // atomic cursor, then helps the other node's list. Which thread computes a row never changes the
  // row's arithmetic, so results are identical in every mode.
  std::atomic<size_t> cur[2];
  double T1 = now_ms(), Tp0 = 0.0;
  #pragma omp parallel
  {
    int nd = numa ? ((sched_getcpu() >= 32) ? 1 : 0) : 0;
    float d[16], g[16], u[16];
    for (int pass = 0; pass < 2; pass++) {
      #pragma omp single
      { cur[0] = 0; cur[1] = 0; }
      for (int hop = 0; hop < (numa ? 2 : 1); hop++) {
        int Ls = numa ? (nd ^ hop) : 0;
        const std::vector<int>& T = (pass == 0) ? t1[Ls] : t2[Ls];
        const int blocks = (pass == 0 ? I : H) / 16;
        for (size_t i = cur[Ls].fetch_add(1); i < T.size(); i = cur[Ls].fetch_add(1)) {
          int e = T[i] / blocks, b = T[i] % blocks;
          if (n == 1) {
            float o[4];
            if (pass == 0) {
              const uint16_t* G = (const uint16_t*)gate[e].data_ptr();
              const uint16_t* U = (const uint16_t*)up[e].data_ptr();
              const float* xe = &xf[(size_t)e * H];
              for (int r = b * 16; r < b * 16 + 16; r += 2) {
                row_dots4_n1(G + (size_t)r * H, G + (size_t)(r + 1) * H, U + (size_t)r * H, U + (size_t)(r + 1) * H, xe, H, o);
                for (int q = 0; q < 2; q++) {
                  hf[(size_t)e * I + r + q] = act_h(ctx, e, r + q, o[q], o[2 + q]);
                }
              }
            } else {
              const uint16_t* Dw = (const uint16_t*)down[e].data_ptr();
              const float* he = &hf[(size_t)e * I];
              for (int r = b * 16; r < b * 16 + 16; r += 4) {
                row_dots4_n1(Dw + (size_t)r * I, Dw + (size_t)(r + 1) * I, Dw + (size_t)(r + 2) * I, Dw + (size_t)(r + 3) * I, he, I, o);
                for (int q = 0; q < 4; q++) yp[(size_t)e * H + r + q] = down_o(ctx, e, r + q, o[q]);
              }
            }
            continue;
          }
          for (int r = b * 16; r < b * 16 + 16; r++) {
            if (pass == 0) {
              row_dots((const uint16_t*)gate[e].data_ptr() + (size_t)r * H, &xf[(size_t)e * n * H], n, H, g);
              row_dots((const uint16_t*)up[e].data_ptr() + (size_t)r * H, &xf[(size_t)e * n * H], n, H, u);
              for (int t = 0; t < n; t++) {
                hf[((size_t)e * n + t) * I + r] = act_h(ctx, e, r, g[t], u[t]);
              }
            } else {
              row_dots((const uint16_t*)down[e].data_ptr() + (size_t)r * I, &hf[(size_t)e * n * I], n, I, d);
              for (int t = 0; t < n; t++) yp[((size_t)e * n + t) * H + r] = down_o(ctx, e, r, d[t]);
            }
          }
        }
      }
      #pragma omp barrier
      if (pass == 0) {
        #pragma omp master
        Tp0 = now_ms();
      }
    }
  }
  const double T2 = now_ms();
  auto nodes = torch::tensor(std::vector<int64_t>(enode.begin(), enode.end()));
  g_ph[0] += T1 - T0; g_ph[1] += Tp0 - T1; g_ph[2] += T2 - Tp0; g_ph[3] += now_ms() - T2; g_ph[4] += 1;
  return {y, nodes};
}
std::vector<torch::Tensor> expert_mlps(torch::Tensor x, std::vector<torch::Tensor> gate,
                                       std::vector<torch::Tensor> up, std::vector<torch::Tensor> down, int64_t numa) {
  return expert_mlps_ctx(x, gate, up, down, numa, nullptr);
}
std::vector<double> phase_ms() { return std::vector<double>(g_ph, g_ph + 5); }


// Phase 28: fused routed-MoE on the CPU worker for the all-CPU (h100) path.
// register_layer() keeps every expert's weight views per layer (no per-call Python packing).
// submit_moe(layer, x [N,H] bf16, ti [N,K] int64, tw [N,K] bf16, out [N,H] bf16 pinned) groups tokens by
// expert (ascending id), runs expert_mlps (same arithmetic as Phase 25/26), then combines each token's
// experts in ascending id order exactly as the GPU invariant combine does:
//   acc = f32(y_0) * f32(w_0);  acc = acc + f32(y_j) * f32(w_j)  (separate roundings, no FMA);  out = bf16(acc)
// Built with -ffp-contract=off, so the bits equal the GPU combine.
#include <algorithm>
#include <map>
namespace {
struct LayerW { std::vector<torch::Tensor> g, u, d; std::vector<torch::Tensor> sg, su, sd; std::vector<torch::Tensor> nx, snx;
                int act = 0; float alpha = 1.702f, limit = 7.0f; std::vector<torch::Tensor> bg, bu, bd; };
std::map<int64_t, LayerW>* LAYERS = new std::map<int64_t, LayerW>();
}
void register_layer(int64_t layer, std::vector<torch::Tensor> g, std::vector<torch::Tensor> u,
                    std::vector<torch::Tensor> d) {
  (*LAYERS)[layer] = LayerW{std::move(g), std::move(u), std::move(d), {}, {}, {}, {}, {}};
}
// M2: activation (0 SiLU-GLU, 1 gpt-oss clamped SwiGLU) and optional per-expert biases (lists may be empty).
void register_layer_act(int64_t layer, int64_t act, double alpha, double limit, std::vector<torch::Tensor> bg,
                        std::vector<torch::Tensor> bu, std::vector<torch::Tensor> bd) {
  LayerW& L = LAYERS->at(layer);
  L.act = (int)act; L.alpha = (float)alpha; L.limit = (float)limit;
  auto f32 = [](std::vector<torch::Tensor> v) { for (auto& t : v) t = t.to(torch::kFloat).contiguous(); return v; };
  L.bg = f32(bg); L.bu = f32(bu); L.bd = f32(bd);
}
static bool layer_has_ctx(const LayerW& L) { return L.act != 0 || !L.bg.empty() || !L.bu.empty() || !L.bd.empty(); }
static void build_ctx(const LayerW& L, const std::vector<int64_t>& ids, bool shared_first, ActCtx& c) {
  c.alpha = L.alpha; c.limit = L.limit;
  auto ptr = [](const std::vector<torch::Tensor>& v, int64_t e) -> const float* {
    return v.empty() ? nullptr : v[e].data_ptr<float>(); };
  if (shared_first) { c.act.push_back(0); c.bg.push_back(nullptr); c.bu.push_back(nullptr); c.bd.push_back(nullptr); }
  for (int64_t e : ids) { c.act.push_back(L.act); c.bg.push_back(ptr(L.bg, e)); c.bu.push_back(ptr(L.bu, e)); c.bd.push_back(ptr(L.bd, e)); }
}
// Phase 30: shared experts of a layer, computed in the same job and added (bf16 add) to the routed sum.
void register_shared(int64_t layer, torch::Tensor g, torch::Tensor u, torch::Tensor d) {
  LayerW& L = LAYERS->at(layer);
  L.sg = {g}; L.su = {u}; L.sd = {d};
}
torch::Tensor expert_mlps_nx(torch::Tensor x, std::vector<torch::Tensor> packs, int64_t I_, int64_t numa);
torch::Tensor expert_mlps_nx_ctx(torch::Tensor x, std::vector<torch::Tensor> packs, int64_t I_, int64_t numa,
                                 const ActCtx* ctx);
static void experts_nx_fused(const std::vector<const torch::Tensor*>& P, const std::vector<int>& Is, const float* xf,
                             int H, const std::vector<uint16_t*>& y, int numa, const ActCtx* ctx);
static std::vector<torch::Tensor> moe_fused(int64_t layer, torch::Tensor x, torch::Tensor ti, torch::Tensor tw,
                                            torch::Tensor out, int64_t numa) {
  const LayerW& L = LAYERS->at(layer);
  const int N = x.size(0), H = x.size(1), K = ti.size(1);
  const int64_t* tip = ti.data_ptr<int64_t>();
  const uint16_t* twp = (const uint16_t*)tw.data_ptr();
  const bool twf = tw.scalar_type() == torch::kFloat;          // M2: fp32 routing weights used as given
  const float* twfp = twf ? tw.data_ptr<float>() : nullptr;
  const uint16_t* xp = (const uint16_t*)x.data_ptr();
  static const bool fuse = [] { const char* v = std::getenv("VAHANA_FUSE"); return !(v && v[0] == '0'); }();
  if (fuse && N == 1 && !L.nx.empty() && (L.sg.empty() || !L.snx.empty()) && H % 64 == 0) {
    // Phase 41: routed + shared in one region; same per-row arithmetic and the same combine as below.
    std::vector<std::pair<int64_t, int>> ek;                  // (expert, k) ascending expert id
    for (int k = 0; k < K; k++) ek.push_back({tip[k], k});
    std::sort(ek.begin(), ek.end());
    const bool sh = !L.sg.empty();
    const int E = K + (sh ? 1 : 0), I = L.g[0].size(0);
    std::vector<const torch::Tensor*> P; std::vector<int> Is;
    if (sh) { P.push_back(L.snx.data()); Is.push_back(L.sg[0].size(0)); }
    for (int q = 0; q < K; q++) { P.push_back(&L.nx[15 * ek[q].first]); Is.push_back(I); }
    thread_local std::vector<float> xf; thread_local std::vector<uint16_t> yb;
    xf.resize(H); yb.resize((size_t)E * H);
    for (int h = 0; h < H; h++) xf[h] = bf2f(xp[h]);
    std::vector<uint16_t*> yv(E);
    for (int j = 0; j < E; j++) yv[j] = yb.data() + (size_t)j * H;
    ActCtx fc; const ActCtx* fcp = nullptr;
    if (layer_has_ctx(L)) {
      std::vector<int64_t> v; for (int q = 0; q < K; q++) v.push_back(ek[q].first);
      build_ctx(L, v, sh, fc); fcp = &fc;
    }
    experts_nx_fused(P, Is, xf.data(), H, yv, (int)numa, fcp);
    uint16_t* op = (uint16_t*)out.data_ptr();
    const int r0 = sh ? 1 : 0;
    std::vector<float> acc(H);
    for (int q = 0; q < K; q++) {
      const uint16_t* yr = yv[r0 + q];
      const float w = twf ? twfp[ek[q].second] : bf2f(twp[ek[q].second]);
      if (q == 0) { for (int h = 0; h < H; h++) acc[h] = bf2f(yr[h]) * w; }
      else { for (int h = 0; h < H; h++) { float p = bf2f(yr[h]) * w; acc[h] = acc[h] + p; } }
    }
    for (int h = 0; h < H; h++) op[h] = f2bf(acc[h]);
    if (sh) for (int h = 0; h < H; h++) op[h] = f2bf(bf2f(yv[0][h]) + bf2f(op[h]));
    return {out};
  }
  std::map<int64_t, std::vector<int>> rows;               // expert -> tokens (ascending)
  for (int b = 0; b < N; b++) for (int k = 0; k < K; k++) rows[tip[b * K + k]].push_back(b);
  const int E = (int)rows.size();
  int nmax = 0;
  for (auto& kv : rows) nmax = std::max(nmax, (int)kv.second.size());
  auto X = torch::zeros({E, nmax, H}, x.options());
  uint16_t* Xp = (uint16_t*)X.data_ptr();
  std::vector<torch::Tensor> g, u, d;
  std::vector<int64_t> ids;
  int j = 0;
  for (auto& kv : rows) {
    ids.push_back(kv.first);
    g.push_back(L.g[kv.first]); u.push_back(L.u[kv.first]); d.push_back(L.d[kv.first]);
    for (size_t t = 0; t < kv.second.size(); t++)
      std::memcpy(Xp + ((size_t)j * nmax + t) * H, xp + (size_t)kv.second[t] * H, (size_t)H * 2);
    j++;
  }
  torch::Tensor Y;
  ActCtx gc; const ActCtx* gcp = nullptr;
  if (layer_has_ctx(L)) { build_ctx(L, ids, false, gc); gcp = &gc; }
  if (nmax == 1 && !L.nx.empty()) {          // Phase 38: decode reads NX12 (lossless, same arithmetic)
    std::vector<torch::Tensor> pk;
    for (int64_t id : ids) for (int q = 0; q < 15; q++) pk.push_back(L.nx[15 * id + q]);
    Y = expert_mlps_nx_ctx(X, pk, L.g[0].size(0), numa, gcp);
  } else {
    Y = expert_mlps_ctx(X, g, u, d, numa, gcp)[0];
  }
  const uint16_t* Yp = (const uint16_t*)Y.data_ptr();
  // per token: its K (expert, weight, row-in-expert) in ascending expert id
  uint16_t* op = (uint16_t*)out.data_ptr();
  std::vector<float> acc(H);
  for (int b = 0; b < N; b++) {
    std::vector<std::pair<int64_t, int>> ek;
    for (int k = 0; k < K; k++) ek.push_back({tip[b * K + k], k});
    std::sort(ek.begin(), ek.end());
    for (int q = 0; q < K; q++) {
      int64_t e = ek[q].first;
      int ei = (int)(std::lower_bound(ids.begin(), ids.end(), e) - ids.begin());
      const std::vector<int>& rl = rows[e];
      int t = (int)(std::lower_bound(rl.begin(), rl.end(), b) - rl.begin());
      const uint16_t* yr = Yp + ((size_t)ei * nmax + t) * H;
      const float w = twf ? twfp[b * K + ek[q].second] : bf2f(twp[b * K + ek[q].second]);
      if (q == 0) { for (int h = 0; h < H; h++) acc[h] = bf2f(yr[h]) * w; }
      else { for (int h = 0; h < H; h++) { float p = bf2f(yr[h]) * w; acc[h] = acc[h] + p; } }
    }
    for (int h = 0; h < H; h++) op[(size_t)b * H + h] = f2bf(acc[h]);
  }
  if (!L.sg.empty()) {
    // shared experts (twin arithmetic) on the same tokens; out = bf16(f32(shared) + f32(routed))
    torch::Tensor Ys;
    if (N == 1 && !L.snx.empty()) Ys = expert_mlps_nx(x.view({1, 1, H}), L.snx, L.sg[0].size(0), numa);
    else Ys = expert_mlps(x.view({1, N, H}), L.sg, L.su, L.sd, numa)[0];
    const uint16_t* sp = (const uint16_t*)Ys.data_ptr();
    for (size_t i = 0; i < (size_t)N * H; i++) op[i] = f2bf(bf2f(sp[i]) + bf2f(op[i]));
  }
  return {out};
}

// Phase 30: twin linear on the CPU. y[t, r] = bf16( tree16( 16 IEEE-fma lanes over k ascending ) ),
// the same per-row arithmetic as the expert kernel (and the GPU twin). x [n, K] bf16, W [R, K] bf16.
#include <unordered_map>
struct NXm;
static std::unordered_map<const void*, std::vector<torch::Tensor>>* LIN_NX = new std::unordered_map<const void*, std::vector<torch::Tensor>>();
void register_linear_nx(torch::Tensor W, std::vector<torch::Tensor> packs) { (*LIN_NX)[W.data_ptr()] = std::move(packs); }
torch::Tensor linear_nx(const float* xf, const std::vector<torch::Tensor>& pk, int R, int K, torch::TensorOptions opt);
torch::Tensor linear_twin(torch::Tensor x, torch::Tensor W) {
  static const int nthr = [] { const char* v = std::getenv("OMP_NUM_THREADS"); int t = v ? std::atoi(v) : 0;
                               return t > 0 ? t : omp_get_num_procs(); }();
  const int n = x.size(0), K = x.size(1), R = W.size(0);
  TORCH_CHECK(K % 16 == 0 && R % 4 == 0 && n <= 16 && W.size(1) == K);
  std::vector<float> xf((size_t)n * K);
  const uint16_t* xp = (const uint16_t*)x.data_ptr();
  for (size_t i = 0; i < xf.size(); i++) xf[i] = bf2f(xp[i]);
  auto y = torch::empty({n, R}, x.options());
  uint16_t* yp = (uint16_t*)y.data_ptr();
  const uint16_t* w = (const uint16_t*)W.data_ptr();
  omp_set_num_threads(nthr);
  if (n == 1 && K % 64 == 0) {                 // Phase 40: NX12 copy registered for this weight
    auto it = LIN_NX->find(W.data_ptr());
    if (it != LIN_NX->end()) return linear_nx(xf.data(), it->second, R, K, x.options());
  }
  static const bool lin_numa = [] { const char* v = std::getenv("VAHANA_LIN_NUMA"); return v && v[0] == '2'; }();
  if (lin_numa && n == 1 && R % 8 == 0) {
    // Phase 40: rows [0, R/2) computed by node-0 threads, [R/2, R) by node-1 threads, local-first stealing.
    std::atomic<int> cur[2]; cur[0] = 0; cur[1] = R / 2;
    #pragma omp parallel
    {
      const int nd = (sched_getcpu() >= 32) ? 1 : 0;
      float o[4];
      for (int hop = 0; hop < 2; hop++) {
        const int Ls = nd ^ hop, end = Ls == 0 ? R / 2 : R;
        for (int r = cur[Ls].fetch_add(4); r < end; r = cur[Ls].fetch_add(4)) {
          row_dots4_n1(w + (size_t)r * K, w + (size_t)(r + 1) * K, w + (size_t)(r + 2) * K, w + (size_t)(r + 3) * K, xf.data(), K, o);
          for (int q = 0; q < 4; q++) yp[r + q] = f2bf(o[q]);
        }
      }
    }
    return y;
  }
  #pragma omp parallel for schedule(dynamic, 4)
  for (int r = 0; r < R; r += 4) {
    if (n == 1) {
      float o[4];
      row_dots4_n1(w + (size_t)r * K, w + (size_t)(r + 1) * K, w + (size_t)(r + 2) * K, w + (size_t)(r + 3) * K, xf.data(), K, o);
      for (int q = 0; q < 4; q++) yp[r + q] = f2bf(o[q]);
    } else {
      float d[16];
      for (int q = 0; q < 4; q++) {
        row_dots(w + (size_t)(r + q) * K, xf.data(), n, K, d);
        for (int t = 0; t < n; t++) yp[(size_t)t * R + r + q] = f2bf(d[t]);
      }
    }
  }
  return y;
}

// Phase 26: a persistent C++ worker so the caller never waits on the Python GIL.
// submit() (GIL held, returns at once) hands the job to the worker; wait_done() releases the GIL and
// blocks until the result is ready. One job in flight at a time. The worker is the OpenMP master,
// so the thread team persists across calls.
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
namespace {
struct Job { torch::Tensor x; std::vector<torch::Tensor> g, u, d; int64_t numa; bool fused = false; bool linear = false;
             int64_t layer = 0; torch::Tensor ti, tw, out; };
// Heap-allocated and never freed: no static destructor runs while the worker waits at exit.
struct State { std::mutex mu; std::condition_variable cv; bool has_job = false, done = true; Job job;
               std::vector<torch::Tensor> result; double last_ms = 0.0; std::string err; };
State* S = new State();
std::thread* worker = nullptr;
void loop() {
  for (;;) {
    Job j;
    { std::unique_lock<std::mutex> l(S->mu); S->cv.wait(l, [] { return S->has_job; }); j = std::move(S->job); S->has_job = false; }
    auto t0 = std::chrono::steady_clock::now();
    std::vector<torch::Tensor> r;
    std::string err;
    try {
      if (j.linear) { j.out.copy_(linear_twin(j.x, j.ti)); r = {j.out}; }
      else r = j.fused ? moe_fused(j.layer, j.x, j.ti, j.tw, j.out, j.numa) : expert_mlps(j.x, j.g, j.u, j.d, j.numa);
    } catch (const std::exception& e) { err = e.what(); }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    j = Job();
    { std::lock_guard<std::mutex> l(S->mu); S->result = std::move(r); S->last_ms = ms; S->err = err; S->done = true; }
    S->cv.notify_all();
  }
}
}
void submit(torch::Tensor x, std::vector<torch::Tensor> g, std::vector<torch::Tensor> u,
            std::vector<torch::Tensor> d, int64_t numa) {
  std::unique_lock<std::mutex> l(S->mu);
  TORCH_CHECK(S->done && !S->has_job, "cpu_moe: a job is already in flight");
  if (!worker) { worker = new std::thread(loop); worker->detach(); }
  S->job = Job{x, std::move(g), std::move(u), std::move(d), numa};
  S->has_job = true; S->done = false;
  l.unlock(); S->cv.notify_all();
}
std::vector<torch::Tensor> wait_done() {
  std::unique_lock<std::mutex> l(S->mu);
  S->cv.wait(l, [] { return S->done; });
  if (!S->err.empty()) { std::string m = S->err; S->err.clear(); TORCH_CHECK(false, "cpu_moe worker: ", m); }
  auto r = std::move(S->result); S->result.clear();
  return r;
}
// Phase 30: one twin linear on the worker (x [n,K] bf16, W [R,K] bf16 -> out [n,R] pinned bf16).
void submit_linear(torch::Tensor x, torch::Tensor W, torch::Tensor out) {
  std::unique_lock<std::mutex> l(S->mu);
  TORCH_CHECK(S->done && !S->has_job, "cpu_moe: a job is already in flight");
  if (!worker) { worker = new std::thread(loop); worker->detach(); }
  Job jb; jb.x = x; jb.numa = 0; jb.linear = true; jb.ti = W; jb.out = out;
  S->job = jb;
  S->has_job = true; S->done = false;
  l.unlock(); S->cv.notify_all();
}
void submit_moe(int64_t layer, torch::Tensor x, torch::Tensor ti, torch::Tensor tw, torch::Tensor out, int64_t numa) {
  std::unique_lock<std::mutex> l(S->mu);
  TORCH_CHECK(S->done && !S->has_job, "cpu_moe: a job is already in flight");
  if (!worker) { worker = new std::thread(loop); worker->detach(); }
  S->job = Job{x, {}, {}, {}, numa, true, false, layer, ti, tw, out};
  S->has_job = true; S->done = false;
  l.unlock(); S->cv.notify_all();
}
double kernel_ms() { std::lock_guard<std::mutex> l(S->mu); return S->last_ms; }

// Phase 34: move the pages of W [R, K] in place so rows [0, R/2) live on node 0 and [R/2, R) on node 1.
// move_pages migrates each page (page cache or anonymous) to the target node; no second copy remains.
// A page straddling the split goes with the row its first byte belongs to. Returns {pages, on_target_after}.
std::vector<int64_t> place_rows(torch::Tensor W) {
  TORCH_CHECK(W.is_contiguous() && W.dim() == 2);
  const size_t R = W.size(0), rb = (size_t)W.size(1) * W.element_size();
  const uintptr_t b0 = (uintptr_t)W.data_ptr(), b1 = b0 + R * rb, mid = b0 + (R / 2) * rb;
  std::vector<void*> pages; std::vector<int> nodes;
  volatile uint8_t sink = 0;
  for (uintptr_t pg = b0 & ~(uintptr_t)4095; pg < b1; pg += 4096) {
    uintptr_t first = pg < b0 ? b0 : pg;
    sink ^= *(volatile const uint8_t*)first;           // fault the page in
    pages.push_back((void*)pg);
    nodes.push_back(first < mid ? 0 : 1);
  }
  (void)sink;
  std::vector<int> status(pages.size(), -1);
  const size_t CH = 1 << 14;
  for (size_t i = 0; i < pages.size(); i += CH) {
    size_t c = std::min(CH, pages.size() - i);
    move_pages(0, c, &pages[i], &nodes[i], &status[i], MPOL_MF_MOVE);
  }
  std::vector<int> st(pages.size(), -1);
  for (size_t i = 0; i < pages.size(); i += CH) {
    size_t c = std::min(CH, pages.size() - i);
    move_pages(0, c, &pages[i], nullptr, &st[i], 0);
  }
  int64_t ok = 0;
  for (size_t i = 0; i < pages.size(); i++) ok += (st[i] == nodes[i]);
  return {(int64_t)pages.size(), ok};
}
// Node histogram of W's pages split at R/2: {n00, n01, n10, n11} = (half, node) counts.
std::vector<int64_t> rows_nodes(torch::Tensor W) {
  const size_t R = W.size(0), rb = (size_t)W.size(1) * W.element_size();
  const uintptr_t b0 = (uintptr_t)W.data_ptr(), b1 = b0 + R * rb, mid = b0 + (R / 2) * rb;
  std::vector<void*> pages; std::vector<int> half;
  for (uintptr_t pg = b0 & ~(uintptr_t)4095; pg < b1; pg += 4096) {
    uintptr_t first = pg < b0 ? b0 : pg; pages.push_back((void*)pg); half.push_back(first < mid ? 0 : 1);
  }
  std::vector<int> st(pages.size(), -1);
  move_pages(0, pages.size(), pages.data(), nullptr, st.data(), 0);
  std::vector<int64_t> h(4, 0);
  for (size_t i = 0; i < pages.size(); i++) if (st[i] == 0 || st[i] == 1) h[half[i] * 2 + st[i]]++;
  return h;
}

// ============================================================================ Phase 37: NX12 v2
// NX12 per matrix [R, K]: lo [R*K] u8 = sign<<7 | mantissa; nib [R*K/2] u8 = 4-bit exponent codes (weight 2i low
// nibble, 2i+1 high); lut [16] u8 = code -> exponent (code 15 = escape); esc [n] u8 = escaped exponents in row-major
// order; off [R+1] i64 = escape offset at each row start. Decode reconstructs the exact BF16 bits, 64 weights per
// step (512-bit, VBMI byte permutes); the dot product then runs the unchanged twin arithmetic (16 fp32 lanes,
// lane l accumulates k = 16j + l in ascending j, fixed tree), so outputs are bit-identical to the raw kernel.
struct alignas(64) PadAt { std::atomic<int> v; };
#if defined(__AVX512VBMI__)
#define VH_NX_FAST 1
struct NXm { const uint8_t* lo; const uint8_t* nib; const uint8_t* esc; const int64_t* off; __m512i lut; int K;
             const uint8_t* lutp; };
#else
#define VH_NX_FAST 0
struct NXm { const uint8_t* lo; const uint8_t* nib; const uint8_t* esc; const int64_t* off; int K; const uint8_t* lutp; };
#endif
static NXm make_nxm(const torch::Tensor* t, int K) {
  NXm m;
  m.lo = t[0].data_ptr<uint8_t>(); m.nib = t[1].data_ptr<uint8_t>(); m.esc = t[3].data_ptr<uint8_t>();
  m.off = t[4].data_ptr<int64_t>();
  m.lutp = t[2].data_ptr<uint8_t>(); m.K = K;
#if VH_NX_FAST
  m.lut = _mm512_zextsi128_si512(_mm_loadu_si128((const __m128i*)t[2].data_ptr<uint8_t>()));
#endif
  return m;
}
static int g_pf = 0;
#if VH_NX_FAST
static __m512i g_plo, g_phi;
static void init_perm() {
  { const char* v = std::getenv("VAHANA_PF"); g_pf = v ? std::atoi(v) : 0; }
  alignas(64) uint8_t a[64], b[64];
  for (int i = 0; i < 32; i++) { a[2 * i] = i; a[2 * i + 1] = 64 + i; b[2 * i] = 32 + i; b[2 * i + 1] = 96 + i; }
  g_plo = _mm512_load_si512(a); g_phi = _mm512_load_si512(b);
}
static inline void dec64(const NXm& m, size_t base, int k, const uint8_t*& ep, V16* w) {
  const __m512i lo = _mm512_loadu_si512(m.lo + base + k);
  const __m512i n16 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(m.nib + (base + k) / 2)));
  const __m512i code = _mm512_or_si512(_mm512_and_si512(n16, _mm512_set1_epi16(0x000F)),
                                       _mm512_slli_epi16(_mm512_and_si512(n16, _mm512_set1_epi16(0x00F0)), 4));
  __m512i ex = _mm512_permutexvar_epi8(code, m.lut);
  const __mmask64 em = _mm512_cmpeq_epi8_mask(code, _mm512_set1_epi8(15));
  if (__builtin_expect(em != 0, 0)) { ex = _mm512_mask_expandloadu_epi8(ex, em, ep); ep += _mm_popcnt_u64(em); }
  const __m512i hib = _mm512_or_si512(_mm512_and_si512(lo, _mm512_set1_epi8((char)0x80)),
                                      _mm512_and_si512(_mm512_srli_epi16(ex, 1), _mm512_set1_epi8(0x7F)));
  const __m512i lob = _mm512_or_si512(_mm512_and_si512(_mm512_slli_epi16(ex, 7), _mm512_set1_epi8((char)0x80)),
                                      _mm512_and_si512(lo, _mm512_set1_epi8(0x7F)));
  const __m512i wa = _mm512_permutex2var_epi8(lob, g_plo, hib), wb = _mm512_permutex2var_epi8(lob, g_phi, hib);
  w[0] = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(wa)), 16));
  w[1] = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(wa, 1)), 16));
  w[2] = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(wb)), 16));
  w[3] = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(wb, 1)), 16));
}
#else
static void init_perm() { const char* v = std::getenv("VAHANA_PF"); g_pf = v ? std::atoi(v) : 0; }
// Portable NX12 decode: weights k .. k+63 of a row in natural order; escaped exponents are consumed from the escape
// stream in ascending position, exactly as the VBMI expand-load does. Same BF16 bits, so the same dot products.
static inline void dec64(const NXm& m, size_t base, int k, const uint8_t*& ep, V16* w) {
  uint16_t b[64];
  for (int i = 0; i < 64; i++) {
    const size_t idx = base + k + i;
    const uint8_t lo = m.lo[idx];
    const uint8_t nb = m.nib[idx / 2];
    const uint8_t code = (idx & 1) ? (nb >> 4) : (nb & 0x0F);
    const uint8_t ex = code == 15 ? *ep++ : m.lutp[code];
    b[i] = (uint16_t)(((lo & 0x80) << 8) | ((uint16_t)ex << 7) | (lo & 0x7F));
  }
  for (int j = 0; j < 4; j++) w[j] = v_ldbf16(b + 16 * j);
}
#endif
// four rows, one token, same per-row arithmetic as row_dots4_n1
static inline void dots4_nx(const NXm* const* m4, const int* r4, const float* X, float* out) {
  V16 a[4] = {v_zero(), v_zero(), v_zero(), v_zero()};
  const uint8_t* ep[4]; size_t rb[4];
  for (int q = 0; q < 4; q++) { ep[q] = m4[q]->esc + m4[q]->off[r4[q]]; rb[q] = (size_t)r4[q] * m4[q]->K; }
  const int K = m4[0]->K, pf = g_pf;
  V16 w[4];
  for (int k = 0; k < K; k += 64) {
    if (pf) {
      for (int q = 0; q < 4; q++) {
        VH_PREFETCH(m4[q]->lo + rb[q] + k + pf);
        VH_PREFETCH(m4[q]->nib + (rb[q] + k + pf) / 2);
      }
    }
    const V16 x0 = v_loadf(X + k), x1 = v_loadf(X + k + 16), x2 = v_loadf(X + k + 32), x3 = v_loadf(X + k + 48);
    for (int q = 0; q < 4; q++) {
      dec64(*m4[q], rb[q], k, ep[q], w);
      a[q] = v_fma(w[0], x0, a[q]);
      a[q] = v_fma(w[1], x1, a[q]);
      a[q] = v_fma(w[2], x2, a[q]);
      a[q] = v_fma(w[3], x3, a[q]);
    }
  }
  for (int q = 0; q < 4; q++) out[q] = tree16(a[q]);
}
// x [E, 1, H] bf16; packs: 15 tensors per expert (gate lo,nib,lut,esc,off; up ...; down ...); numa as expert_mlps.
torch::Tensor expert_mlps_nx_ctx(torch::Tensor x, std::vector<torch::Tensor> packs, int64_t I_, int64_t numa,
                                 const ActCtx* ctx);
torch::Tensor expert_mlps_nx(torch::Tensor x, std::vector<torch::Tensor> packs, int64_t I_, int64_t numa) {
  return expert_mlps_nx_ctx(x, packs, I_, numa, nullptr);
}
torch::Tensor expert_mlps_nx_ctx(torch::Tensor x, std::vector<torch::Tensor> packs, int64_t I_, int64_t numa,
                                 const ActCtx* ctx) {
  static const int nthr = [] { const char* v = std::getenv("OMP_NUM_THREADS"); int t = v ? std::atoi(v) : 0;
                               init_perm(); return t > 0 ? t : omp_get_num_procs(); }();
  omp_set_num_threads(nthr);
  const int E = x.size(0), H = x.size(2), I = (int)I_;
  TORCH_CHECK(x.size(1) == 1 && (int)packs.size() == 15 * E && H % 64 == 0 && I % 64 == 0);
  std::vector<NXm> G(E), U(E), D(E);
  for (int e = 0; e < E; e++) {
    G[e] = make_nxm(&packs[15 * e], H); U[e] = make_nxm(&packs[15 * e + 5], H); D[e] = make_nxm(&packs[15 * e + 10], I);
  }
  std::vector<float> xf((size_t)E * H), hf((size_t)E * I);
  const uint16_t* xp = (const uint16_t*)x.data_ptr();
  for (size_t i = 0; i < xf.size(); i++) xf[i] = bf2f(xp[i]);
  auto y = torch::empty({E, 1, H}, x.options());
  uint16_t* yp = (uint16_t*)y.data_ptr();
  std::vector<int> t1[2], t2[2];
  for (int e = 0; e < E; e++) {
    for (int b = 0; b < I / 16; b++) t1[numa == 2 ? (b >= I / 32) : 0].push_back(e * (I / 16) + b);
    for (int b = 0; b < H / 16; b++) t2[numa == 2 ? (b >= H / 32) : 0].push_back(e * (H / 16) + b);
  }
  std::atomic<size_t> cur[2];
  #pragma omp parallel
  {
    const int nd = numa == 2 ? ((sched_getcpu() >= 32) ? 1 : 0) : 0;
    float o[4];
    for (int pass = 0; pass < 2; pass++) {
      #pragma omp single
      { cur[0] = 0; cur[1] = 0; }
      for (int hop = 0; hop < (numa == 2 ? 2 : 1); hop++) {
        const int Ls = numa == 2 ? (nd ^ hop) : 0;
        const std::vector<int>& T = pass == 0 ? t1[Ls] : t2[Ls];
        const int blocks = (pass == 0 ? I : H) / 16;
        for (size_t i = cur[Ls].fetch_add(1); i < T.size(); i = cur[Ls].fetch_add(1)) {
          const int e = T[i] / blocks, b = T[i] % blocks;
          if (pass == 0) {
            const float* xe = &xf[(size_t)e * H];
            for (int r = b * 16; r < b * 16 + 16; r += 2) {
              const NXm* m4[4] = {&G[e], &G[e], &U[e], &U[e]}; const int r4[4] = {r, r + 1, r, r + 1};
              dots4_nx(m4, r4, xe, o);
              for (int q = 0; q < 2; q++) {
                hf[(size_t)e * I + r + q] = act_h(ctx, e, r + q, o[q], o[2 + q]);
              }
            }
          } else {
            const float* he = &hf[(size_t)e * I];
            for (int r = b * 16; r < b * 16 + 16; r += 4) {
              const NXm* m4[4] = {&D[e], &D[e], &D[e], &D[e]}; const int r4[4] = {r, r + 1, r + 2, r + 3};
              dots4_nx(m4, r4, he, o);
              for (int q = 0; q < 4; q++) yp[(size_t)e * H + r + q] = down_o(ctx, e, r + q, o[q]);
            }
          }
        }
      }
      #pragma omp barrier
    }
  }
  return y;
}
// Move the pages of a 1-D buffer in place: bytes [0, split) -> node 0, [split, end) -> node 1.
std::vector<int64_t> place_split(torch::Tensor t, int64_t split) {
  const uintptr_t b0 = (uintptr_t)t.data_ptr(), b1 = b0 + (size_t)t.numel() * t.element_size(), mid = b0 + split;
  std::vector<void*> pages; std::vector<int> nodes;
  volatile uint8_t sink = 0;
  for (uintptr_t pg = b0 & ~(uintptr_t)4095; pg < b1; pg += 4096) {
    uintptr_t first = pg < b0 ? b0 : pg;
    sink ^= *(volatile const uint8_t*)first;
    pages.push_back((void*)pg); nodes.push_back(first < mid ? 0 : 1);
  }
  (void)sink;
  std::vector<int> st(pages.size(), -1);
  const size_t CH = 1 << 14;
  for (size_t i = 0; i < pages.size(); i += CH)
    move_pages(0, std::min(CH, pages.size() - i), &pages[i], &nodes[i], &st[i], MPOL_MF_MOVE);
  for (size_t i = 0; i < pages.size(); i += CH)
    move_pages(0, std::min(CH, pages.size() - i), &pages[i], nullptr, &st[i], 0);
  int64_t ok = 0;
  for (size_t i = 0; i < pages.size(); i++) ok += (st[i] == nodes[i]);
  return {(int64_t)pages.size(), ok};
}

// Phase 38: parallel NX12 encoder. W [R, K] bf16 contiguous -> {lo, nib, lut, esc(+64 slack), off}.
std::vector<torch::Tensor> encode_nx(torch::Tensor W) {
  TORCH_CHECK(W.is_contiguous() && W.dim() == 2 && W.size(1) % 64 == 0);
  static const int nthr = [] { const char* v = std::getenv("OMP_NUM_THREADS"); int t = v ? std::atoi(v) : 0;
                               return t > 0 ? t : omp_get_num_procs(); }();
  omp_set_num_threads(nthr);
  const int64_t R = W.size(0), K = W.size(1);
  const uint16_t* w = (const uint16_t*)W.data_ptr();
  std::vector<int64_t> cnt(256, 0);
  #pragma omp parallel
  {
    std::vector<int64_t> h(256, 0);
    #pragma omp for schedule(static)
    for (int64_t r = 0; r < R; r++) for (int64_t k = 0; k < K; k++) h[(w[r * K + k] >> 7) & 0xFF]++;
    #pragma omp critical
    for (int i = 0; i < 256; i++) cnt[i] += h[i];
  }
  std::vector<int> order(256);
  for (int i = 0; i < 256; i++) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
  auto u8 = torch::TensorOptions().dtype(torch::kUInt8);
  auto lut = torch::zeros({16}, u8);
  uint8_t inv[256];
  for (int i = 0; i < 256; i++) inv[i] = 15;
  for (int i = 0; i < 15; i++) { lut.data_ptr<uint8_t>()[i] = (uint8_t)order[i]; inv[order[i]] = (uint8_t)i; }
  auto lo = torch::empty({R * K}, u8), nib = torch::empty({R * K / 2}, u8);
  auto off = torch::empty({R + 1}, torch::TensorOptions().dtype(torch::kInt64));
  uint8_t* lp = lo.data_ptr<uint8_t>(); uint8_t* np = nib.data_ptr<uint8_t>(); int64_t* op = off.data_ptr<int64_t>();
  std::vector<int64_t> rc(R, 0);
  #pragma omp parallel for schedule(static)
  for (int64_t r = 0; r < R; r++) {
    int64_t c = 0;
    for (int64_t k = 0; k < K; k += 2) {
      uint16_t a = w[r * K + k], b = w[r * K + k + 1];
      uint8_t ca = inv[(a >> 7) & 0xFF], cb = inv[(b >> 7) & 0xFF];
      lp[r * K + k] = (uint8_t)(((a >> 15) << 7) | (a & 0x7F));
      lp[r * K + k + 1] = (uint8_t)(((b >> 15) << 7) | (b & 0x7F));
      np[(r * K + k) / 2] = (uint8_t)(ca | (cb << 4));
      c += (ca == 15) + (cb == 15);
    }
    rc[r] = c;
  }
  op[0] = 0;
  for (int64_t r = 0; r < R; r++) op[r + 1] = op[r] + rc[r];
  auto esc = torch::zeros({op[R] + 64}, u8);
  uint8_t* ep = esc.data_ptr<uint8_t>();
  #pragma omp parallel for schedule(static)
  for (int64_t r = 0; r < R; r++) {
    int64_t j = op[r];
    for (int64_t k = 0; k < K; k++) { uint16_t a = w[r * K + k]; if (inv[(a >> 7) & 0xFF] == 15) ep[j++] = (uint8_t)((a >> 7) & 0xFF); }
  }
  return {lo, nib, lut, esc, off};
}
// packs: 15 per expert in expert order (gate lo,nib,lut,esc,off; up ...; down ...)
void register_layer_nx(int64_t layer, std::vector<torch::Tensor> packs) { LAYERS->at(layer).nx = std::move(packs); }

void register_shared_nx(int64_t layer, std::vector<torch::Tensor> packs) { LAYERS->at(layer).snx = std::move(packs); }

// Phase 40: twin linear over an NX12 matrix, one token; rows [0, R/2) on node-0 threads first (local-first stealing).
torch::Tensor linear_nx(const float* xf, const std::vector<torch::Tensor>& pk, int R, int K, torch::TensorOptions opt) {
  static bool inited = (init_perm(), true); (void)inited;
  NXm m = make_nxm(pk.data(), K);
  auto y = torch::empty({1, R}, opt);
  uint16_t* yp = (uint16_t*)y.data_ptr();
  PadAt cur[2]; cur[0].v = 0; cur[1].v = R / 2;
  const int CH = (R / 2) % 8 == 0 ? 8 : 4;
  #pragma omp parallel
  {
    const int nd = (sched_getcpu() >= 32) ? 1 : 0;
    float o[4];
    for (int hop = 0; hop < 2; hop++) {
      const int Ls = nd ^ hop, end = Ls == 0 ? R / 2 : R;
      for (int r0 = cur[Ls].v.fetch_add(CH); r0 < end; r0 = cur[Ls].v.fetch_add(CH)) {
        for (int r = r0; r < r0 + CH; r += 4) {
          const NXm* m4[4] = {&m, &m, &m, &m}; const int r4[4] = {r, r + 1, r + 2, r + 3};
          dots4_nx(m4, r4, xf, o);
          for (int q = 0; q < 4; q++) yp[r + q] = f2bf(o[q]);
        }
      }
    }
  }
  return y;
}

// ============================================================================ Phase 41: fused small-expert decode
// One token. Experts j = 0..E-1 (packs P[j], intermediate Is[j]); x as fp32; y[j] bf16 [H].
// Gate/up blocks (16 rows of I_j) of all experts, then down blocks (16 rows of H). Rows in the first half of a
// matrix go to node-0 threads first (placement split), with local-first stealing. A down block of expert j
// starts once all gate/up blocks of j are done (per-expert counter), so there is no global barrier.
static void experts_nx_fused(const std::vector<const torch::Tensor*>& P, const std::vector<int>& Is, const float* xf,
                             int H, const std::vector<uint16_t*>& y, int numa, const ActCtx* ctx) {
  static const int nthr = [] { const char* v = std::getenv("OMP_NUM_THREADS"); int t = v ? std::atoi(v) : 0;
                               init_perm(); return t > 0 ? t : omp_get_num_procs(); }();
  omp_set_num_threads(nthr);
  const int E = (int)P.size();
  std::vector<NXm> G(E), U(E), D(E);
  std::vector<size_t> ho(E + 1, 0);
  for (int j = 0; j < E; j++) {
    G[j] = make_nxm(P[j], H); U[j] = make_nxm(P[j] + 5, H); D[j] = make_nxm(P[j] + 10, Is[j]);
    ho[j + 1] = ho[j] + Is[j];
  }
  std::vector<float> hf(ho[E]);
  std::vector<int> ta[2], tb[2];                       // task = j << 16 | block
  for (int j = 0; j < E; j++) {
    for (int b = 0; b < Is[j] / 16; b++) ta[numa == 2 ? (b >= Is[j] / 32) : 0].push_back(j << 16 | b);
    for (int b = 0; b < H / 16; b++) tb[numa == 2 ? (b >= H / 32) : 0].push_back(j << 16 | b);
  }
  std::vector<PadAt> done(E);
  for (int j = 0; j < E; j++) done[j].v = 0;
  PadAt ca[2], cb[2];
  ca[0].v = 0; ca[1].v = 0; cb[0].v = 0; cb[1].v = 0;
  const int nh = numa == 2 ? 2 : 1;
  #pragma omp parallel
  {
    const int nd = numa == 2 ? ((sched_getcpu() >= 32) ? 1 : 0) : 0;
    float o[4];
    for (int hop = 0; hop < nh; hop++) {
      const int Ls = nd ^ hop;
      const std::vector<int>& T = ta[Ls];
      for (int i = ca[Ls].v.fetch_add(1); i < (int)T.size(); i = ca[Ls].v.fetch_add(1)) {
        const int j = T[i] >> 16, b = T[i] & 0xFFFF;
        float* hj = &hf[ho[j]];
        for (int r = b * 16; r < b * 16 + 16; r += 2) {
          const NXm* m4[4] = {&G[j], &G[j], &U[j], &U[j]}; const int r4[4] = {r, r + 1, r, r + 1};
          dots4_nx(m4, r4, xf, o);
          for (int q = 0; q < 2; q++) {
            hj[r + q] = act_h(ctx, j, r + q, o[q], o[2 + q]);
          }
        }
        done[j].v.fetch_add(1, std::memory_order_release);
      }
    }
    for (int hop = 0; hop < nh; hop++) {
      const int Ls = nd ^ hop;
      const std::vector<int>& T = tb[Ls];
      for (int i = cb[Ls].v.fetch_add(1); i < (int)T.size(); i = cb[Ls].v.fetch_add(1)) {
        const int j = T[i] >> 16, b = T[i] & 0xFFFF;
        while (done[j].v.load(std::memory_order_acquire) < Is[j] / 16) VH_PAUSE();
        const float* hj = &hf[ho[j]];
        for (int r = b * 16; r < b * 16 + 16; r += 4) {
          const NXm* m4[4] = {&D[j], &D[j], &D[j], &D[j]}; const int r4[4] = {r, r + 1, r + 2, r + 3};
          dots4_nx(m4, r4, hj, o);
          for (int q = 0; q < 4; q++) y[j][r + q] = down_o(ctx, j, r + q, o[q]);
        }
      }
    }
  }
}

// Phase 41: node-local streaming read of NX12 lo/nib buffers (packs, 5 per matrix); returns GB/s.
double read_bw(std::vector<torch::Tensor> packs, int64_t numa) {
  static const int nthr = [] { const char* v = std::getenv("OMP_NUM_THREADS"); int t = v ? std::atoi(v) : 0;
                               return t > 0 ? t : omp_get_num_procs(); }();
  omp_set_num_threads(nthr);
  std::vector<std::pair<const uint8_t*, size_t>> ch[2];
  size_t tot = 0;
  const size_t C = 65536;
  for (size_t i = 0; i < packs.size(); i++) {
    if (i % 5 > 1) continue;
    const uint8_t* p = packs[i].data_ptr<uint8_t>(); const size_t n = packs[i].numel(), half = n / 2;
    for (size_t o = 0; o + C <= n; o += C) { ch[numa == 2 ? (o >= half) : 0].push_back({p + o, C}); tot += C; }
  }
  PadAt cur[2]; cur[0].v = 0; cur[1].v = 0;
  std::atomic<long long> sink{0};
  const double t0 = now_ms();
  #pragma omp parallel
  {
    const int nd = numa == 2 ? ((sched_getcpu() >= 32) ? 1 : 0) : 0;
    uint64_t acc[4] = {0, 0, 0, 0};
    for (int hop = 0; hop < (numa == 2 ? 2 : 1); hop++) {
      const int Ls = nd ^ hop;
      for (int i = cur[Ls].v.fetch_add(1); i < (int)ch[Ls].size(); i = cur[Ls].v.fetch_add(1)) {
        const uint8_t* p = ch[Ls][i].first;
        for (size_t o = 0; o < C; o += 256) {
          { uint64_t t_[8]; std::memcpy(t_, p + o, 64); for (int z = 0; z < 8; z++) acc[0] += t_[z]; }
          { uint64_t t_[8]; std::memcpy(t_, p + o + 64, 64); for (int z = 0; z < 8; z++) acc[1] += t_[z]; }
          { uint64_t t_[8]; std::memcpy(t_, p + o + 128, 64); for (int z = 0; z < 8; z++) acc[2] += t_[z]; }
          { uint64_t t_[8]; std::memcpy(t_, p + o + 192, 64); for (int z = 0; z < 8; z++) acc[3] += t_[z]; }
        }
      }
    }
    sink += (long long)(acc[0] + acc[1] + acc[2] + acc[3]);
  }
  const double ms = now_ms() - t0;
  return tot / ms / 1e6;
}

// ============================================================================ M0: host-node jobs
#include <dlfcn.h>
namespace {
typedef int (*LaunchHostFn)(void* stream, void (*fn)(void*), void* data);
LaunchHostFn g_launch_host = nullptr;
struct HostJob { int64_t layer; torch::Tensor x, ti, tw, out; int64_t numa; bool noop; int kind = 0; };
std::vector<HostJob*>* HOST_JOBS = new std::vector<HostJob*>();
std::atomic<long long> g_host_calls{0};
void host_cb(void* p) {
  HostJob* j = static_cast<HostJob*>(p);
  g_host_calls++;
  if (j->noop) return;
  if (j->kind == 1) {                                                                  // ti holds W
    try { submit_linear(j->x, j->ti, j->out); wait_done(); }
    catch (const std::exception& e) { std::fprintf(stderr, "vahana host linear failed: %s\n", e.what()); std::fflush(stderr); std::abort(); }
    return;
  }
  if (j->kind == 2) {                                                                  // embedding rows
    const int64_t n = j->x.numel(), H = j->ti.size(1);
    const int64_t* ids = j->x.data_ptr<int64_t>();
    const char* w = (const char*)j->ti.data_ptr(); char* o = (char*)j->out.data_ptr();
    const size_t rb = (size_t)H * j->ti.element_size();
    for (int64_t i = 0; i < n; i++) std::memcpy(o + i * rb, w + (size_t)ids[i] * rb, rb);
    return;
  }
  try {
    submit_moe(j->layer, j->x, j->ti, j->tw, j->out, j->numa);
    wait_done();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vahana host job failed: %s\n", e.what()); std::fflush(stderr); std::abort();
  }
}
LaunchHostFn launch_host() {
  if (!g_launch_host) {
    for (const char* lib : {"libcudart.so.12", "libcudart.so"}) {
      void* h = dlopen(lib, RTLD_NOW | RTLD_NOLOAD);
      if (h) { g_launch_host = (LaunchHostFn)dlsym(h, "cudaLaunchHostFunc"); if (g_launch_host) break; }
    }
    TORCH_CHECK(g_launch_host, "cudaLaunchHostFunc not found (is torch.cuda initialised?)");
  }
  return g_launch_host;
}
}
// Register a job on fixed pinned buffers; returns its id. noop=true makes an empty host node (overhead probe).
int64_t make_host_job(int64_t layer, torch::Tensor x, torch::Tensor ti, torch::Tensor tw, torch::Tensor out,
                      int64_t numa, bool noop) {
  HOST_JOBS->push_back(new HostJob{layer, x, ti, tw, out, numa, noop});
  return (int64_t)HOST_JOBS->size() - 1;
}
// Enqueue job `id` on a CUDA stream (torch.cuda.current_stream().cuda_stream); capturable.
void enqueue_host_job(int64_t stream, int64_t id) {
  const int rc = launch_host()((void*)stream, host_cb, HOST_JOBS->at(id));
  TORCH_CHECK(rc == 0, "cudaLaunchHostFunc failed: ", rc);
}
int64_t host_calls() { return g_host_calls.load(); }
int64_t make_host_linear(torch::Tensor W, torch::Tensor x, torch::Tensor out) {
  HostJob* j = new HostJob{0, x, W, torch::Tensor(), out, 0, false}; j->kind = 1;
  HOST_JOBS->push_back(j); return (int64_t)HOST_JOBS->size() - 1;
}
int64_t make_host_embed(torch::Tensor W, torch::Tensor ids, torch::Tensor out) {
  TORCH_CHECK(W.is_contiguous() && W.dim() == 2 && ids.scalar_type() == torch::kLong);
  HostJob* j = new HostJob{0, ids, W, torch::Tensor(), out, 0, false}; j->kind = 2;
  HOST_JOBS->push_back(j); return (int64_t)HOST_JOBS->size() - 1;
}

std::string isa() { return VH_ISA; }
bool nx_fast() { return VH_NX_FAST != 0; }
bool have_numa() { return VH_HAVE_NUMA != 0; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("have_numa", &have_numa);
  m.def("isa", &isa);
  m.def("nx_fast", &nx_fast);
  m.def("register_layer_act", &register_layer_act);
  m.def("make_host_job", &make_host_job);
  m.def("enqueue_host_job", &enqueue_host_job);
  m.def("host_calls", &host_calls);
  m.def("make_host_linear", &make_host_linear);
  m.def("make_host_embed", &make_host_embed);
  m.def("read_bw", &read_bw, py::call_guard<py::gil_scoped_release>());
  m.def("register_linear_nx", &register_linear_nx);
  m.def("register_shared_nx", &register_shared_nx);
  m.def("encode_nx", &encode_nx, py::call_guard<py::gil_scoped_release>());
  m.def("register_layer_nx", &register_layer_nx);
  m.def("expert_mlps_nx", &expert_mlps_nx, py::call_guard<py::gil_scoped_release>());
  m.def("place_split", &place_split, py::call_guard<py::gil_scoped_release>());
  m.def("place_rows", &place_rows, py::call_guard<py::gil_scoped_release>());
  m.def("rows_nodes", &rows_nodes, py::call_guard<py::gil_scoped_release>());
  m.def("expert_mlps", &expert_mlps, py::call_guard<py::gil_scoped_release>());
  m.def("submit", &submit);
  m.def("wait_done", &wait_done, py::call_guard<py::gil_scoped_release>());
  m.def("kernel_ms", &kernel_ms);
  m.def("phase_ms", &phase_ms);
  m.def("register_layer", &register_layer);
  m.def("submit_moe", &submit_moe);
  m.def("register_shared", &register_shared);
  m.def("submit_linear", &submit_linear);
  m.def("linear_twin", &linear_twin, py::call_guard<py::gil_scoped_release>());
}
