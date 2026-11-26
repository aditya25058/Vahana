// v1.4: microbenchmark of the twin-q4 CPU dot product (one thread). Each variant must give the same bits as V0 (the
// 1.3 kernel); it prints cycles per 32 weights for an L2-resident working set and for a DRAM-sized one.
//   g++ -O3 -mavx2 -mfma -ffp-contract=off -std=c++20 bench_q4.cpp -o bench_q4 && ./bench_q4
#include <immintrin.h>
#include <x86intrin.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>

struct V16 { __m256 lo, hi; };
static inline V16 v_zero() { return {_mm256_setzero_ps(), _mm256_setzero_ps()}; }
static inline V16 v_loadf(const float* p) { return {_mm256_loadu_ps(p), _mm256_loadu_ps(p + 8)}; }
static inline V16 v_fma(V16 a, V16 b, V16 c) { return {_mm256_fmadd_ps(a.lo, b.lo, c.lo), _mm256_fmadd_ps(a.hi, b.hi, c.hi)}; }
static inline float tree16(V16 v) {
  float a[16]; _mm256_storeu_ps(a, v.lo); _mm256_storeu_ps(a + 8, v.hi);
  for (int s = 8; s > 0; s >>= 1) for (int i = 0; i < s; i++) a[i] = a[i] + a[i + s];
  return a[0];
}
static inline float bf2f(uint16_t h) { uint32_t u = ((uint32_t)h) << 16; float f; std::memcpy(&f, &u, 4); return f; }

static inline void v_q4_pair(const uint8_t* b, float s, V16& lo, V16& hi) {
  const __m128i v = _mm_loadu_si128((const __m128i*)b), m4 = _mm_set1_epi8(0x0F);
  const __m128i bl = _mm_and_si128(v, m4), bh = _mm_and_si128(_mm_srli_epi16(v, 4), m4);
  const __m256i e8 = _mm256_set1_epi32(8);
  const __m256 sv = _mm256_set1_ps(s);
  lo.lo = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(bl), e8)), sv);
  lo.hi = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(bl, 8)), e8)), sv);
  hi.lo = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(bh), e8)), sv);
  hi.hi = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(bh, 8)), e8)), sv);
}

// V0: the 1.3 kernel (k0 / G per block)
static void v0(const uint8_t* const* R, const float* X, int K, int G, float* out) {
  V16 a[4] = {v_zero(), v_zero(), v_zero(), v_zero()};
  for (int k0 = 0; k0 < K; k0 += 32) {
    const V16 x0 = v_loadf(X + k0), x1 = v_loadf(X + k0 + 16);
    for (int i = 0; i < 4; i++) {
      V16 wl, wh;
      v_q4_pair(R[i] + k0 / 2, bf2f(((const uint16_t*)(R[i] + K / 2))[k0 / G]), wl, wh);
      a[i] = v_fma(wl, x0, a[i]);
      a[i] = v_fma(wh, x1, a[i]);
    }
  }
  for (int i = 0; i < 4; i++) out[i] = tree16(a[i]);
}

// V1: scale once per group
static void v1(const uint8_t* const* R, const float* X, int K, int G, float* out) {
  V16 a[4] = {v_zero(), v_zero(), v_zero(), v_zero()};
  for (int g0 = 0, gi = 0; g0 < K; g0 += G, gi++) {
    float s[4];
    for (int i = 0; i < 4; i++) s[i] = bf2f(((const uint16_t*)(R[i] + K / 2))[gi]);
    for (int k0 = g0; k0 < g0 + G; k0 += 32) {
      const V16 x0 = v_loadf(X + k0), x1 = v_loadf(X + k0 + 16);
      for (int i = 0; i < 4; i++) {
        V16 wl, wh;
        v_q4_pair(R[i] + k0 / 2, s[i], wl, wh);
        a[i] = v_fma(wl, x0, a[i]);
        a[i] = v_fma(wh, x1, a[i]);
      }
    }
  }
  for (int i = 0; i < 4; i++) out[i] = tree16(a[i]);
}

// V2: magic-number nibble -> float (no int convert): float bits 0x4B000000 | n = 2^23 + n exactly; minus (2^23 + 8)
// gives q = n - 8 exactly; times s exactly (q*s <= 12 bits). Same weight values, so the same fma inputs and bits.
// Bytes are widened with one vpshufb per 8 values from a broadcast 128-bit load (lane 0 takes bytes 0-3/8-11, ...).
static inline void q4_magic(const uint8_t* b, __m256 sv, V16& lo, V16& hi) {
  const __m256i v = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)b));
  const __m256i m4 = _mm256_set1_epi8(0x0F);
  const __m256i bl = _mm256_and_si256(v, m4), bh = _mm256_and_si256(_mm256_srli_epi16(v, 4), m4);
  // lane 0 of the result gets bytes i..i+3 into dword low bytes, lane 1 bytes i+4..i+7 (both lanes hold all 16 bytes)
  const __m256i s0 = _mm256_setr_epi8(0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1,
                                      4, -1, -1, -1, 5, -1, -1, -1, 6, -1, -1, -1, 7, -1, -1, -1);
  const __m256i s1 = _mm256_setr_epi8(8, -1, -1, -1, 9, -1, -1, -1, 10, -1, -1, -1, 11, -1, -1, -1,
                                      12, -1, -1, -1, 13, -1, -1, -1, 14, -1, -1, -1, 15, -1, -1, -1);
  const __m256i mg = _mm256_set1_epi32(0x4B000000);
  const __m256 off = _mm256_set1_ps(8388616.0f);
  auto cv = [&](__m256i bytes, __m256i sh) {
    return _mm256_mul_ps(_mm256_sub_ps(_mm256_castsi256_ps(_mm256_or_si256(_mm256_shuffle_epi8(bytes, sh), mg)), off), sv);
  };
  lo.lo = cv(bl, s0); lo.hi = cv(bl, s1); hi.lo = cv(bh, s0); hi.hi = cv(bh, s1);
}
static void v2(const uint8_t* const* R, const float* X, int K, int G, float* out) {
  V16 a[4] = {v_zero(), v_zero(), v_zero(), v_zero()};
  for (int g0 = 0, gi = 0; g0 < K; g0 += G, gi++) {
    __m256 s[4];
    for (int i = 0; i < 4; i++) s[i] = _mm256_set1_ps(bf2f(((const uint16_t*)(R[i] + K / 2))[gi]));
    for (int k0 = g0; k0 < g0 + G; k0 += 32) {
      const V16 x0 = v_loadf(X + k0), x1 = v_loadf(X + k0 + 16);
      for (int i = 0; i < 4; i++) {
        V16 wl, wh;
        q4_magic(R[i] + k0 / 2, s[i], wl, wh);
        a[i] = v_fma(wl, x0, a[i]);
        a[i] = v_fma(wh, x1, a[i]);
      }
    }
  }
  for (int i = 0; i < 4; i++) out[i] = tree16(a[i]);
}

// V3: V2 + software prefetch g_pf bytes ahead in each row (one prefetch per 64-byte line)
static int g_pf = 512;
static void v3(const uint8_t* const* R, const float* X, int K, int G, float* out) {
  V16 a[4] = {v_zero(), v_zero(), v_zero(), v_zero()};
  for (int g0 = 0, gi = 0; g0 < K; g0 += G, gi++) {
    __m256 s[4];
    for (int i = 0; i < 4; i++) s[i] = _mm256_set1_ps(bf2f(((const uint16_t*)(R[i] + K / 2))[gi]));
    for (int k0 = g0; k0 < g0 + G; k0 += 32) {
      if ((k0 & 127) == 0)
        for (int i = 0; i < 4; i++) _mm_prefetch((const char*)(R[i] + k0 / 2 + g_pf), _MM_HINT_T0);
      const V16 x0 = v_loadf(X + k0), x1 = v_loadf(X + k0 + 16);
      for (int i = 0; i < 4; i++) {
        V16 wl, wh;
        q4_magic(R[i] + k0 / 2, s[i], wl, wh);
        a[i] = v_fma(wl, x0, a[i]);
        a[i] = v_fma(wh, x1, a[i]);
      }
    }
  }
  for (int i = 0; i < 4; i++) out[i] = tree16(a[i]);
}

typedef void (*Fn)(const uint8_t* const*, const float*, int, int, float*);

static int mt_main(int nthr, int f, int K) {
  const int G = 128, RB = K / 2 + 2 * (K / G);
  const size_t rows = (size_t)24 * 768 * 8 * 2048 / K;          // ~ 8 experts x 3 matrices of a Qwen3-30B layer, x24 layers (~0.4 GB)
  std::vector<uint8_t> W(rows * RB);
  for (size_t i = 0; i < W.size(); i += 64) W[i] = (uint8_t)i;
  for (size_t r = 0; r < rows; r++) { uint16_t* sc = (uint16_t*)(W.data() + r * RB + K / 2); for (int g = 0; g < K / G; g++) sc[g] = 0x3c00; }
  static float X[2048]; for (int k = 0; k < K; k++) X[k] = 0.001f * (k % 7);
  std::vector<float> out(rows);
  Fn fns[] = {v0, v1, v2, v3}; const int nr[] = {4, 4, 4, 4};
  double best = 1e9;
  for (int rep = 0; rep < 5; rep++) {
    std::atomic<size_t> next{0};
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> th;
    for (int t = 0; t < nthr; t++) th.emplace_back([&] {
      const size_t chunk = 64;
      for (size_t c = next.fetch_add(chunk); c < rows; c = next.fetch_add(chunk))
        for (size_t r = c; r + nr[f] <= c + chunk && r + nr[f] <= rows; r += nr[f]) {
          const uint8_t* R[8]; for (int i = 0; i < nr[f]; i++) R[i] = W.data() + (r + i) * RB;
          fns[f](R, X, K, G, &out[r]);
        }
    });
    for (auto& x : th) x.join();
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (dt < best) best = dt;
  }
  printf("K %4d threads %2d variant V%d: %.1f GB/s of q4 records (%.1f ms for %.0f MB)\n", K, nthr, f, W.size() / best / 1e9, best * 1e3, W.size() / 1e6);
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 4) g_pf = atoi(argv[4]);
  if (argc > 2) return mt_main(atoi(argv[1]), atoi(argv[2]), argc > 3 ? atoi(argv[3]) : 2048);
  const int K = 2048, G = 128, RB = K / 2 + 2 * (K / G);
  float X[K];
  for (int k = 0; k < K; k++) X[k] = bf2f((uint16_t)(0x3c00 + (rand() & 0x3ff)) ^ ((rand() & 1) << 15));
  for (int big = 0; big < 2; big++) {
    const size_t rows = big ? 400000 : 128;     // ~0.4 GB vs ~135 KB
    std::vector<uint8_t> W(rows * RB);
    for (size_t i = 0; i < W.size(); i++) W[i] = (uint8_t)rand();
    for (size_t r = 0; r < rows; r++) {
      uint16_t* sc = (uint16_t*)(W.data() + r * RB + K / 2);
      for (int g = 0; g < K / G; g++) sc[g] = (uint16_t)(0x3a00 + (rand() & 0x1ff));
    }
    const char* names[] = {"V0 1.3", "V1 hoist", "V2 magic", "V3 magic+pf"};
    Fn fns[] = {v0, v1, v2, v3};
    const int nr[] = {4, 4, 4, 4};
    std::vector<float> ref(rows);
    for (int f = 0; f < 4; f++) {
      std::vector<float> out(rows);
      const int reps = big ? 2 : 4000;
      uint64_t best = ~0ull;
      for (int rep = 0; rep < reps; rep++) {
        const uint64_t t0 = __rdtsc();
        for (size_t r = 0; r + nr[f] <= rows; r += nr[f]) {
          const uint8_t* R[8];
          for (int i = 0; i < nr[f]; i++) R[i] = W.data() + (r + i) * RB;
          fns[f](R, X, K, G, &out[r]);
        }
        const uint64_t dt = __rdtsc() - t0;
        if (dt < best) best = dt;
      }
      if (f == 0) ref = out;
      const bool same = std::memcmp(ref.data(), out.data(), rows * 4) == 0;
      printf("%s %-12s %6.2f TSC cycles / 32 weights   same bits as V0: %s\n", big ? "DRAM" : "L2  ", names[f],
             (double)best / (rows * (K / 32.0)), same ? "yes" : "NO");
    }
  }
}
