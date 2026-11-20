"""P1: one kernel source for every CPU. V16 = 16 fp32 lanes with the same fused-multiply-add per lane on every ISA:
AVX-512 (one register), AVX2+FMA (two 8-wide halves), ARM NEON (four 4-wide parts), scalar (std::fma). tree16 and
everything else is shared code, so every build computes the same bits. NX12 decode: AVX-512 VBMI byte permutes when
available, a portable decoder otherwise. isa() / nx_fast() report what was compiled."""
import re
import sys

p = sys.argv[1]
s = open(p).read()


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a[:90])
    s = s.replace(a, b)


# ---- includes: x86 intrinsics only on x86
rep("#include <immintrin.h>\n", """#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif
""")

# ---- V16 primitives replace the AVX-512-only ones
old = s[s.index("static inline __m512 ld_bf16(const uint16_t* p) {"):s.index("// dots of one weight row (K bf16) with n fp32 vectors X[t*K ..]")]
new = r'''// ---- V16: 16 fp32 lanes; lane l of a dot product accumulates k = 16j + l in ascending j with a fused multiply-add.
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
#if defined(__GNUC__) || defined(__clang__)
#define VH_PREFETCH(p) __builtin_prefetch((const void*)(p))
#else
#define VH_PREFETCH(p) ((void)0)
#endif
'''
s = s.replace(old, new)
rep("""  __m512 acc[16];
  for (int t = 0; t < n; t++) acc[t] = _mm512_setzero_ps();
  for (int k = 0; k < K; k += 16) {
    __m512 wv = ld_bf16(w + k);
    for (int t = 0; t < n; t++) acc[t] = _mm512_fmadd_ps(wv, _mm512_loadu_ps(X + (size_t)t * K + k), acc[t]);
  }""", """  V16 acc[16];
  for (int t = 0; t < n; t++) acc[t] = v_zero();
  for (int k = 0; k < K; k += 16) {
    V16 wv = v_ldbf16(w + k);
    for (int t = 0; t < n; t++) acc[t] = v_fma(wv, v_loadf(X + (size_t)t * K + k), acc[t]);
  }""")
rep("""  __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
  for (int k = 0; k < K; k += 16) {
    __m512 xv = _mm512_loadu_ps(X + k);
    a0 = _mm512_fmadd_ps(ld_bf16(w0 + k), xv, a0);
    a1 = _mm512_fmadd_ps(ld_bf16(w1 + k), xv, a1);
    a2 = _mm512_fmadd_ps(ld_bf16(w2 + k), xv, a2);
    a3 = _mm512_fmadd_ps(ld_bf16(w3 + k), xv, a3);
  }""", """  V16 a0 = v_zero(), a1 = v_zero(), a2 = v_zero(), a3 = v_zero();
  for (int k = 0; k < K; k += 16) {
    V16 xv = v_loadf(X + k);
    a0 = v_fma(v_ldbf16(w0 + k), xv, a0);
    a1 = v_fma(v_ldbf16(w1 + k), xv, a1);
    a2 = v_fma(v_ldbf16(w2 + k), xv, a2);
    a3 = v_fma(v_ldbf16(w3 + k), xv, a3);
  }""")

# ---- NX12: VBMI decoder when available, portable decoder otherwise
rep("struct NXm { const uint8_t* lo; const uint8_t* nib; const uint8_t* esc; const int64_t* off; __m512i lut; int K; };",
    """#if defined(__AVX512VBMI__)
#define VH_NX_FAST 1
struct NXm { const uint8_t* lo; const uint8_t* nib; const uint8_t* esc; const int64_t* off; __m512i lut; int K;
             const uint8_t* lutp; };
#else
#define VH_NX_FAST 0
struct NXm { const uint8_t* lo; const uint8_t* nib; const uint8_t* esc; const int64_t* off; int K; const uint8_t* lutp; };
#endif""")
rep("""  m.lut = _mm512_zextsi128_si512(_mm_loadu_si128((const __m128i*)t[2].data_ptr<uint8_t>())); m.K = K;""",
    """  m.lutp = t[2].data_ptr<uint8_t>(); m.K = K;
#if VH_NX_FAST
  m.lut = _mm512_zextsi128_si512(_mm_loadu_si128((const __m128i*)t[2].data_ptr<uint8_t>()));
#endif""")
i0 = s.index("static __m512i g_plo, g_phi;")
i1 = s.index("// four rows, one token, same per-row arithmetic as row_dots4_n1")
vbmi = s[i0:i1]
generic = r'''#else
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
'''
# g_pf is declared inside the VBMI block; keep a single declaration for both
vbmi_new = vbmi.replace("static __m512i g_plo, g_phi;\nstatic int g_pf = 0;\n", "static int g_pf = 0;\n#if VH_NX_FAST\nstatic __m512i g_plo, g_phi;\n", 1)
assert vbmi_new != vbmi, "g_pf declaration not found"
vbmi_new = vbmi_new.replace("__m512* w) {", "V16* w) {", 1)
s = s[:i0] + vbmi_new + generic + s[i1:]
rep("""  __m512 a[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};""",
    """  V16 a[4] = {v_zero(), v_zero(), v_zero(), v_zero()};""")
rep("""  __m512 w[4];
  for (int k = 0; k < K; k += 64) {""", """  V16 w[4];
  for (int k = 0; k < K; k += 64) {""")
rep("""        _mm_prefetch((const char*)(m4[q]->lo + rb[q] + k + pf), _MM_HINT_T0);
        _mm_prefetch((const char*)(m4[q]->nib + (rb[q] + k + pf) / 2), _MM_HINT_T0);""",
    """        VH_PREFETCH(m4[q]->lo + rb[q] + k + pf);
        VH_PREFETCH(m4[q]->nib + (rb[q] + k + pf) / 2);""")
rep("""    const __m512 x0 = _mm512_loadu_ps(X + k), x1 = _mm512_loadu_ps(X + k + 16),
                 x2 = _mm512_loadu_ps(X + k + 32), x3 = _mm512_loadu_ps(X + k + 48);""",
    """    const V16 x0 = v_loadf(X + k), x1 = v_loadf(X + k + 16), x2 = v_loadf(X + k + 32), x3 = v_loadf(X + k + 48);""")
rep("""      a[q] = _mm512_fmadd_ps(w[0], x0, a[q]);
      a[q] = _mm512_fmadd_ps(w[1], x1, a[q]);
      a[q] = _mm512_fmadd_ps(w[2], x2, a[q]);
      a[q] = _mm512_fmadd_ps(w[3], x3, a[q]);""", """      a[q] = v_fma(w[0], x0, a[q]);
      a[q] = v_fma(w[1], x1, a[q]);
      a[q] = v_fma(w[2], x2, a[q]);
      a[q] = v_fma(w[3], x3, a[q]);""")
rep("while (done[j].v.load(std::memory_order_acquire) < Is[j] / 16) _mm_pause();",
    "while (done[j].v.load(std::memory_order_acquire) < Is[j] / 16) VH_PAUSE();")
rep("""#if defined(__GNUC__) || defined(__clang__)
#define VH_PREFETCH(p) __builtin_prefetch((const void*)(p))""", """#if defined(__x86_64__) || defined(_M_X64)
#define VH_PAUSE() _mm_pause()
#elif defined(__aarch64__)
#define VH_PAUSE() __asm__ __volatile__("yield")
#else
#define VH_PAUSE() ((void)0)
#endif
#if defined(__GNUC__) || defined(__clang__)
#define VH_PREFETCH(p) __builtin_prefetch((const void*)(p))""")
# read_bw: portable 64-bit sum (a bandwidth probe; the value only defeats dead-code elimination)
i0 = s.index("    __m512i acc = _mm512_setzero_si512();")
i1 = s.index("    sink += _mm512_reduce_add_epi64(acc);") + len("    sink += _mm512_reduce_add_epi64(acc);")
blk = s[i0:i1]
blk2 = blk.replace("    __m512i acc = _mm512_setzero_si512();", "    uint64_t acc[4] = {0, 0, 0, 0};")
for off, q in (("", 0), (" + 64", 1), (" + 128", 2), (" + 192", 3)):
    a = f"acc = _mm512_add_epi64(acc, _mm512_loadu_si512(p + o{off}));"
    b = (f"{{ uint64_t t_[8]; std::memcpy(t_, p + o{off}, 64); for (int z = 0; z < 8; z++) acc[{q}] += t_[z]; }}")
    assert blk2.count(a) == 1, a
    blk2 = blk2.replace(a, b)
blk2 = blk2.replace("    sink += _mm512_reduce_add_epi64(acc);", "    sink += (long long)(acc[0] + acc[1] + acc[2] + acc[3]);")
s = s[:i0] + blk2 + s[i1:]
# report what was compiled
s = s.replace("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n",
              "std::string isa() { return VH_ISA; }\nbool nx_fast() { return VH_NX_FAST != 0; }\n"
              "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n  m.def(\"isa\", &isa);\n  m.def(\"nx_fast\", &nx_fast);\n", 1)
open(p, "w").write(s)
left = [ln for ln in s.splitlines() if re.search(r"_mm512|__m512|_mm256|_mm_", ln)]
print("P1 patch applied; remaining x86-only lines:", len(left))
for ln in left:
    print("   ", ln.strip()[:110])
