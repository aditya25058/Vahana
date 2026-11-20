// vt.h: the small part of the torch::Tensor API the Vahana kernel uses, without libtorch (P3).
// The kernel builds as a plain C++ shared library with a C ABI (capi in cpu_moe.cpp); Python passes tensors as
// descriptors (pointer, dtype, sizes, strides). So one binary works with any torch version, on any OS.
// Tensors made here own their memory (shared_ptr); tensors from descriptors borrow Python's memory.
#pragma once
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vt {
// dtype codes shared with vahana/_kernel.py
enum ScalarType : int { kFloat = 0, kBFloat16 = 1, kLong = 2, kInt = 3, kUInt8 = 4, kInt8 = 5, kHalf = 6, kDouble = 7 };
static constexpr int kFloat32 = kFloat, kInt64 = kLong, kInt32 = kInt;
inline size_t esize(int t) {
  switch (t) {
    case kFloat: case kInt: return 4;
    case kBFloat16: case kHalf: return 2;
    case kLong: case kDouble: return 8;
    default: return 1;
  }
}
inline void msg_(std::ostringstream&) {}
template <typename A, typename... R> inline void msg_(std::ostringstream& s, const A& a, const R&... r) { s << a; msg_(s, r...); }
template <typename... R> [[noreturn]] inline void fail(const char* cond, const R&... r) {
  std::ostringstream s; s << "check failed: " << cond; if (sizeof...(r)) { s << ": "; msg_(s, r...); }
  throw std::runtime_error(s.str());
}
struct TensorOptions {
  int dt = kFloat;
  TensorOptions dtype(int d) const { TensorOptions o; o.dt = d; return o; }
};
struct Tensor {
  std::shared_ptr<char> own;
  void* p = nullptr;
  int dt = kFloat, nd = 0;
  int64_t sz[8] = {0}, st[8] = {0};
  bool defined() const { return p != nullptr || nd > 0; }
  void* data_ptr() const { return p; }
  template <typename T> T* data_ptr() const { return (T*)p; }
  int dim() const { return nd; }
  int64_t size(int i) const { return sz[i < 0 ? i + nd : i]; }
  int64_t stride(int i) const { return st[i < 0 ? i + nd : i]; }
  int64_t numel() const { int64_t n = 1; for (int i = 0; i < nd; i++) n *= sz[i]; return n; }
  size_t element_size() const { return esize(dt); }
  int scalar_type() const { return dt; }
  TensorOptions options() const { TensorOptions o; o.dt = dt; return o; }
  bool is_contiguous() const {
    int64_t s = 1;
    for (int i = nd - 1; i >= 0; i--) { if (sz[i] != 1 && st[i] != s) return false; s *= sz[i]; }
    return true;
  }
  Tensor contiguous() const;
  Tensor view(std::initializer_list<int64_t> shape) const {
    if (!is_contiguous()) fail("view of a non-contiguous tensor");
    Tensor t = *this; t.nd = (int)shape.size(); int i = 0; for (int64_t s : shape) t.sz[i++] = s;
    int64_t s = 1; for (int k = t.nd - 1; k >= 0; k--) { t.st[k] = s; s *= t.sz[k]; }
    if (t.numel() != numel()) fail("view: numel mismatch");
    return t;
  }
  Tensor to(int dtype) const;
  Tensor& copy_(const Tensor& o) {
    if (o.dt != dt || o.numel() != numel() || !is_contiguous() || !o.is_contiguous()) fail("copy_: unsupported");
    std::memcpy(p, o.p, (size_t)numel() * esize(dt));
    return *this;
  }
};
inline Tensor empty(const std::vector<int64_t>& shape, TensorOptions o) {
  Tensor t; t.dt = o.dt; t.nd = (int)shape.size();
  int64_t s = 1;
  for (int k = t.nd - 1; k >= 0; k--) { t.sz[k] = shape[k]; t.st[k] = s; s *= shape[k]; }
  const size_t bytes = (size_t)(s > 0 ? s : 1) * esize(o.dt) + 64;
  t.own = std::shared_ptr<char>(new char[bytes], std::default_delete<char[]>());
  t.p = (void*)(((uintptr_t)t.own.get() + 63) & ~(uintptr_t)63);     // 64-byte aligned
  return t;
}
inline Tensor empty(std::initializer_list<int64_t> shape, TensorOptions o) { return empty(std::vector<int64_t>(shape), o); }
inline Tensor zeros(std::initializer_list<int64_t> shape, TensorOptions o) {
  Tensor t = empty(shape, o); std::memset(t.p, 0, (size_t)t.numel() * esize(o.dt)); return t;
}
inline Tensor empty_like(const Tensor& x) {
  std::vector<int64_t> s(x.sz, x.sz + x.nd); return empty(s, x.options());
}
inline Tensor tensor(const std::vector<int64_t>& v) {
  Tensor t = empty({(int64_t)v.size()}, TensorOptions().dtype(kLong));
  if (!v.empty()) std::memcpy(t.p, v.data(), v.size() * 8);
  return t;
}
inline Tensor Tensor::contiguous() const {
  if (is_contiguous()) return *this;
  std::vector<int64_t> s(sz, sz + nd);
  Tensor t = empty(s, options());
  const size_t es = esize(dt); const int64_t n = numel();
  std::vector<int64_t> idx(nd, 0);
  for (int64_t i = 0; i < n; i++) {
    int64_t off = 0; for (int k = 0; k < nd; k++) off += idx[k] * st[k];
    std::memcpy((char*)t.p + (size_t)i * es, (const char*)p + (size_t)off * es, es);
    for (int k = nd - 1; k >= 0; k--) { if (++idx[k] < sz[k]) break; idx[k] = 0; }
  }
  return t;
}
inline Tensor Tensor::to(int dtype) const {
  if (dtype == dt) return *this;
  Tensor c = contiguous();
  std::vector<int64_t> s(sz, sz + nd);
  Tensor t = empty(s, TensorOptions().dtype(dtype));
  const int64_t n = numel();
  if (dt == kBFloat16 && dtype == kFloat) {
    const uint16_t* a = (const uint16_t*)c.p; float* b = (float*)t.p;
    for (int64_t i = 0; i < n; i++) { uint32_t u = (uint32_t)a[i] << 16; std::memcpy(&b[i], &u, 4); }
  } else if (dt == kFloat && dtype == kBFloat16) {
    const float* a = (const float*)c.p; uint16_t* b = (uint16_t*)t.p;
    for (int64_t i = 0; i < n; i++) {
      uint32_t u; std::memcpy(&u, &a[i], 4);
      b[i] = (u & 0x7fffffffu) > 0x7f800000u ? (uint16_t)((u >> 16) | 0x40) : (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
  } else if (dt == kDouble && dtype == kFloat) {
    const double* a = (const double*)c.p; float* b = (float*)t.p;
    for (int64_t i = 0; i < n; i++) b[i] = (float)a[i];
  } else if (dt == kHalf && dtype == kFloat) {
    fail("to: half -> float is converted on the Python side");
  } else {
    fail("to: unsupported dtype conversion ", dt, " -> ", dtype);
  }
  return t;
}
}  // namespace vt

namespace torch {
using vt::Tensor; using vt::TensorOptions; using vt::empty; using vt::zeros; using vt::empty_like; using vt::tensor;
using vt::kFloat; using vt::kBFloat16; using vt::kLong; using vt::kInt; using vt::kUInt8; using vt::kInt8;
using vt::kHalf; using vt::kDouble; using vt::kFloat32; using vt::kInt64; using vt::kInt32;
}
#define TORCH_CHECK(cond, ...) do { if (!(cond)) vt::fail(#cond, ##__VA_ARGS__); } while (0)
