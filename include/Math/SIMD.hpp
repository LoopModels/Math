#pragma once
#ifndef SIMD_hpp_INCLUDED
#define SIMD_hpp_INCLUDED

#include <Math/AxisTypes.hpp>
#include <Utilities/Invariant.hpp>
#include <Utilities/LoopMacros.hpp>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>

#ifdef __x86_64__
#include <immintrin.h>
#endif
namespace poly::math::simd {

template <ptrdiff_t W, typename T>
using Vec [[gnu::vector_size(W * sizeof(T))]] = T;

// Supported means by this library currently; more types may be added in the
// future as needed.
template <typename T>
concept SIMDSupported = std::same_as<T, int64_t> || std::same_as<T, double>;

namespace mask {
template <ptrdiff_t W> struct None {
  static constexpr auto lastUnmasked() -> ptrdiff_t { return W; }
};

// Alternatives we can have: BitMask and VectorMask
} // namespace mask

// template <ptrdiff_t W, typename T> auto zero()->Vec<W,T>{return Vec<W,T>{};}

template <ptrdiff_t W,
          typename I = std::conditional_t<W == 2, int64_t, int32_t>>
consteval auto range() -> Vec<W, I> {
  static_assert(std::popcount(size_t(W)) == 1);
  if constexpr (W == 2) return Vec<W, I>{0, 1};
  else if constexpr (W == 4) return Vec<W, I>{0, 1, 2, 3};
  else if constexpr (W == 8) return Vec<W, I>{0, 1, 2, 3, 4, 5, 6, 7};
  else if constexpr (W == 16)
    return Vec<W, I>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  else {
    Vec<W, I> r;
    for (ptrdiff_t w = 0; w < W; ++w) r[w] = I(w);
    return r;
  }
}

#ifdef __x86_64__

// TODO: make `consteval` when clang supports it
template <ptrdiff_t W, typename T> consteval auto mmzero() {
  // Extend if/when supporting more types
  static_assert(std::popcount(size_t(W)) == 1 && W <= 8);
  constexpr Vec<W, T> z{};
  if constexpr (std::same_as<T, double>) {
    if constexpr (W == 8) return std::bit_cast<__m512d>(z);
    else if constexpr (W == 4) return std::bit_cast<__m256d>(z);
    else return std::bit_cast<__m128d>(z);
  } else {
    static_assert(std::same_as<T, int64_t>);
    if constexpr (W == 8) return std::bit_cast<__m512i>(z);
    else if constexpr (W == 4) return std::bit_cast<__m256i>(z);
    else return std::bit_cast<__m128i>(z);
  }
}
template <ptrdiff_t W> inline auto vindex(int32_t stride) {
  if constexpr (W <= 4) return std::bit_cast<__m128i>(range<W>() * stride);
  else if constexpr (W == 8) return std::bit_cast<__m256i>(range<W>() * stride);
  else {
    static_assert(W == 16);
    return std::bit_cast<__m512i>(range<W>() * stride);
  }
}
#ifdef __AVX512F__
namespace mask {
template <ptrdiff_t W> struct Bit {
  uint64_t mask;
  template <std::unsigned_integral U> explicit constexpr operator U() {
    return U(m);
  }
  explicit constexpr operator bool() const { return m; }
  [[nodiscard]] constexpr auto lastUnmasked() const -> ptrdiff_t {
    // could make this `countr_ones` if we decide to only
    // support leading masks
    uint64_t m = mask & ((uint64_t(1) << W) - uint64_t(1));
    reutrn 64 - ptrdiff_t(std::countl_zeros(m));
  }
};
template <ptrdiff_t W> constexpr auto operator&(Bit<W> a, Bit<W> b) -> Bit<W> {
  return {a.mask & b.mask};
}
template <ptrdiff_t W> constexpr auto operator&(None<W> a, Bit<W> b) -> Bit<W> {
  return b;
}
template <ptrdiff_t W> constexpr auto operator&(Bit<W> a, None<W> b) -> Bit<W> {
  return a;
}
} // namespace mask

#endif
#ifndef __AVX512VL__
namespace mask {

template <ptrdiff_t W, typename I = int64_t> struct Vector {
  static_assert((W == 2) || (W == 4));
  // TODO: add support for smaller mask types, we we can use smaller eltypes
  Vec<W, I> m;
  [[nodiscard]] constexpr auto intmask() const -> int32_t {
    if constexpr (sizeof(I) == 8)
      if constexpr (W == 2) return _mm_movemask_pd(std::bit_cast<__m128i>(m));
      else return _mm256_movemask_pd(std::bit_cast<__m256i>(m));
    else if constexpr (sizeof(I) == 4)
      if constexpr (W == 4) return _mm_movemask_ps(std::bit_cast<__m128i>(m));
      else return _mm256_movemask_ps(std::bit_cast<__m256i>(m));
    else if constexpr (W == 16)
      return _mm_movemask_epi8(std::bit_cast<__m128i>(m));
    else return _mm256_movemask_epi8(std::bit_cast<__m256i>(m));
  }
  explicit constexpr operator bool() const { return intmask(); }
  [[nodiscard]] constexpr auto lastUnmasked() const -> ptrdiff_t {
    return 32 - std::countl_zero(uint32_t(intmask()));
  }
  constexpr operator __m128i()
  requires(W == 2)
  {
    return std::bit_cast<__m128i>(m);
  }
  constexpr operator __m256i()
  requires(W == 4)
  {
    return std::bit_cast<__m256i>(m);
  }
};
template <ptrdiff_t W>
constexpr auto operator&(Vector<W> a, Vector<W> b) -> Vector<W> {
  return {a.m & b.m};
}
template <ptrdiff_t W>
constexpr auto operator&(mask::None<W>, Vector<W> b) -> Vector<W> {
  return b;
}
template <ptrdiff_t W>
constexpr auto operator&(Vector<W> a, mask::None<W>) -> Vector<W> {
  return a;
}
#ifdef __AVX512F__
template <ptrdiff_t W> constexpr auto create(ptrdiff_t i) {
  if constexpr (W == 8)
    return Bit<8>{_bzhi_u64(0xffffffffffffffff, uint64_t(i) & uint64_t(7))};
  else return Vector<W>{range<W, int64_t>() < (i & (W - 1))};
}
template <ptrdiff_t W> constexpr auto create(ptrdiff_t i, ptrdiff_t len) {
  if constexpr (W == 8)
    return Bit<8>{_bzhi_u64(0xffffffffffffffff, uint64_t(len - i))};
  else return Vector<W>{range<W, int64_t>() + i < len};
}
#else
template <ptrdiff_t W> constexpr auto create(ptrdiff_t i) -> Vector<W> {
  return {range<W, int64_t>() < (i & (W - 1))};
}
template <ptrdiff_t W>
constexpr auto create(ptrdiff_t i, ptrdiff_t len) -> Vector<W> {
  return {range<W, int64_t>() + i < len};
}
#endif

} // namespace mask
#endif

#ifdef __AVX512F__
static constexpr ptrdiff_t REGISTERS = 32;
static constexpr ptrdiff_t VECTORWIDTH = 64;

#ifdef __AVX512VL__
// if not VL, we defined `create` above
namespace mask {

// In: iteration count `i.i` is the total length of the loop
// Out: mask for the final iteration. Zero indicates no masked iter.
template <ptrdiff_t W> constexpr auto create(ptrdiff_t i) -> Bit<W> {
  static_assert(std::popcount(size_t(W)) == 1);
  invariant(i >= 0);
  return {_bzhi_u64(0xffffffffffffffff, uint64_t(i) & uint64_t(W - 1))};
};
// In: index::Vector where `i.i` is for the current iteration, and total loop
// length. Out: mask for the current iteration, 0 indicates exit loop.
template <ptrdiff_t W>
constexpr auto create(ptrdiff_t i, ptrdiff_t len) -> Bit<W> {
  static_assert(std::popcount(size_t(W)) == 1);
  // invariant(len >= i);
  ptrdiff_t srem = len - i;
  return {_bzhi_u64(0xffffffffffffffff, uint64_t(srem))};
};
} // namespace mask
#endif

[[gnu::always_inline, gnu::artificial]] inline auto load(const double *p,
                                                         mask::None<8>)
  -> Vec<8, double> {
  return std::bit_cast<Vec<8, double>>(_mm512_loadu_pd(p));
}
[[gnu::always_inline, gnu::artificial]] inline auto load(const double *p,
                                                         mask::Bit<8> i)
  -> Vec<8, double> {
  return std::bit_cast<Vec<8, double>>(
    _mm512_maskz_loadu_pd(uint8_t(i.mask), p));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(const double *p, mask::None<8>, Vec<8, double> x) {
  _mm512_storeu_pd(p + i.i, std::bit_cast<__m512d>(x));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, mask::Bit<8> i, Vec<8, double> x) {
  _mm512_mask_storeu_pd(p, uint8_t(i.mask), std::bit_cast<__m512d>(x));
}

[[gnu::always_inline, gnu::artificial]] inline auto load(const int64_t *p,
                                                         mask::None<8>) {
  return std::bit_cast<Vec<8, int64_t>>(_mm512_loadu_epi64(p));
}
[[gnu::always_inline, gnu::artificial]] inline auto load(const int64_t *p,
                                                         mask::Bit<8> i)
  -> Vec<8, int64_t> {
  return std::bit_cast<Vec<8, int64_t>>(
    _mm512_maskz_loadu_epi64(uint8_t(i.mask), p));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, mask::None<8>, Vec<8, int64_t> x)
  ->Vec<8, int64_t> {
  _mm512_storeu_epi64(p, std::bit_cast<__m512i>(x));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, mask::Bit<8> i, Vec<8, int64_t> x) {
  _mm512_mask_storeu_epi64(p, uint8_t(i.mask), std::bit_cast<__m512i>(x));
}
// strided memory accesses
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, mask::None<8>, int32_t stride) -> Vec<8, double> {
  return std::bit_cast<Vec<8, double>>(
    _mm512_i32gather_pd(vindex<8>(stride), p, 8));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, mask::Bit<8> i, int32_t stride) -> Vec<8, double> {
  return std::bit_cast<Vec<8, double>>(_mm512_mask_i32gather_pd(
    mmzero<8, double>(), uint8_t(i.mask), vindex<8>(stride), p, 8));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(const double *p, mask::None<8>, Vec<8, double> x, int32_t stride) {
  _mm512_i32scatter_pd(p, vindex<8>(stride), std::bit_cast<__m512d>(x), 8);
}
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, mask::Bit<8> i, Vec<8, double> x, int32_t stride) {
  _mm512_mask_i32scatter_pd(p, uint8_t(i.mask), vindex<8>(stride),
                            std::bit_cast<__m512d>(x), 8);
}

[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, mask::None<8> i, int32_t stride) -> Vec<8, int64_t> {
  return std::bit_cast<Vec<8, double>>(
    _mm512_i32gather_pd(vindex<8>(stride), p, 8));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, mask::Bit<8> i, int32_t stride) -> Vec<8, int64_t> {
  return std::bit_cast<Vec<8, int64_t>>(_mm512_mask_i32gather_epi64(
    mmzero<8, int64_t>(), uint8_t(i.mask), vindex<8>(stride), p, 8));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, mask::None<8> i, Vec<8, int64_t> x, int32_t stride) {
  _mm512_i32scatter_epi64(p, vindex<8>(stride), std::bit_cast<__m512i>(x), 8);
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, mask::Bit<8> i, Vec<8, int64_t> x, int32_t stride) {
  _mm512_mask_i32scatter_epi64(p, uint8_t(i.mask), vindex<8>(stride),
                               std::bit_cast<__m512i>(x), 8);
}
#else // no AVX512F
static constexpr ptrdiff_t REGISTERS = 16;
#ifdef __AVX__
static constexpr ptrdiff_t VECTORWIDTH = 32;
#else  // no AVX
static constexpr ptrdiff_t VECTORWIDTH = 16;
#endif // no AVX
#endif // no AVX512F
#ifdef __AVX2__

// Non-masked gather/scatter are the same with AVX512VL and AVX2
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, mask::None<4>, int32_t stride) -> Vec<4, double> {
  auto x{vindex<4>(stride)};
  return std::bit_cast<Vec<4, double>>(_mm256_i32gather_pd(p, x, 8));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, mask::None<2>, int32_t stride) -> Vec<2, double> {
  auto x{vindex<2>(stride)};
  return std::bit_cast<Vec<2, double>>(_mm_i64gather_pd(p, x, 8));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, mask::None<4>, int32_t stride) -> Vec<4, int64_t> {
  auto x{vindex<4>(stride)};
  return std::bit_cast<Vec<4, int64_t>>(_mm256_i32gather_pd(p, x, 8));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, mask::None<2>, int32_t stride) -> Vec<2, int64_t> {
  auto x{vindex<2>(stride)};
  return std::bit_cast<Vec<2, int64_t>>(_mm_i64gather_pd(p, x, 8));
}
#endif // AVX2

// Here, we handle masked loads/stores
#ifdef __AVX512VL__
[[gnu::always_inline, gnu::artificial]] inline auto load(const double *p,
                                                         mask::Bit<4> i) {
  return std::bit_cast<Vec<4, double>>(
    _mm256_maskz_loadu_pd(uint8_t(i.mask), p));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, mask::Bit<4> i, Vec<4, double> x) {
  _mm256_mask_storeu_pd(p, uint8_t(i.mask), std::bit_cast<__m256d>(x));
}
[[gnu::always_inline, gnu::artificial]] inline auto load(const int64_t *p,
                                                         mask::Bit<4> i) {
  return std::bit_cast<Vec<4, int64_t>>(
    _mm256_maskz_loadu_epi64(uint8_t(i.mask), p));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, mask::Bit<4> i, Vec<4, int64_t> x) {
  _mm256_mask_storeu_epi64(p, uint8_t(i.mask), std::bit_cast<__m256i>(x));
}

[[gnu::always_inline, gnu::artificial]] inline auto load(const double *p,
                                                         mask::Bit<2> i) {
  return std::bit_cast<Vec<2, double>>(_mm_maskz_loadu_pd(uint8_t(i.mask), p));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, mask::Bit<2> i, Vec<2, double> x) {
  _mm_mask_storeu_pd(p, uint8_t(i.mask), std::bit_cast<__128d>(x));
}
[[gnu::always_inline, gnu::artificial]] inline auto load(const int64_t *p,
                                                         mask::Bit<2> i) {
  return std::bit_cast<Vec<2, int64_t>>(
    _mm_maskz_loadu_epi64(uint8_t(i.mask), p));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, mask::Bit<2> i, Vec<2, int64_t> x) {
  _mm_mask_storeu_epi64(p, uint8_t(i.mask), std::bit_cast<__m128i>(x));
}
// gather/scatter
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, mask::Bit<4> i, int32_t stride) -> Vec<4, double> {
  return std::bit_cast<Vec<4, double>>(_mm256_mmask_i32gather_pd(
    mmzero<4, double>(), uint8_t(i.mask), vindex<4>(stride), p, 8));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(const double *p, mask::None<4>, Vec<4, double> x, int32_t stride) {
  _mm256_i32scatter_pd(p, vindex<4>(stride), std::bit_cast<__m256d>(x), 8);
}
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, mask::Bit<4> i, Vec<4, double> x, int32_t stride) {
  _mm256_mask_i32scatter_pd(p, uint8_t(i.mask), vindex<4>(stride),
                            std::bit_cast<__m256d>(x), 8);
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, mask::Bit<2> i, int32_t stride) -> Vec<2, double> {
  return std::bit_cast<Vec<2, double>>(_mm_mmask_i64gather_pd(
    mmzero<2, double>(), uint8_t(i.mask), vindex<2>(stride), p, 8));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(const double *p, mask::None<2>, Vec<2, double> x, int32_t stride) {
  _mm_i64scatter_pd(p, vindex<2>(stride), std::bit_cast<__m128d>(x), 8);
}
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, mask::Bit<2> i, Vec<2, double> x, int32_t stride) {
  _mm_mask_i64scatter_pd(p, uint8_t(i.mask), vindex<2>(stride),
                         std::bit_cast<__m128d>(x), 8);
}

[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, mask::Bit<4> i, int32_t stride) -> Vec<4, int64_t> {
  return std::bit_cast<Vec<4, int64_t>>(_mm256_mmask_i32gather_epi64(
    mmzero<4, int64_t>(), uint8_t(i.mask), vindex<4>(stride), p, 8));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, mask::None<4>, Vec<4, int64_t> x, int32_t stride) {
  _mm256_i32scatter_epi64(p, vindex<4>(stride), std::bit_cast<__m256i>(x), 8);
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, mask::Bit<4> i, Vec<4, int64_t> x, int32_t stride) {
  _mm256_mask_i32scatter_epi64(p, uint8_t(i.mask), vindex<4>(stride),
                               std::bit_cast<__m256i>(x), 8);
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, mask::Bit<2> i, int32_t stride) -> Vec<2, int64_t> {
  return std::bit_cast<Vec<2, int64_t>>(_mm_mmask_i64gather_epi64(
    mmzero<2, int64_t>(), uint8_t(i.mask), vindex<2>(stride), p, 8));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, mask::None<2>, Vec<2, int64_t> x, int32_t stride) {
  _mm_i64scatter_epi64(p, vindex<2>(stride), std::bit_cast<__m128i>(x), 8);
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, mask::Bit<2> i, Vec<2, int64_t> x, int32_t stride) {
  _mm_mask_i64scatter_epi64(p, uint8_t(i.mask), vindex<2>(stride),
                            std::bit_cast<__m128i>(x), 8);
}

[[gnu::always_inline, gnu::artificial]] inline auto load(const int64_t *p,
                                                         mask::None<4>) {
  return std::bit_cast<Vec<4, int64_t>>(_mm256_loadu_epi64(p));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, mask::None<4>, Vec<4, int64_t> x) {
  _mm256_storeu_epi64(p, std::bit_cast<__m256i>(x));
}
[[gnu::always_inline, gnu::artificial]] inline auto load(const int64_t *p,
                                                         mask::None<2>) {
  return std::bit_cast<Vec<2, int64_t>>(_mm_loadu_epi64(p));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, mask::None<2>, Vec<2, int64_t> x) {
  _mm_storeu_epi64(p, std::bit_cast<__m128i>(x));
}

#else // No AVX512VL

// We need [gather, scatter, load, store] * [unmasked, masked]

#ifdef __AVX__
// we need 256 bit fallback scatters
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::None<4>, Vec<4, T> x, int32_t stride) {
  p[0] = x[0];
  p[stride] = x[1];
  p[2 * stride] = x[2];
  p[3 * stride] = x[3];
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Vector<4> i, Vec<4, T> x, int32_t stride) {
  if (i.m[0] != 0) p[0] = x[0];
  if (i.m[1] != 0) p[stride] = x[1];
  if (i.m[2] != 0) p[2 * stride] = x[2];
  if (i.m[3] != 0) p[3 * stride] = x[3];
}

#endif // AVX
// 128 bit fallback scatters
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::None<2>, Vec<2, T> x, int32_t stride) {
  p[0] = x[0];
  p[stride] = x[1];
}

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Vector<2> i, Vec<2, T> x, int32_t stride) {
  if (i.m[0] != 0) p[0] = x[0];
  if (i.m[1] != 0) p[stride] = x[1];
}

#ifdef __AVX2__
// masked gathers
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, mask::Vector<4> m, int32_t stride) -> Vec<4, double> {
  constexpr __m256d z = mmzero<4, double>();
  __m128i x = vindex<4>(stride);
  __m256i mask = __m256i(m);
  return std::bit_cast<Vec<4, double>>(
    _mm256_mask_i32gather_pd(z, p, x, mask, 8));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, mask::Vector<2> m, int32_t stride) -> Vec<2, double> {
  constexpr __m128d z = mmzero<2, double>();
  __m128i x = vindex<2>(stride), mask = __m128i(m);
  return std::bit_cast<Vec<2, double>>(_mm_mask_i64gather_pd(z, p, x, mask, 8));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, mask::Vector<4> m, int32_t stride) -> Vec<4, int64_t> {
  constexpr __m256i z = mmzero<4, int64_t>();
  __m128i x = vindex<4>(stride);
  __m256i mask = __m256i(m);
  return std::bit_cast<Vec<4, int64_t>>(
    _mm256_mask_i32gather_epi64(z, p, x, mask, 8));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, mask::Vector<2> m, int32_t stride) -> Vec<2, int64_t> {
  constexpr __m128i z = mmzero<2, int64_t>();
  __m128i x = vindex<2>(stride), mask = __m128i(m);
  return std::bit_cast<Vec<2, int64_t>>(
    _mm_mask_i64gather_epi64(z, p, x, mask, 8));
}

#else          // no AVX2
// fallback 128-bit gather
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::None<2>, int32_t stride) -> Vec<2, T> {
  return Vec<2, T>{p[0], p[stride]};
}

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Vector<2> i, int32_t stride) -> Vec<2, T> {
  return Vec<2, T>{(i.m[0] != 0) ? p[0] : T{}, (i.m[1] != 0) ? p[stride] : T{}};
}
#ifdef __AVX__ // no AVX2, but AVX
// fallback 256-bit gather
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::None<4>, int32_t stride) -> Vec<4, T> {
  return Vec<4, T>{p[0], p[stride], p[2 * stride], p[3 * stride]};
}

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Vector<4> i, int32_t stride) -> Vec<4, T> {
  return Vec<4, T>{
    (i.m[0] != 0) ? p[0] : T{},
    (i.m[1] != 0) ? p[stride] : T{},
    (i.m[2] != 0) ? p[2 * stride] : T{},
    (i.m[3] != 0) ? p[3 * stride] : T{},
  };
}

#endif // AVX
#endif // no AVX2
#ifdef __AVX__
[[gnu::always_inline, gnu::artificial]] inline auto load(const double *p,
                                                         mask::Vector<4> i)
  -> Vec<4, double> {
  return std::bit_cast<Vec<4, double>>(_mm256_maskload_pd(p, i));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, mask::Vector<4> i, Vec<4, double> x) {
  _mm256_maskstore_pd(p, i, std::bit_cast<__m256d>(x));
}
[[gnu::always_inline, gnu::artificial]] inline auto load(const int64_t *p,
                                                         mask::Vector<4> i)
  -> Vec<4, int64_t> {
  return std::bit_cast<Vec<4, int64_t>>(
    _mm256_maskload_epi64((const long long *)p, i));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, mask::Vector<4> i, Vec<4, int64_t> x) {
  _mm256_maskstore_epi64((long long *)p, i, std::bit_cast<__m256i>(x));
}
[[gnu::always_inline, gnu::artificial]] inline auto load(const double *p,
                                                         mask::Vector<2> i)
  -> Vec<2, double> {
  return std::bit_cast<Vec<2, double>>(_mm_maskload_pd(p, i));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, mask::Vector<2> i, Vec<2, double> x) {
  _mm_maskstore_pd(p, i, std::bit_cast<__m128d>(x));
}
[[gnu::always_inline, gnu::artificial]] inline auto load(const int64_t *p,
                                                         mask::Vector<2> i)
  -> Vec<2, int64_t> {
  return std::bit_cast<Vec<2, int64_t>>(
    _mm_maskload_epi64((const long long *)p, i));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, mask::Vector<2> i, Vec<2, int64_t> x) {
  _mm_maskstore_epi64((long long *)p, i, std::bit_cast<__m128i>(x));
}

[[gnu::always_inline, gnu::artificial]] inline auto load(const int64_t *p,
                                                         mask::None<4>) {
  return std::bit_cast<Vec<4, int64_t>>(_mm256_loadu_si256((const __m256i *)p));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, mask::None<4>, Vec<4, int64_t> x) {
  _mm256_storeu_si256((__m256i *)p, std::bit_cast<__m256i>(x));
}
#else // No AVX
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto load(const T *p,
                                                         mask::Vector<2> i)
  -> Vec<2, T> {
  return Vec<2, T>{(i.m[0] != 0) ? p[0] : T{}, (i.m[1] != 0) ? p[1] : T{}};
}

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Vector<2> i, Vec<2, T> x) {
  if (i.m[0] != 0) p[0] = x[0];
  if (i.m[1] != 0) p[1] = x[1];
}

#endif // No AVX
#endif // No AVX512VL
#ifdef __AVX__
[[gnu::always_inline, gnu::artificial]] inline auto load(const double *p,
                                                         mask::None<4>) {
  return std::bit_cast<Vec<4, double>>(_mm256_loadu_pd(p));
}

[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, mask::None<4>, Vec<4, double> x) {
  _mm256_storeu_pd(p, std::bit_cast<__m256d>(x));
}

// // non-power-of-2 memory ops
// template <ptrdiff_t N, typename M>
// constexpr auto fixupnonpow2(index::Vector<N, M> i) {
//   static_assert(std::popcount(size_t(N)) > 1,
//                 "Shouldn't be calling this if not needed.");
//   static constexpr ptrdiff_t W = std::bit_ceil(size_t(N));
//   auto m = i.mask & mask(index::VectorMask<W>{N});
//   return index::Vector<W, decltype(m)>{i.i, m};
// }

// template <typename T, ptrdiff_t N, typename M>
// [[gnu::always_inline, gnu::artificial]] inline auto
// load(const T *p, index::Vector<N, M> i) {
//   return load(p, fixupnonpow2(i));
// }
// template <typename T, ptrdiff_t N, typename M>
// [[gnu::always_inline, gnu::artificial]] inline auto
// load(const T *p, index::Vector<N, M> i, int32_t stride) {
//   return load(p, fixupnonpow2(i), stride);
// }
// template <typename T, ptrdiff_t N, typename M, ptrdiff_t W>
// [[gnu::always_inline, gnu::artificial]] inline void
// store(T *p, index::Vector<N, M> i, Vec<W, T> x) {
//   store(p, fixupnonpow2(i), x);
// }
// template <typename T, ptrdiff_t N, typename M, ptrdiff_t W>
// [[gnu::always_inline, gnu::artificial]] inline void
// store(T *p, index::Vector<N, M> i, Vec<W, T> x, int32_t stride) {
//   store(p, fixupnonpow2(i), stride);
// }
#endif // AVX

[[gnu::always_inline, gnu::artificial]] inline auto load(const double *p,
                                                         mask::None<2>)
  -> Vec<2, double> {
  return std::bit_cast<Vec<2, double>>(_mm_loadu_pd(p));
}

[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, mask::None<2>, Vec<2, double> x) {
  _mm_storeu_pd(p, std::bit_cast<__m128d>(x));
}

[[gnu::always_inline, gnu::artificial]] inline auto load(const int64_t *p,
                                                         mask::None<2>) {
  return std::bit_cast<Vec<2, int64_t>>(_mm_loadu_si128((const __m128i *)p));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, mask::None<2>, Vec<2, int64_t> x) {
  _mm_storeu_si128((__m128i *)p, std::bit_cast<__m128i>(x));
}
#else // not __x86_64__
static constexpr ptrdiff_t REGISTERS = 32;
static constexpr ptrdiff_t VECTORWIDTH = 16;
namespace mask {
template <ptrdiff_t W> struct Vector {
  Vec<W, int64_t> m;
  explicit constexpr operator bool() {
    bool any{false};
    for (ptrdiff_t w = 0; w < W; ++w) any |= m[w];
    return any;
  }
  [[nodiscard]] constexpr auto lastUnmasked() const -> ptrdiff_t {
    ptrdiff_t l = 0;
    for (ptrdiff_t w = 0; w < W; ++w)
      if (m[w]) l = w;
    return l;
  }
};

template <ptrdiff_t W> constexpr auto create(ptrdiff_t i) -> Vector<W> {
  return {range<W, int64_t>() < (i & (W - 1))};
}
template <ptrdiff_t W>
constexpr auto create(ptrdiff_t i, ptrdiff_t len) -> Vector<W> {
  return {range<W, int64_t>() + i < len};
}
} // namespace mask
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto load(const T *p,
                                                         mask::None<W>)
  -> Vec<W, T> {
  Vec<W, T> ret;
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w) ret[w] = p[w];
  return ret;
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline void store(T *p, mask::None<W>,
                                                          Vec<W, T> x) {
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w) p[w] = x[w];
}

template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto load(const T *p,
                                                         mask::Vector<W> i)
  -> Vec<W, T> {
  Vec<W, T> ret;
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w) ret[w] = (i.m[w] != 0) ? p[w] : T{};
  return ret;
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Vector<W> i, Vec<W, T> x) {
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w)
    if (i.m[w] != 0) p[w] = x[w];
}

template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::None<W>, int32_t stride) -> Vec<W, T> {
  Vec<W, T> ret;
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w) ret[w] = p[w * stride];
  return ret;
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::None<W>, Vec<W, T> x, int32_t stride) {
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w) p[w * stride] = x[w];
}

template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Vector<W> i, int32_t stride) -> Vec<W, T> {
  Vec<W, T> ret;
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w)
    ret[w] = (i.m[w] != 0) ? p[w * stride] : T{};
  return ret;
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Vector<W> i, Vec<W, T> x, int32_t stride) {
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w)
    if (i.m[w] != 0) p[w * stride] = x[w];
}

#endif

template <typename T>
static constexpr ptrdiff_t Width = VECTORWIDTH / sizeof(T);

namespace index {
// Unroll rows by a factor of `R` and cols by `C`, vectorizing with width `W`
template <ptrdiff_t U, ptrdiff_t W = 1, typename M = mask::None<W>>
struct Unroll {
  ptrdiff_t index;
  [[no_unique_address]] M mask{};
  explicit constexpr operator ptrdiff_t() const { return index; }
  explicit constexpr operator bool() const { return bool(mask); }
};
template <ptrdiff_t U, ptrdiff_t W>
[[gnu::always_inline]] constexpr auto unrollmask(ptrdiff_t L, ptrdiff_t i) {
  // mask applies to last iter
  // We can't check that the last iter is non-empty, because that
  // could be the loop exit condition
  auto m{mask::create<W>(i + (U - 1) * W, L)};
  return Unroll<U, W, decltype(m)>{i, m};
};

// template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename M, ptrdiff_t X>
// constexpr auto unroll(ptrdiff_t index, M mask, RowStride<X> rs){
//   return Unroll<R,C,W,M,X>{index,mask,rs};
// }

template <ptrdiff_t R, ptrdiff_t C = 1, ptrdiff_t W = 1,
          typename M = mask::None<W>, bool Transposed = false, ptrdiff_t X = -1>
struct UnrollDims {
  [[no_unique_address]] M mask;
  [[no_unique_address]] RowStride<X> rs;
};

template <typename T> static constexpr bool issimd = false;

template <ptrdiff_t U, ptrdiff_t W, typename M>
static constexpr bool issimd<Unroll<U, W, M>> = true;
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename M, bool Transposed,
          ptrdiff_t X>
static constexpr bool issimd<UnrollDims<R, C, W, M, Transposed, X>> = true;

} // namespace index

// Vector goes across cols
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t N, typename T> struct Unroll {
  static constexpr ptrdiff_t W = ptrdiff_t(std::bit_ceil(size_t(N)));
  using VT = std::conditional_t<W == 1, T, Vec<W, T>>;
  VT data[R * C];
  constexpr auto operator[](ptrdiff_t i) -> VT & { return data[i]; }
  constexpr auto operator[](ptrdiff_t r, ptrdiff_t c) -> VT & {
    return data[r * C + c];
  }
  constexpr auto operator[](ptrdiff_t i) const -> VT { return data[i]; }
  constexpr auto operator[](ptrdiff_t r, ptrdiff_t c) const -> VT {
    return data[r * C + c];
  }

  [[gnu::always_inline]] constexpr auto operator-() {
    Unroll a;
    for (ptrdiff_t i = 0; i < R * C; ++i) a.data[i] = -data[i];
    return a;
  }
  [[gnu::always_inline]] constexpr auto operator+=(const Unroll &a)
    -> Unroll & {
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) data[i] += a.data[i];
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator-=(const Unroll &a)
    -> Unroll & {
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) data[i] -= a.data[i];
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator*=(const Unroll &a)
    -> Unroll & {
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) data[i] *= a.data[i];
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator/=(const Unroll &a)
    -> Unroll & {
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) data[i] /= a.data[i];
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator+=(VT a) -> Unroll & {
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) data[i] += a;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator-=(VT a) -> Unroll & {
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) data[i] -= a;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator*=(VT a) -> Unroll & {
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) data[i] *= a;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator/=(VT a) -> Unroll & {
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) data[i] /= a;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator+=(T a) -> Unroll &
  requires(W != 1)
  {
    return (*this) += (typename Unroll<R, C, W, T>::VT{} + a);
  }
  [[gnu::always_inline]] constexpr auto operator-=(T a) -> Unroll &
  requires(W != 1)
  {
    return (*this) -= (typename Unroll<R, C, W, T>::VT{} + a);
  }
  [[gnu::always_inline]] constexpr auto operator*=(T a) -> Unroll &
  requires(W != 1)
  {
    return (*this) *= (typename Unroll<R, C, W, T>::VT{} + a);
  }
  [[gnu::always_inline]] constexpr auto operator/=(T a) -> Unroll &
  requires(W != 1)
  {
    return (*this) /= (typename Unroll<R, C, W, T>::VT{} + a);
  }
};

template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator+(const Unroll<R, C, W, T> &a,
                                                const Unroll<R, C, W, T> &b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a.data[i] + b.data[i];
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator-(const Unroll<R, C, W, T> &a,
                                                const Unroll<R, C, W, T> &b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a.data[i] - b.data[i];
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator*(const Unroll<R, C, W, T> &a,
                                                const Unroll<R, C, W, T> &b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a.data[i] * b.data[i];
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator/(const Unroll<R, C, W, T> &a,
                                                const Unroll<R, C, W, T> &b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a.data[i] / b.data[i];
  return c;
}

template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator+(const Unroll<R, C, W, T> &a, typename Unroll<R, C, W, T>::VT b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a.data[i] + b;
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator-(const Unroll<R, C, W, T> &a, typename Unroll<R, C, W, T>::VT b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a.data[i] - b;
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator*(const Unroll<R, C, W, T> &a, typename Unroll<R, C, W, T>::VT b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a.data[i] * b;
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator/(const Unroll<R, C, W, T> &a, typename Unroll<R, C, W, T>::VT b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a.data[i] / b;
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator+(const Unroll<R, C, W, T> &a,
                                                T b) -> Unroll<R, C, W, T>
requires(W != 1)
{
  return a + (typename Unroll<R, C, W, T>::VT{} + b);
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator-(const Unroll<R, C, W, T> &a,
                                                T b) -> Unroll<R, C, W, T>
requires(W != 1)
{
  return a - (typename Unroll<R, C, W, T>::VT{} + b);
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator*(const Unroll<R, C, W, T> &a,
                                                T b) -> Unroll<R, C, W, T>
requires(W != 1)
{
  return a * (typename Unroll<R, C, W, T>::VT{} + b);
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator/(const Unroll<R, C, W, T> &a,
                                                T b) -> Unroll<R, C, W, T>
requires(W != 1)
{
  return a / (typename Unroll<R, C, W, T>::VT{} + b);
}

template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator+(typename Unroll<R, C, W, T>::VT a, const Unroll<R, C, W, T> &b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a + b.data[i];
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator-(typename Unroll<R, C, W, T>::VT a, const Unroll<R, C, W, T> &b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a - b.data[i];
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator*(typename Unroll<R, C, W, T>::VT a, const Unroll<R, C, W, T> &b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a * b.data[i];
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator/(typename Unroll<R, C, W, T>::VT a, const Unroll<R, C, W, T> &b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a / b.data[i];
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator+(T b,
                                                const Unroll<R, C, W, T> &a)
  -> Unroll<R, C, W, T>
requires(W != 1)
{
  return (typename Unroll<R, C, W, T>::VT{} + b) + a;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator-(T b,
                                                const Unroll<R, C, W, T> &a)
  -> Unroll<R, C, W, T>
requires(W != 1)
{
  return (typename Unroll<R, C, W, T>::VT{} + b) - a;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator*(T b,
                                                const Unroll<R, C, W, T> &a)
  -> Unroll<R, C, W, T>
requires(W != 1)
{
  return (typename Unroll<R, C, W, T>::VT{} + b) * a;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator/(T b,
                                                const Unroll<R, C, W, T> &a)
  -> Unroll<R, C, W, T>
requires(W != 1)
{
  return (typename Unroll<R, C, W, T>::VT{} + b) / a;
}

template <typename T>
constexpr auto load(const T *p, mask::None<1>) -> const T & {
  return *p;
}
template <typename T>
constexpr auto load(const T *p, mask::None<1>, int32_t) -> const T & {
  return *p;
}

template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t N, typename T, ptrdiff_t X,
          size_t NM, typename MT = mask::None<N>>
[[gnu::always_inline]] constexpr auto
loadunroll(const T *ptr, RowStride<X> rowStride, std::array<MT, NM> masks)
  -> Unroll<R, C, N, T> {
  static constexpr auto W = ptrdiff_t(std::bit_ceil(size_t(N)));
  auto rs = ptrdiff_t(rowStride);
  Unroll<R, C, N, T> ret;
  POLYMATHFULLUNROLL
  for (ptrdiff_t r = 0; r < R; ++r, ptr += rs) {
    if constexpr (NM == 0) {
      POLYMATHFULLUNROLL
      for (ptrdiff_t c = 0; c < C; ++c)
        ret[r, c] = load(ptr + c * W, mask::None<W>{});
    } else if constexpr (NM == C) {
      POLYMATHFULLUNROLL
      for (ptrdiff_t c = 0; c < C; ++c) ret[r, c] = load(ptr + c * W, masks[c]);
    } else {
      POLYMATHFULLUNROLL
      for (ptrdiff_t c = 0; c < C - 1; ++c)
        ret[r, c] = load(ptr + c * W, mask::None<W>{});
      ret[r, C - 1] = load(ptr + (C - 1) * W, masks[0]);
    }
  }
  return ret;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t N, typename T, ptrdiff_t X,
          size_t NM, typename MT = mask::None<N>>
[[gnu::always_inline]] constexpr auto
loadstrideunroll(const T *ptr, RowStride<X> rowStride, std::array<MT, NM> masks)
  -> Unroll<R, C, N, T> {
  static constexpr auto W = ptrdiff_t(std::bit_ceil(size_t(N)));
  Unroll<R, C, N, T> ret;
  auto s = int32_t(ptrdiff_t(rowStride));
  POLYMATHFULLUNROLL
  for (ptrdiff_t r = 0; r < R; ++r, ++ptr) {
    if constexpr (NM == 0) {
      POLYMATHFULLUNROLL
      for (ptrdiff_t c = 0; c < C; ++c)
        ret[r, c] = load(ptr + c * W * s, mask::None<W>{}, s);
    } else if constexpr (NM == C) {
      POLYMATHFULLUNROLL
      for (ptrdiff_t c = 0; c < C; ++c)
        ret[r, c] = load(ptr + c * W * s, masks[c], s);
    } else {
      POLYMATHFULLUNROLL
      for (ptrdiff_t c = 0; c < C - 1; ++c)
        ret[r, c] = load(ptr + c * W * s, mask::None<W>{}, s);
      ret[r, C - 1] = load(ptr + (C - 1) * W * s, masks[0], s);
    }
  }
  return ret;
}

// Represents a reference for a SIMD load, in particular so that we can store.
// Needs to support masking
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t N, typename T, ptrdiff_t X,
          ptrdiff_t NM, typename MT = mask::None<N>, bool Transposed = false>
struct UnrollRef {
  static constexpr ptrdiff_t W = ptrdiff_t(std::bit_ceil(size_t(N)));
  static_assert(N == W || C == 1,
                "If N != the next power of `2`, then `C` should be `1`");
  static_assert(
    NM == 0 || NM == 1 || NM == C,
    "Should have no masks, one mask for last `C`, or one mask per `C`");
  T *ptr;
  [[no_unique_address]] RowStride<X> rowStride;
  [[no_unique_address]] std::array<MT, NM> masks;
  constexpr operator Unroll<R, C, N, T>() {
    if constexpr (!Transposed)
      return loadunroll<R, C, N, T, X, NM, MT>(ptr, rowStride, masks);
    else return loadstrideunroll<R, C, N, T, X, NM, MT>(ptr, rowStride, masks);
  }
  constexpr auto operator=(Unroll<R, C, N, T> x) -> UnrollRef &
  requires(!Transposed)
  {
    auto rs = ptrdiff_t(rowStride);
    T *p = ptr;
    POLYMATHFULLUNROLL
    for (ptrdiff_t r = 0; r < R; ++r, p += rs) {
      if constexpr (NM == 0) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C; ++c)
          store(p + c * W, mask::None<W>{}, x[r, c]);
      } else if constexpr (NM == C) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C; ++c) store(p + c * W, masks[c], x[r, c]);
      } else { // NM == 1
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C - 1; ++c)
          store(p + c * W, mask::None<W>{}, x[r, c]);
        store(p + (C - 1) * W, masks[0], x[r, C - 1]);
      }
    }
    return *this;
  }
  constexpr auto operator=(Vec<W, T> v) -> UnrollRef &
  requires(!Transposed)
  {
    auto rs = ptrdiff_t(rowStride);
    T *p = ptr;
    POLYMATHFULLUNROLL
    for (ptrdiff_t r = 0; r < R; ++r, p += rs) {
      if constexpr (NM == 0) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C; ++c) store(p + c * W, mask::None<W>{}, v);
      } else if constexpr (NM == C) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C; ++c) store(p + c * W, masks[c], v);
      } else { // NM == 1
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C - 1; ++c)
          store(p + c * W, mask::None<W>{}, v);
        store(p + (C - 1) * W, masks[0], v);
      }
    }
    return *this;
  }
  constexpr auto operator=(Unroll<R, C, N, T> x) -> UnrollRef &
  requires(Transposed)
  {
    auto s = int32_t(ptrdiff_t(rowStride));
    T *p = ptr;
    for (ptrdiff_t r = 0; r < R; ++r, ++p) {
      if constexpr (NM == 0) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C; ++c)
          store(p + c * W * s, mask::None<W>{}, x[0, c], s);
      } else if constexpr (NM == C) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C; ++c)
          store(p + c * W * s, masks[c], x[0, c], s);
      } else {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C - 1; ++c)
          store(p + c * W * s, mask::None<W>{}, x[0, c], s);
        store(p + (C - 1) * W * s, masks[0], x[0, C - 1], s);
      }
    }
    return *this;
  }
  constexpr auto operator=(Vec<W, T> v) -> UnrollRef &
  requires(Transposed)
  {
    auto s = int32_t(ptrdiff_t(rowStride));
    T *p = ptr;
    for (ptrdiff_t r = 0; r < R; ++r, ++p) {
      if constexpr (NM == 0) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C; ++c)
          store(p + c * W * s, mask::None<W>{}, v, s);
      } else if constexpr (NM == C) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C; ++c) store(p + c * W * s, masks[c], v, s);
      } else {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C - 1; ++c)
          store(p + c * W * s, mask::None<W>{}, v, s);
        store(p + (C - 1) * W * s, masks[0], v, s);
      }
    }
    return *this;
  }
  constexpr auto operator=(std::convertible_to<T> auto x) -> UnrollRef & {
    *this = Vec<W, T>{} + T(x);
    return *this;
  }
  constexpr auto operator+=(const auto &x) -> UnrollRef & {
    return (*this) = Unroll<R, C, N, T>(*this) + x;
  }
  constexpr auto operator-=(const auto &x) -> UnrollRef & {
    return (*this) = Unroll<R, C, N, T>(*this) - x;
  }
  constexpr auto operator*=(const auto &x) -> UnrollRef & {
    return (*this) = Unroll<R, C, N, T>(*this) * x;
  }
  constexpr auto operator/=(const auto &x) -> UnrollRef & {
    return (*this) = Unroll<R, C, N, T>(*this) / x;
  }
};

// 4 x 4C -> 4C x 4
template <ptrdiff_t C, typename T>
constexpr auto transpose(Unroll<2, C, 2, T> u) -> Unroll<2 * C, 1, 2, T> {
  Unroll<2 * C, 1, 2, T> z;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < C; ++i) {
    Vec<2, T> a{u[0, i]}, b{u[1, i]};
    z[0, i] = __builtin_shufflevector(a, b, 0, 2);
    z[1, i] = __builtin_shufflevector(a, b, 1, 3);
  }
  return z;
}
template <ptrdiff_t C, typename T>
constexpr auto transpose(Unroll<4, C, 4, T> u) -> Unroll<4 * C, 1, 4, T> {
  Unroll<4 * C, 1, 4, T> z;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < C; ++i) {
    Vec<4, T> a{u[0, i]}, b{u[1, i]}, c{u[2, i]}, d{u[3, i]};
    Vec<4, T> e{__builtin_shufflevector(a, b, 0, 1, 4, 5)};
    Vec<4, T> f{__builtin_shufflevector(a, b, 2, 3, 6, 7)};
    Vec<4, T> g{__builtin_shufflevector(c, d, 0, 1, 4, 5)};
    Vec<4, T> h{__builtin_shufflevector(c, d, 2, 3, 6, 7)};
    z[0, i] = __builtin_shufflevector(e, g, 0, 2, 4, 6);
    z[1, i] = __builtin_shufflevector(e, g, 1, 3, 5, 7);
    z[2, i] = __builtin_shufflevector(f, h, 0, 2, 4, 6);
    z[3, i] = __builtin_shufflevector(f, h, 1, 3, 5, 7);
  }
  return z;
}
template <ptrdiff_t C, typename T>
constexpr auto transpose(Unroll<8, C, 8, T> u) -> Unroll<8 * C, 1, 8, T> {
  Unroll<8 * C, 1, 8, T> z;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < C; ++i) {
    Vec<4, T> a{u[0, i]}, b{u[1, i]}, c{u[2, i]}, d{u[3, i]}, e{u[4, i]},
      f{u[5, i]}, g{u[6, i]}, h{u[7, i]};
    Vec<4, T> j{__builtin_shufflevector(a, b, 0, 8, 2, 10, 4, 12, 6, 14)},
      k{__builtin_shufflevector(a, b, 1, 9, 3, 11, 5, 13, 7, 15)},
      l{__builtin_shufflevector(c, d, 0, 8, 2, 10, 4, 12, 6, 14)},
      m{__builtin_shufflevector(c, d, 1, 9, 3, 11, 5, 13, 7, 15)},
      n{__builtin_shufflevector(e, f, 0, 8, 2, 10, 4, 12, 6, 14)},
      o{__builtin_shufflevector(e, f, 1, 9, 3, 11, 5, 13, 7, 15)},
      p{__builtin_shufflevector(g, h, 0, 8, 2, 10, 4, 12, 6, 14)},
      q{__builtin_shufflevector(g, h, 1, 9, 3, 11, 5, 13, 7, 15)};
    a = __builtin_shufflevector(j, l, 0, 1, 8, 9, 4, 5, 12, 13);
    b = __builtin_shufflevector(j, l, 2, 3, 10, 11, 6, 7, 14, 15);
    c = __builtin_shufflevector(k, m, 0, 1, 8, 9, 4, 5, 12, 13);
    d = __builtin_shufflevector(k, m, 2, 3, 10, 11, 6, 7, 14, 15);
    e = __builtin_shufflevector(n, p, 0, 1, 8, 9, 4, 5, 12, 13);
    f = __builtin_shufflevector(n, p, 2, 3, 10, 11, 6, 7, 14, 15);
    g = __builtin_shufflevector(o, q, 0, 1, 8, 9, 4, 5, 12, 13);
    h = __builtin_shufflevector(o, q, 2, 3, 10, 11, 6, 7, 14, 15);
    z[0, i] = __builtin_shufflevector(a, e, 0, 1, 2, 3, 8, 9, 10, 11);
    z[1, i] = __builtin_shufflevector(a, e, 4, 5, 6, 7, 12, 13, 14, 15);
    z[2, i] = __builtin_shufflevector(b, f, 0, 1, 2, 3, 8, 9, 10, 11);
    z[3, i] = __builtin_shufflevector(b, f, 4, 5, 6, 7, 12, 13, 14, 15);
    z[4, i] = __builtin_shufflevector(c, g, 0, 1, 2, 3, 8, 9, 10, 11);
    z[5, i] = __builtin_shufflevector(c, g, 4, 5, 6, 7, 12, 13, 14, 15);
    z[6, i] = __builtin_shufflevector(d, h, 0, 1, 2, 3, 8, 9, 10, 11);
    z[7, i] = __builtin_shufflevector(d, h, 4, 5, 6, 7, 12, 13, 14, 15);
  }
  return z;
}

// returns { vector_size, num_vectors, remainder }
template <ptrdiff_t L, typename T>
consteval auto VectorDivRem() -> std::array<ptrdiff_t, 3> {
  constexpr ptrdiff_t W = Width<T>;
  if constexpr (L <= W) {
    constexpr auto V = ptrdiff_t(std::bit_ceil(size_t(L)));
    return {V, L / V, L % V};
  } else return {W, L / W, L % W};
};

template <typename T, ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename M,
          bool Transposed, ptrdiff_t X>
[[gnu::always_inline]] constexpr auto
ref(const T *p, index::UnrollDims<R, C, W, M, Transposed, X> i)
  -> Unroll<R, C, W, T> {
  if constexpr (Transposed)
    return loadstrideunroll<R, C, W>(p, i.rs, std::array<M, 1>{i.mask});
  else return loadunroll<R, C, W>(p, i.rs, std::array<M, 1>{i.mask});
}
template <typename T, ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename M,
          bool Transposed, ptrdiff_t X>
[[gnu::always_inline]] constexpr auto
ref(T *p, index::UnrollDims<R, C, W, M, Transposed, X> i)
  -> UnrollRef<R, C, W, T, X, 1, M, Transposed> {
  return {p, i.rs, std::array<M, 1>{i.mask}};
}

} // namespace poly::math::simd
#endif // SIMD_hpp_INCLUDED
