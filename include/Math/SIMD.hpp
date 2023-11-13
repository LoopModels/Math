#pragma once
#ifndef SIMD_hpp_INCLUDED
#define SIMD_hpp_INCLUDED

#include <Utilities/Invariant.hpp>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace poly::math::simd {

template <ptrdiff_t W, typename T>
using Vec [[gnu::vector_size(W * sizeof(T))]] = T;

template <typename T> static constexpr bool SupportsSIMD = false;
template <> static constexpr bool SupportsSIMD<double> = true;
template <> static constexpr bool SupportsSIMD<int64_t> = true;

struct NoMask {};
// Alternatives we can have: BitMask and VectorMask
template <ptrdiff_t W, typename U = NoMask> struct VectorIndex {
  ptrdiff_t i;
  [[no_unique_address]] U mask{};
  explicit constexpr operator ptrdiff_t() const { return ptrdiff_t(i); }
};
static_assert(sizeof(VectorIndex<2>) == sizeof(ptrdiff_t));

// template <ptrdiff_t W, typename T> auto zero()->Vec<W,T>{return Vec<W,T>{};}

#ifdef __x86_64__
#include <immintrin.h>
#ifdef __AVX512F__
static constexpr ptrdiff_t REGISTERS = 32;
static constexpr ptrdiff_t VECTORWIDTH = 64;

struct BitMask {
  uint64_t m;
  template <std::unsigned_integral U> explicit constexpr operator U() {
    return U(m);
  }
};

// In: VectorIndex, where `i.i` is the total length of the loop
// Out: mask for the final iteration. Zero indicates no masked iter.
template <ptrdiff_t W> constexpr auto mask(VectorIndex<W> i) -> BitMask {
  static_assert(std::popcount(size_t(W)) == 1);
  invariant(i.i >= 0);
  return {_bzhi_u64(0xffffffffffffffff, uint64_t(i.i) & uint64_t(W - 1))};
};
// In: VectorIndex where `i.i` is for the current iteration, and total loop
// length. Out: mask for the current iteration, 0 indicates exit loop.
template <ptrdiff_t W>
constexpr auto mask(VectorIndex<W> i, ptrdiff_t len) -> BitMask {
  static_assert(std::popcount(size_t(W)) == 1);
  ptrdiff_t srem = len - i.i;
  invariant(srem >= 0);
  return {_bzhi_u64(0xffffffffffffffff, uint64_t(srem))};
};
template <ptrdiff_t W>
constexpr auto exitLoop(VectorIndex<W, BitMask> i) -> bool {
  return i.mask.m == 0;
}

[[gnu::always_inline, gnu::artificial]] inline auto load(const double *p,
                                                         VectorIndex<8> i)
  -> Vec<8, double> {
  return std::bit_cast<Vec<8, double>>(_mm512_loadu_pd(p + i.i));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<8, BitMask> i) -> Vec<8, double> {
  return std::bit_cast<Vec<8, double>>(
    _mm512_maskz_loadu_pd(uint8_t(i.mask), p + i.i));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(const double *p, VectorIndex<8> i, Vec<8, double> x) {
  _mm512_storeu_pd(p + i.i, std::bit_cast<__m512d>(x));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, VectorIndex<8, BitMask> i, Vec<8, double> x) {
  _mm512_mask_storeu_pd(p + i.i, uint8_t(i.mask), std::bit_cast<__m512d>(x));
}

[[gnu::always_inline, gnu::artificial]] inline auto load(const int64_t *p,
                                                         VectorIndex<8> i) {
  return std::bit_cast<Vec<8, int64_t>>(_mm512_loadu_epi64(p + i.i));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<8, BitMask> i) -> Vec<8, int64_t> {
  return std::bit_cast<Vec<8, int64_t>>(
    _mm512_maskz_loadu_epi64(uint8_t(i.mask), p + i.i));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<8> i, Vec<8, int64_t> x)
  ->Vec<8, int64_t> {
  _mm512_storeu_epi64(p + i.i, std::bit_cast<__m512i>(x));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<8, BitMask> i, Vec<8, int64_t> x) {
  _mm512_mask_storeu_epi64(p + i.i, uint8_t(i.mask), std::bit_cast<__m512i>(x));
}
// strided memory accesses
template <typename U>
inline auto vindex(VectorIndex<8, U> i, int32_t stride) -> __m256i {
  return std::bit_cast<__m256i>(
    Vec<8, int32_t>{0, 1, 2, 3, 4, 5, 6, 7} * stride + int32_t(i.i));
}
// TODO: make `consteval` when clang supports it
template <ptrdiff_t W, typename T> inline auto mmzero() {
  // Extend if/when supporting more types
  static_assert(std::popcount(size_t(W)) == 1 && W <= 8);
  Vec<W, T> z{};
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
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<8> i, int32_t stride) -> Vec<8, double> {
  return std::bit_cast<Vec<8, double>>(
    _mm512_i32gather_pd(vindex(i, stride), p, 8));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<8, BitMask> i, int32_t stride)
  -> Vec<8, double> {
  return std::bit_cast<Vec<8, double>>(_mm512_mask_i32gather_pd(
    mmzero<8, double>(), uint8_t(i.mask), vindex(i, stride), p, 8));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(const double *p, VectorIndex<8> i, Vec<8, double> x, int32_t stride) {
  _mm512_i32scatter_pd(p, vindex(i, stride), std::bit_cast<__m512d>(x), 8);
}
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, VectorIndex<8, BitMask> i, Vec<8, double> x, int32_t stride) {
  _mm512_mask_i32scatter_pd(p, uint8_t(i.mask), vindex(i, stride),
                            std::bit_cast<__m512d>(x), 8);
}

[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<8> i, int32_t stride) -> Vec<8, int64_t> {
  return std::bit_cast<Vec<8, double>>(
    _mm512_i32gather_pd(vindex(i, stride), p, 8));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<8, BitMask> i, int32_t stride)
  -> Vec<8, int64_t> {
  return std::bit_cast<Vec<8, int64_t>>(_mm512_mask_i32gather_epi64(
    mmzero<8, int64_t>(), uint8_t(i.mask), vindex(i, stride), p, 8));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<8> i, Vec<8, int64_t> x, int32_t stride) {
  _mm512_i32scatter_epi64(p, vindex(i, stride), std::bit_cast<__m512i>(x), 8);
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<8, BitMask> i, Vec<8, int64_t> x,
      int32_t stride) {
  _mm512_mask_i32scatter_epi64(p, uint8_t(i.mask), vindex(i, stride),
                               std::bit_cast<__m512i>(x), 8);
}
#else // no AVX512F
static constexpr ptrdiff_t REGISTERS = 16;
#ifdef __AVX__
static constexpr ptrdiff_t VECTORWIDTH = 32;
#else // no AVX
static constexpr ptrdiff_t VECTORWIDTH = 16;
#endif
#endif
#ifdef __AVX2__
template <typename U>
inline auto vindex(VectorIndex<4, U> i, int32_t stride) -> __m128i {
  return std::bit_cast<__m128i>(Vec<4, int32_t>{0, 1, 2, 3} * stride +
                                int32_t(i.i));
}
template <typename U>
inline auto vindex(VectorIndex<2, U> i, int32_t stride) -> __m128i {
  return std::bit_cast<__m128i>(Vec<2, int64_t>{0, 1} * int64_t(stride) +
                                int64_t(i.i));
}
// Non-masked gather/scatter are the same with AVX512VL and AVX2
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<4> i, int32_t stride) -> Vec<4, double> {
  return std::bit_cast<Vec<4, double>>(
    _mm256_i32gather_pd(vindex(i, stride), p, 8));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<2> i, int32_t stride) -> Vec<2, double> {
  return std::bit_cast<Vec<2, double>>(
    _mm_i64gather_pd(vindex(i, stride), p, 8));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<4> i, int32_t stride) -> Vec<4, int64_t> {
  return std::bit_cast<Vec<4, double>>(
    _mm256_i32gather_pd(vindex(i, stride), p, 8));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<2> i, int32_t stride) -> Vec<2, int64_t> {
  return std::bit_cast<Vec<2, double>>(
    _mm_i64gather_pd(vindex(i, stride), p, 8));
}
#endif

// Here, we handle masked loads/stores
#ifdef __AVX512VL__
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<4, BitMask> i) {
  return std::bit_cast<Vec<4, double>>(
    _mm256_maskz_loadu_pd(uint8_t(i.mask), p + i.i));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, VectorIndex<4, BitMask> i, Vec<4, double> x) {
  _mm256_mask_storeu_pd(p + i.i, uint8_t(i.mask), std::bit_cast<__m256d>(x));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<4, BitMask> i) {
  return std::bit_cast<Vec<4, int64_t>>(
    _mm256_maskz_loadu_epi64(uint8_t(i.mask), p + i.i));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<4, BitMask> i, Vec<4, int64_t> x) {
  _mm256_mask_storeu_epi64(p + i.i, uint8_t(i.mask), std::bit_cast<__m256i>(x));
}

[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<2, BitMask> i) {
  return std::bit_cast<Vec<2, double>>(
    _mm_maskz_loadu_pd(uint8_t(i.mask), p + i.i));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, VectorIndex<2, BitMask> i, Vec<2, double> x) {
  _mm_mask_storeu_pd(p + i.i, uint8_t(i.mask), std::bit_cast<__128d>(x));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<2, BitMask> i) {
  return std::bit_cast<Vec<2, int64_t>>(
    _mm_maskz_loadu_epi64(uint8_t(i.mask), p + i.i));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<2, BitMask> i, Vec<2, int64_t> x) {
  _mm_mask_storeu_epi64(p + i.i, uint8_t(i.mask), std::bit_cast<__m128i>(x));
}
// gather/scatter
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<4, BitMask> i, int32_t stride)
  -> Vec<4, double> {
  return std::bit_cast<Vec<4, double>>(_mm256_mmask_i32gather_pd(
    mmzero<4, double>(), uint8_t(i.mask), vindex(i, stride), p, 8));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(const double *p, VectorIndex<4> i, Vec<4, double> x, int32_t stride) {
  _mm256_i32scatter_pd(p, vindex(i, stride), std::bit_cast<__m256d>(x), 8);
}
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, VectorIndex<4, BitMask> i, Vec<4, double> x, int32_t stride) {
  _mm256_mask_i32scatter_pd(p, uint8_t(i.mask), vindex(i, stride),
                            std::bit_cast<__m256d>(x), 8);
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<2, BitMask> i, int32_t stride)
  -> Vec<2, double> {
  return std::bit_cast<Vec<2, double>>(_mm_mmask_i64gather_pd(
    mmzero<2, double>(), uint8_t(i.mask), vindex(i, stride), p, 8));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(const double *p, VectorIndex<2> i, Vec<2, double> x, int32_t stride) {
  _mm_i64scatter_pd(p, vindex(i, stride), std::bit_cast<__m128d>(x), 8);
}
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, VectorIndex<2, BitMask> i, Vec<2, double> x, int32_t stride) {
  _mm_mask_i64scatter_pd(p, uint8_t(i.mask), vindex(i, stride),
                         std::bit_cast<__m128d>(x), 8);
}

[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<4, BitMask> i, int32_t stride)
  -> Vec<4, int64_t> {
  return std::bit_cast<Vec<4, int64_t>>(_mm256_mmask_i32gather_epi64(
    mmzero<4, int64_t>(), uint8_t(i.mask), vindex(i, stride), p, 8));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<4> i, Vec<4, int64_t> x, int32_t stride) {
  _mm256_i32scatter_epi64(p, vindex(i, stride), std::bit_cast<__m256i>(x), 8);
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<4, BitMask> i, Vec<4, int64_t> x,
      int32_t stride) {
  _mm256_mask_i32scatter_epi64(p, uint8_t(i.mask), vindex(i, stride),
                               std::bit_cast<__m256i>(x), 8);
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<2, BitMask> i, int32_t stride)
  -> Vec<2, int64_t> {
  return std::bit_cast<Vec<2, int64_t>>(_mm_mmask_i64gather_epi64(
    mmzero<2, int64_t>(), uint8_t(i.mask), vindex(i, stride), p, 8));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<2> i, Vec<2, int64_t> x, int32_t stride) {
  _mm_i64scatter_epi64(p, vindex(i, stride), std::bit_cast<__m128i>(x), 8);
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<2, BitMask> i, Vec<2, int64_t> x,
      int32_t stride) {
  _mm_mask_i64scatter_epi64(p, uint8_t(i.mask), vindex(i, stride),
                            std::bit_cast<__m128i>(x), 8);
}
#else // No AVX512VL

// We need [gather, scatter, load, store] * [unmasked, masked]
//
constexpr auto mask(VectorIndex<2> i) {
  return Vec<2, int64_t>{0, 1} < (i.i & 1);
}
constexpr auto mask(VectorIndex<2> i, ptrdiff_t len) {
  return Vec<2, int64_t>{0, 1} + i.i < len;
}
template <ptrdiff_t W>
constexpr auto exitLoop(VectorIndex<W, Vec<W, int64_t>> i) -> bool {
  return i.mask[0] == 0;
}
#ifdef __AVX__
constexpr auto mask(VectorIndex<4> i) {
  return Vec<4, int64_t>{0, 1, 2, 3} < (i.i & 3);
}
constexpr auto mask(VectorIndex<4> i, ptrdiff_t len) {
  return Vec<4, int64_t>{0, 1, 2, 3} + i.i < len;
}

// we need 256 bit fallback scatters
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, VectorIndex<4> i, Vec<4, double> x, int32_t stride) {
  p[i.i] = x[0];
  p[i.i + stride] = x[1];
  p[i.i + 2 * stride] = x[2];
  p[i.i + 3 * stride] = x[3];
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<4> i, Vec<4, int64_t> x, int32_t stride) {
  p[i.i] = x[0];
  p[i.i + stride] = x[1];
  p[i.i + 2 * stride] = x[2];
  p[i.i + 3 * stride] = x[3];
}
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, VectorIndex<4, Vec<4, int64_t>> i, Vec<4, double> x,
      int32_t stride) {
  if (i.mask[0] != 0) p[i.i] = x[0];
  if (i.mask[1] != 0) p[i.i + stride] = x[1];
  if (i.mask[2] != 0) p[i.i + 2 * stride] = x[2];
  if (i.mask[3] != 0) p[i.i + 3 * stride] = x[3];
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<4, Vec<4, int64_t>> i, Vec<4, int64_t> x,
      int32_t stride) {
  if (i.mask[0] != 0) p[i.i] = x[0];
  if (i.mask[1] != 0) p[i.i + stride] = x[1];
  if (i.mask[2] != 0) p[i.i + 2 * stride] = x[2];
  if (i.mask[3] != 0) p[i.i + 3 * stride] = x[3];
}

#endif // AVX
// 128 bit fallback scatters
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, VectorIndex<2> i, Vec<2, double> x, int32_t stride) {
  p[i.i] = x[0];
  p[i.i + stride] = x[1];
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<2> i, Vec<2, int64_t> x, int32_t stride) {
  p[i.i] = x[0];
  p[i.i + stride] = x[1];
}
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, VectorIndex<2, Vec<2, int64_t>> i, Vec<2, double> x,
      int32_t stride) {
  if (i.mask[0] != 0) p[i.i] = x[0];
  if (i.mask[1] != 0) p[i.i + stride] = x[1];
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<2, Vec<2, int64_t>> i, Vec<2, int64_t> x,
      int32_t stride) {
  if (i.mask[0] != 0) p[i.i] = x[0];
  if (i.mask[1] != 0) p[i.i + stride] = x[1];
}

#ifdef __AVX2__
// masked gathers
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<4, Vec<4, int64_t>> i, int32_t stride)
  -> Vec<4, double> {
  return std::bit_cast<Vec<4, double>>(_m256_mmask_i32gather_pd(
    mmzero<4, double>(), std::bit_cast<__m128i>(i.mask), vindex(i, stride), p,
    8));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<2, Vec<2, int64_t>> i, int32_t stride)
  -> Vec<2, double> {
  return std::bit_cast<Vec<2, double>>(
    _mm_mask_i64gather_pd(mmzero<2, double>(), std::bit_cast<__m128i>(i.mask),
                          vindex(i, stride), p, 8));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<4, Vec<4, int64_t>> i, int32_t stride)
  -> Vec<4, int64_t> {
  return std::bit_cast<Vec<4, int64_t>>(_m256_mmask_i32gather_epi64(
    mmzero<4, int64_t>(), std::bit_cast<__m128i>(i.mask), vindex(i, stride), p,
    8));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<2, Vec<2, int64_t>> i, int32_t stride)
  -> Vec<2, int64_t> {
  return std::bit_cast<Vec<2, int64_t>>(_mm_mask_i64gather_epi64(
    mmzero<2, int64_t>(), std::bit_cast<__m128i>(i.mask), vindex(i, stride), p,
    8));
}

#else          // no AVX2
// fallback 128-bit gather
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<2> i, int32_t stride) -> Vec<2, double> {
  Vec<2, double> ret;
  ret[0] = p[i.i];
  ret[1] = p[i.i + stride];
  return ret;
}

[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<2> i, int32_t stride) -> Vec<2, int64_t> {
  Vec<2, int64_t> ret;
  ret[0] = p[i.i];
  ret[1] = p[i.i + stride];
  return ret;
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<2, Vec<2, int64_t>> i, int32_t stride)
  -> Vec<2, double> {
  Vec<2, double> ret;
  ret[0] = (i.mask[0] != 0) ? p[i.i] : 0;
  ret[1] = (i.mask[1] != 0) ? p[i.i + stride] : 0;
  return ret;
}

[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<2, Vec<2, int64_t>> i, int32_t stride)
  -> Vec<2, int64_t> {
  Vec<2, int64_t> ret;
  ret[0] = (i.mask[0] != 0) ? p[i.i] : 0;
  ret[1] = (i.mask[1] != 0) ? p[i.i + stride] : 0;
  return ret;
}
#ifdef __AVX__ // no AVX2, but AVX
// fallback 256-bit gather
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<4> i, int32_t stride) -> Vec<4, double> {
  Vec<4, double> ret;
  ret[0] = p[i.i];
  ret[1] = p[i.i + stride];
  ret[2] = p[i.i + 2 * stride];
  ret[3] = p[i.i + 3 * stride];
  return ret;
}

[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<4> i, int32_t stride) -> Vec<4, int64_t> {
  Vec<4, int64_t> ret;
  ret[0] = p[i.i];
  ret[1] = p[i.i + stride];
  ret[2] = p[i.i + 2 * stride];
  ret[3] = p[i.i + 3 * stride];
  return ret;
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<4, Vec<24 int64_t>> i, int32_t stride)
  -> Vec<4, double> {
  Vec<4, double> ret;
  ret[0] = (i.mask[0] != 0) ? p[i.i] : 0;
  ret[1] = (i.mask[1] != 0) ? p[i.i + stride] : 0;
  ret[2] = (i.mask[2] != 0) ? p[i.i + 2 * stride] : 0;
  ret[3] = (i.mask[3] != 0) ? p[i.i + 3 * stride] : 0;
  return ret;
}

[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<4, Vec<4, int64_t>> i, int32_t stride)
  -> Vec<4, double> {
  Vec<4, int64_t> ret;
  ret[0] = (i.mask[0] != 0) ? p[i.i] : 0;
  ret[1] = (i.mask[1] != 0) ? p[i.i + stride] : 0;
  ret[2] = (i.mask[2] != 0) ? p[i.i + 2 * stride] : 0;
  ret[3] = (i.mask[3] != 0) ? p[i.i + 3 * stride] : 0;
  return ret;
}
#endif         // AVX
#endif         // no AVX2
#ifdef __AVX__
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<4, Vec<4, int64_t>> i) -> Vec<4, double> {
  return std::bit_cast<Vec<4, double>>(
    _mm256_maskload_pd(p + i.i, std::bit_cast<__m256i>(i.mask)));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, VectorIndex<4, uint8_t> i, Vec<4, double> x) {
  _mm256_maskstore_pd(p + i.i, std::bit_cast<__m256i>(i.mask),
                      std::bit_cast<__m256d>(x));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<4, uint8_t> i) -> Vec<4, int64_t> {
  return std::bit_cast<Vec<4, int64_t>>(
    _mm256_maskload_epi64(p + i.i, std::bit_cast<__m256i>(i.mask)));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<4, uint8_t> i, Vec<4, int64_t> x) {
  _mm256_maskstore_epi64(p + i.i, std::bit_cast<__m256i>(i.mask),
                         std::bit_cast<__m256i>(x));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<2, Vec<2, int64_t>> i) -> Vec<2, double> {
  return std::bit_cast<Vec<2, double>>(
    _mm_maskload_pd(p + i.i, std::bit_cast<__m128i>(i.mask)));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, VectorIndex<2, uint8_t> i, Vec<2, double> x) {
  _mm_maskstore_pd(p + i.i, std::bit_cast<__m128i>(i.mask),
                   std::bit_cast<__m128d>(x));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<2, uint8_t> i) -> Vec<2, int64_t> {
  return std::bit_cast<Vec<2, int64_t>>(
    _mm_maskload_epi64(p + i.i, std::bit_cast<__m128i>(i.mask)));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<2, uint8_t> i, Vec<2, int64_t> x) {
  _mm_maskstore_epi64(p + i.i, std::bit_cast<__m128i>(i.mask),
                      std::bit_cast<__m128i>(x));
}
#else  // No AVX
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<2, Vec<2, int64_t>> i) -> Vec<2, double> {
  Vec<2, double> ret;
  ret[0] = (i.mask[0] != 0) ? p[i.i] : 0;
  ret[1] = (i.mask[1] != 0) ? p[i.i + 1] : 0;
  return ret;
}

[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, VectorIndex<2, Vec<2, int64_t>> i, Vec<2, double> x) {
  if (i.mask[0] != 0) p[i.i] = x[0];
  if (i.mask[1] != 0) p[i.i + 1] = x[1];
}

[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<2, Vec<2, int64_t>> i) -> Vec<2, int64_t> {
  Vec<2, int64_t> ret;
  ret[0] = (i.mask[0] != 0) ? p[i.i] : 0;
  ret[1] = (i.mask[1] != 0) ? p[i.i + 1] : 0;
  return ret;
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<2, Vec<2, int64_t>> i, Vec<2, int64_t> x) {
  if (i.mask[0] != 0) p[i.i] = x[0];
  if (i.mask[1] != 0) p[i.i + 1] = x[1];
}
#endif // No AVX
#endif // No AVX512VL
#ifdef __AVX__
[[gnu::always_inline, gnu::artificial]] inline auto load(const double *p,
                                                         VectorIndex<4> i) {
  return std::bit_cast<Vec<4, double>>(_mm256_loadu_pd(p + i.i));
}

[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, VectorIndex<4> i, Vec<4, double> x) {
  _mm256_storeu_pd(p + i.i, std::bit_cast<__m256d>(x));
}

[[gnu::always_inline, gnu::artificial]] inline auto load(const int64_t *p,
                                                         VectorIndex<4> i) {
  return std::bit_cast<Vec<4, int64_t>>(_mm256_loadu_epi64(p + i.i));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<4> i, Vec<4, int64_t> x) {
  _mm256_storeu_epi64(p + i.i, std::bit_cast<__m256i>(x));
}
#endif // AVX

[[gnu::always_inline, gnu::artificial]] inline auto load(const double *p,
                                                         VectorIndex<2> i)
  -> Vec<2, double> {
  return std::bit_cast<Vec<2, double>>(_mm_loadu_pd(p + i.i));
}

[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, VectorIndex<2> i, Vec<2, double> x) {
  _mm_storeu_pd(p + i.i, std::bit_cast<__m128d>(x));
}

[[gnu::always_inline, gnu::artificial]] inline auto load(const int64_t *p,
                                                         VectorIndex<2> i) {
  return std::bit_cast<Vec<2, int64_t>>(_mm_loadu_epi64(p + i.i));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<2> i, Vec<2, int64_t> x) {
  _mm_storeu_epi64(p, std::bit_cast<__m128i>(x + i.i));
}

#else // not __x86_64__
static constexpr ptrdiff_t REGISTERS = 32;
static constexpr ptrdiff_t VECTORWIDTH = 16;
constexpr auto mask(VectorIndex<2> i) {
  return Vec<2, int64_t>{0, 1} < (i.i & 1);
}
constexpr auto mask(VectorIndex<2> i, ptrdiff_t len) {
  return Vec<2, int64_t>{0, 1} + i.i < len;
}
template <ptrdiff_t W>
constexpr auto exitLoop(VectorIndex<W, Vec<W, int64_t>> i) -> bool {
  return i.mask[0] == 0;
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto load(const T *p,
                                                         VectorIndex<W> i)
  -> Vec<W, T> {
  Vec<W, T> ret;
#pragma unroll
  for (ptrdiff_t w = 0; w < W; ++w) ret[w] = p[i.i + w];
  return ret;
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, VectorIndex<W> i, Vec<W, T> x) {
#pragma unroll
  for (ptrdiff_t w = 0; w < W; ++w) p[i.i + w] = x[w];
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, VectorIndex<W, Vec<W, int64_t>> i) -> Vec<W, T> {
  Vec<W, T> ret;
#pragma unroll
  for (ptrdiff_t w = 0; w < W; ++w) ret[w] = (i.mask[w] != 0) ? p[i.i + w] : 0;
  return ret;
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, VectorIndex<W, Vec<W, int64_t>> i, Vec<W, T> x) {
#pragma unroll
  for (ptrdiff_t w = 0; w < W; ++w)
    if (i.mask[w] != 0) p[i.i + w] = x[w];
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, VectorIndex<W> i, int32_t stride) -> Vec<W, T> {
  Vec<W, T> ret;
#pragma unroll
  for (ptrdiff_t w = 0; w < W; ++w) ret[w] = p[i.i + w * stride];
  return ret;
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, VectorIndex<W> i, Vec<W, T> x, int32_t stride) {
#pragma unroll
  for (ptrdiff_t w = 0; w < W; ++w) p[i.i + w * stride] = x[w];
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, VectorIndex<W, Vec<W, int64_t>> i, int32_t stride)
  -> Vec<W, T> {
  Vec<W, T> ret;
#pragma unroll
  for (ptrdiff_t w = 0; w < W; ++w)
    ret[w] = (i.mask[w] != 0) ? p[i.i + w * stride] : 0;
  return ret;
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, VectorIndex<W, Vec<W, int64_t>> i, Vec<W, T> x, int32_t stride) {
#pragma unroll
  for (ptrdiff_t w = 0; w < W; ++w)
    if (i.mask[w] != 0) p[i.i + w * stride] = x[w];
}
#endif

template <typename T>
static constexpr ptrdiff_t Width = VECTORWIDTH / sizeof(T);

// note, when `I` isa `VectorIndex<W,uint8_t>`
// then the mask only gets applied to the last unroll!
template <ptrdiff_t U, typename I> struct Unroll {
  I index;
  explicit constexpr operator ptrdiff_t() const { return ptrdiff_t(index); }
};

// returns { vector_size, num_vectors, remainder }
template <ptrdiff_t L, typename T>
consteval auto VectorDivRem() -> std::array<ptrdiff_t, 3> {
  constexpr ptrdiff_t W = Width<T>;
  if constexpr (L <= W) {
    constexpr auto V = ptrdiff_t(std::bit_ceil(size_t(L)));
    return {V, L / V, L % V};
  } else return {W, L / W, L % W};
};

} // namespace poly::math::simd

#endif // SIMD_hpp_INCLUDED
