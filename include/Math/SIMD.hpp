#pragma once
#ifndef SIMD_hpp_INCLUDED
#define SIMD_hpp_INCLUDED

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

namespace poly::math::simd {

template <ptrdiff_t W, typename T>
using Vec [[gnu::vector_size(W * sizeof(T))]] = T;

template <typename T> static constexpr bool SupportsSIMD = false;
template <> static constexpr bool SupportsSIMD<double> = true;
template <> static constexpr bool SupportsSIMD<int64_t> = true;

struct NoMask {};

template <ptrdiff_t W, typename U = NoMask> struct VectorIndex {
  ptrdiff_t i;
  [[no_unique_address]] U mask{};
  explicit constexpr operator ptrdiff_t() const { return ptrdiff_t(index); }
};
static_assert(sizeof(VectorIndex<2>) == sizeof(ptrdiff_t));

#ifdef __x86_64__
#include <immintrin.h>
#ifdef __AVX512F__
static constexpr ptrdiff_t REGISTERS = 32;
static constexpr ptrdiff_t VECTORWIDTH = 64;
[[gnu::always_inline, gnu::artificial]] inline auto load(const double *p,
                                                         VectorIndex<8> i) {
  return std::bit_cast<Vec<8, double>>(_mm512_loadu_pd(p + i.i));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<8, uint8_t> i) {
  return std::bit_cast<Vec<8, double>>(_mm512_maskz_loadu_pd(i.mask, p + i.i));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(const double *p, VectorIndex<8> i, Vec<8, double> x) {
  _mm512_storeu_pd(p + i.i, std::bit_cast<__m512d>(x));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, VectorIndex<8, uint8_t> i, Vec<8, double> x) {
  _mm512_mask_storeu_pd(p + i.i, i.mask, std::bit_cast<__m512d>(x));
}

[[gnu::always_inline, gnu::artificial]] inline auto load(const int64_t *p,
                                                         VectorIndex<8> i) {
  return std::bit_cast<Vec<8, int64_t>>(_mm512_loadu_epi64(p + i.i));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<8, uint8_t> i) {
  return std::bit_cast<Vec<8, int64_t>>(
    _mm512_maskz_loadu_epi64(i.mask, p + i.i));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<8> i, Vec<8, int64_t> x) {
  _mm512_storeu_epi64(p + i.i, std::bit_cast<__m512i>(x));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<8, uint8_t> i, Vec<8, int64_t> x) {
  _mm512_mask_storeu_epi64(p + i.i, i.mask, std::bit_cast<__m512i>(x));
}
#else
static constexpr ptrdiff_t REGISTERS = 16;
#ifdef __AVX__
static constexpr ptrdiff_t VECTORWIDTH = 32;
#else
static constexpr ptrdiff_t VECTORWIDTH = 16;
#endif
#endif
// Here, we handle masked loads/stores
#ifdef __AVX512VF__
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<4, uint8_t> i) {
  return std::bit_cast<Vec<4, double>>(_mm256_maskz_loadu_pd(i.mask, p + i.i));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, VectorIndex<4, uint8_t> i, Vec<4, double> x) {
  _mm256_mask_storeu_pd(p + i.i, i.mask, std::bit_cast<__m256d>(x));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<4, uint8_t> i) {
  return std::bit_cast<Vec<4, int64_t>>(
    _mm256_maskz_loadu_epi64(i.mask, p + i.i));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<4, uint8_t> i, Vec<4, int64_t> x) {
  _mm256_mask_storeu_epi64(p + i.i, i.mask, std::bit_cast<__m256i>(x));
}

[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<2, uint8_t> i) {
  return std::bit_cast<Vec<2, double>>(_mm_maskz_loadu_pd(i.mask, p + i.i));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, VectorIndex<2, uint8_t> i, Vec<2, double> x) {
  _mm_mask_storeu_pd(p + i.i, i.mask, std::bit_cast<__128d>(x));
}
[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<2, uint8_t> i) {
  return std::bit_cast<Vec<2, int64_t>>(_mm_maskz_loadu_epi64(i.mask, p + i.i));
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<2, uint8_t> i, Vec<2, int64_t> x) {
  _mm_mask_storeu_epi64(p + i.i, i.mask, std::bit_cast<__m128i>(x));
}
#else // No AVX512VL
#ifdef __AVX__
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<4, uint8_t> i) -> Vec<4, double> {
  Vec<4, double> ret;
  ret[0] = (i.mask & 1) ? p[i.i] : 0;
  ret[1] = (i.mask & 2) ? p[i.i + 1] : 0;
  ret[2] = (i.mask & 4) ? p[i.i + 2] : 0;
  ret[3] = (i.mask & 8) ? p[i.i + 3] : 0;
  return ret;
}

[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, VectorIndex<2, uint8_t> i, Vec<2, double> x) {
  if (i.mask & 1) p[i.i] = x[0];
  if (i.mask & 2) p[i.i + 1] = x[1];
  if (i.mask & 4) p[i.i + 2] = x[2];
  if (i.mask & 8) p[i.i + 3] = x[3];
}

[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<2, uint8_t> i) {
  Vec<2, int64_t> ret;
  ret[0] = (i.mask & 1) ? p[i.i] : 0;
  ret[1] = (i.mask & 2) ? p[i.i + 1] : 0;
  ret[2] = (i.mask & 4) ? p[i.i + 2] : 0;
  ret[3] = (i.mask & 8) ? p[i.i + 3] : 0;
  return ret;
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<2, uint8_t> i, Vec<2, int64_t> x) {
  if (i.mask & 1) p[i.i] = x[0];
  if (i.mask & 2) p[i.i + 1] = x[1];
  if (i.mask & 4) p[i.i + 2] = x[2];
  if (i.mask & 8) p[i.i + 3] = x[3];
}
#endif
#endif
[[gnu::always_inline, gnu::artificial]] inline auto
load(const double *p, VectorIndex<2, uint8_t> i) -> Vec<2, double> {
  Vec<2, double> ret;
  ret[0] = (i.mask & 1) ? p[i.i] : 0;
  ret[1] = (i.mask & 2) ? p[i.i + 1] : 0;
  return ret;
}

[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, VectorIndex<2, uint8_t> i, Vec<2, double> x) {
  if (i.mask & 1) p[i.i] = x[0];
  if (i.mask & 2) p[i.i + 1] = x[1];
}

[[gnu::always_inline, gnu::artificial]] inline auto
load(const int64_t *p, VectorIndex<2, uint8_t> i) {
  Vec<2, int64_t> ret;
  ret[0] = (i.mask & 1) ? p[i.i] : 0;
  ret[1] = (i.mask & 2) ? p[i.i + 1] : 0;
  return ret;
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<2, uint8_t> i, Vec<2, int64_t> x) {
  if (i.mask & 1) p[i.i] = x[0];
  if (i.mask & 2) p[i.i + 1] = x[1];
}
#ifdef __AVX512F__

#endif
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

#endif

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

#else
static constexpr ptrdiff_t REGISTERS = 32;
static constexpr ptrdiff_t VECTORWIDTH = 16;
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
load(const T *p, VectorIndex<W, uint8_t> i) -> Vec<W, T> {
  Vec<W, T> ret;
#pragma unroll
  for (ptrdiff_t w = 0; w < W; ++w)
    ret[w] = (i.mask & (1 << w)) ? p[i.i + w] : 0;
  return ret;
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, VectorIndex<W, uint8_t> i, Vec<W, T> x) {
#pragma unroll
  for (ptrdiff_t w = 0; w < W; ++w)
    if (i.mask & (1 << w)) p[i.i + w] = x[w];
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
