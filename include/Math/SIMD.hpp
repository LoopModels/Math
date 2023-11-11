#pragma once
#ifndef SIMD_hpp_INCLUDED
#define SIMD_hpp_INCLUDED

#include <cstddef>
#include <cstdint>

template <ptrdiff_t W, typename T>
using Vec __attribute__((vector_size(W * sizeof(T)))) = T;

template <typename T> static constexpr bool SupportsSIMD = false;
template <> static constexpr bool SupportsSIMD<double> = true;
template <> static constexpr bool SupportsSIMD<int64_t> = true;

struct NoMask {};

template <ptrdiff_t W, typename U = NoMask> struct VectorIndex {
  ptrdiff_t i;
  [[no_unique_address]] U mask{};
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
[[gnu::always_inline, gnu::artificial]] inline auto load(const double *p,
                                                         VectorIndex<2> i)
  -> Vec<2, double> {
  Vec<2, double> ret;
  ret[0] = p[i.i];
  ret[1] = p[i.i + 1];
  return ret;
}

[[gnu::always_inline, gnu::artificial]] inline void
store(double *p, VectorIndex<2> i, Vec<2, double> x) {
  p[i.i] = x[0];
  p[i.i + 1] = x[1];
}

[[gnu::always_inline, gnu::artificial]] inline auto load(const int64_t *p,
                                                         VectorIndex<2> i) {
  Vec<2, int64_t> ret;
  ret[0] = p[i.i];
  ret[1] = p[i.i + 1];
  return ret;
}
[[gnu::always_inline, gnu::artificial]] inline void
store(int64_t *p, VectorIndex<2> i, Vec<2, int64_t> x) {
  p[i.i] = x[0];
  p[i.i + 1] = x[1];
}
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
#endif

#endif // SIMD_hpp_INCLUDED
