#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#include "LoopMacros.hxx"
#ifdef __x86_64__
#include <immintrin.h>
#endif

#ifndef USE_MODULE
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>

#include "Math/AxisTypes.cxx"
#include "SIMD/Masks.cxx"
#include "SIMD/Vec.cxx"
#include "Utilities/Invariant.cxx"
#else
export module SIMD:Intrin;

import AxisTypes;
import Invariant;
import STL;
import :Mask;
import :Vec;
#endif

#ifdef USE_MODULE
export namespace simd {
#else
namespace simd {
#endif

// Supported means by this library currently; more types may be added in the
// future as needed.

#ifndef POLYMATHNOEXPLICITSIMDARRAY
template <typename T>
concept SIMDSupported =
  (std::integral<T> || std::floating_point<T>) && (sizeof(T) >= 4);
#else
template <typename T>
concept SIMDSupported = false;
#endif

template <ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto vbroadcast(Vec<W, T> v) -> Vec<W, T> {
  if constexpr (W == 1) return v;
  else if constexpr (W == 2) return __builtin_shufflevector(v, v, 0, 0);
  else if constexpr (W == 4) return __builtin_shufflevector(v, v, 0, 0, 0, 0);
  else if constexpr (W == 8)
    return __builtin_shufflevector(v, v, 0, 0, 0, 0, 0, 0, 0, 0);
  else if constexpr (W == 16)
    return __builtin_shufflevector(v, v, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0);
  else if constexpr (W == 32)
    return __builtin_shufflevector(v, v, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0);
  else if constexpr (W == 64)
    return __builtin_shufflevector(
      v, v, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  else static_assert(false);
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto vbroadcast(T x) -> Vec<W, T> {
  if constexpr (W > 1) {
    if consteval {
      return Vec<W, T>{} + x;
    } else {
      return vbroadcast<W, T>(Vec<W, T>{x});
    }
  } else return x;
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto load(const T *p,
                                                         mask::None<1>) -> T {
  return *p;
}
template <typename T>
constexpr auto load(const T *p, mask::None<1>, int32_t) -> const T & {
  return *p;
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void store(T *p, mask::None<1>,
                                                          T x) {
  *p = x;
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void store(T *p, mask::None<1>,
                                                          T x, int32_t) {
  *p = x;
}
#ifdef __x86_64__

template <ptrdiff_t W, typename T> consteval auto mmzero() {
  // Extend if/when supporting more types
  static_assert(std::popcount(size_t(W)) == 1 && (W * sizeof(T) <= 64));
  if constexpr (std::same_as<T, double>) {
    if constexpr (W == 8) return __m512d{};
    else if constexpr (W == 4) return __m256d{};
    else return __m128d{};
  } else if constexpr (std::same_as<T, float>) {
    if constexpr (W == 16) return __m512{};
    else if constexpr (W == 8) return __m256{};
    else return __m128{};
  } else {
    static_assert(std::integral<T>);
    if constexpr (W * sizeof(T) == 64) return __m512i{};
    else if constexpr (W * sizeof(T) == 32) return __m256i{};
    else return __m128i{};
  }
}

#ifdef __AVX512F__

template auto vbroadcast<8, int64_t>(Vec<8, int64_t>) -> Vec<8, int64_t>;

// namespace hw {
// typedef double __m512d_u __attribute__((__vector_size__(64),
// __aligned__(8)));

// typedef int64_t __m512i_u __attribute__((__vector_size__(64),
// __aligned__(8))); } // namespace hw

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::None<16>) -> Vec<16, T> {
  const void *vp = p;
  if constexpr (std::same_as<T, float>) {
    return std::bit_cast<Vec<16, float>>(_mm512_loadu_ps(vp));
  } else if constexpr (sizeof(T) == 4) {
    return std::bit_cast<Vec<16, T>>(_mm512_loadu_epi32(vp));
#ifdef __AVX512VL__
  } else if constexpr (sizeof(T) == 2) {
    return std::bit_cast<Vec<16, T>>(_mm256_loadu_epi16(vp));
#endif
  } else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Bit<8> i) -> Vec<8, T> {
  const void *vp = p;
  if constexpr (std::same_as<T, double>) {
    return std::bit_cast<Vec<8, double>>(
      _mm512_maskz_loadu_pd(uint8_t(i.mask_), vp));
  } else if constexpr (sizeof(T) == 8) {
    return std::bit_cast<Vec<8, T>>(
      _mm512_maskz_loadu_epi64(uint8_t(i.mask_), vp));
#ifdef __AVX512VL__
  } else if constexpr (std::same_as<T, float>) {
    return std::bit_cast<Vec<8, float>>(
      _mm256_maskz_loadu_ps(uint8_t(i.mask_), vp));
  } else if constexpr (sizeof(T) == 4) {
    return std::bit_cast<Vec<8, T>>(
      _mm256_maskz_loadu_epi32(uint8_t(i.mask_), vp));
  } else if constexpr (sizeof(T) == 2) {
    return std::bit_cast<Vec<8, T>>(
      _mm_maskz_loadu_epi16(uint8_t(i.mask_), vp));
#endif
  } else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *pt, mask::Bit<16> i) -> Vec<16, T> {
  const void *p = pt;
  if constexpr (std::same_as<T, float>)
    return std::bit_cast<Vec<16, float>>(
      _mm512_maskz_loadu_ps(uint16_t(i.mask_), p));
  else if constexpr (sizeof(T) == 4)
    return std::bit_cast<Vec<16, T>>(
      _mm512_maskz_loadu_epi32(uint16_t(i.mask_), p));
#ifdef __AVX512VL__
  else if constexpr (sizeof(T) == 2)
    return std::bit_cast<Vec<16, T>>(
      _mm256_maskz_loadu_epi16(uint16_t(i.mask_), p));
#endif
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void store(T *pt, mask::None<16>,
                                                          Vec<16, T> x) {
  void *p = pt;
  if constexpr (std::same_as<T, float>) {
    __m512 xf = std::bit_cast<__m512>(x);
    _mm512_storeu_ps(p, xf);
  } else if constexpr (sizeof(T) == 4) {
    __m512i xi = std::bit_cast<__m512i>(x);
    _mm512_storeu_epi32(p, xi);
#ifdef __AVX512VL__
  } else if constexpr (sizeof(T) == 2) {
    __m256i xi = std::bit_cast<__m256i>(x);
    _mm256_storeu_epi16(p, xi);
#endif
  } else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void store(T *p, mask::Bit<8> i,
                                                          Vec<8, T> x) {
  void *vp = p;
  if constexpr (std::same_as<T, double>) {
    __m512d xd = std::bit_cast<__m512d>(x);
    _mm512_mask_storeu_pd(vp, uint8_t(i.mask_), xd);
  } else if constexpr (sizeof(T) == 8) {
    __m512i xi = std::bit_cast<__m512i>(x);
    _mm512_mask_storeu_epi64(vp, uint8_t(i.mask_), xi);
#ifdef __AVX512VL__
  } else if constexpr (std::same_as<T, float>) {
    __m256 xf = std::bit_cast<__m256>(x);
    _mm256_mask_storeu_ps(vp, uint8_t(i.mask_), xf);
  } else if constexpr (sizeof(T) == 4) {
    __m256i xi = std::bit_cast<__m256i>(x);
    _mm256_mask_storeu_epi32(vp, uint8_t(i.mask_), xi);
  } else if constexpr (sizeof(T) == 2) {
    __m128i xi = std::bit_cast<__m128i>(x);
    _mm_mask_storeu_epi16(vp, uint8_t(i.mask_), xi);
#endif
  } else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *pt, mask::Bit<16> i, Vec<16, T> x) {
  void *p = pt;
  if constexpr (std::same_as<T, float>) {
    __m512 xf = std::bit_cast<__m512>(x);
    _mm512_mask_storeu_ps(p, uint16_t(i.mask_), xf);
  } else if constexpr (sizeof(T) == 4) {
    __m512i xi = std::bit_cast<__m512i>(x);
    _mm512_mask_storeu_epi32(p, uint16_t(i.mask_), xi);
#ifdef __AVX512VL__
  } else if constexpr (sizeof(T) == 2) {
    __m256i xi = std::bit_cast<__m256i>(x);
    _mm256_mask_storeu_epi16(p, uint16_t(i.mask_), xi);
#endif
  } else static_assert(false);
}

// strided memory accesses
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *pt, mask::None<16>, Vec<16, int32_t> i) -> Vec<16, T> {
  auto inds = std::bit_cast<__m512i>(i);
  const void *p = pt;
  if constexpr (std::same_as<T, float>)
    return std::bit_cast<Vec<16, T>>(_mm512_i32gather_ps(inds, p, 4));
  else if constexpr (sizeof(T) == 4)
    return std::bit_cast<Vec<8, T>>(_mm512_i32gather_epi32(inds, p, 4));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::None<16>, int32_t stride) -> Vec<16, T> {
  return gather(p, mask::None<16>{}, range<16>() * stride);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *pt, mask::Bit<8> i, Vec<8, int32_t> indv) -> Vec<8, T> {
  auto inds = std::bit_cast<__m256i>(indv);
  const void *p = pt;
  static constexpr auto src = mmzero<8, T>();
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<8, double>>(
      _mm512_mask_i32gather_pd(src, uint8_t(i.mask_), inds, p, 8));
  else if constexpr (sizeof(T) == 8)
    return std::bit_cast<Vec<8, T>>(
      _mm512_mask_i32gather_epi64(src, uint8_t(i.mask_), inds, p, 8));
#ifdef __AVX512VL__
  else if constexpr (std::same_as<T, float>)
    return std::bit_cast<Vec<8, float>>(
      _mm256_mmask_i32gather_ps(src, uint8_t(i.mask_), inds, p, 4));
  else if constexpr (sizeof(T) == 4)
    return std::bit_cast<Vec<8, T>>(
      _mm256_mmask_i32gather_epi32(src, uint8_t(i.mask_), inds, p, 4));
#endif
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *pt, mask::Bit<8> i, Vec<8, int64_t> indv) -> Vec<8, T> {
  auto inds = std::bit_cast<__m512i>(indv);
  const void *p = pt;
  static constexpr auto src = mmzero<8, T>();
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<8, double>>(
      _mm512_mask_i64gather_pd(src, uint8_t(i.mask_), inds, p, 8));
  else if constexpr (sizeof(T) == 8)
    return std::bit_cast<Vec<8, T>>(
      _mm512_mask_i64gather_epi64(src, uint8_t(i.mask_), inds, p, 8));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Bit<8> i, int32_t stride) -> Vec<8, T> {
  return gather(p, i, range<8>() * stride);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *pt, mask::Bit<16> i, Vec<16, int32_t> indv) -> Vec<16, T> {
  auto inds = std::bit_cast<__m512i>(indv);
  const void *p = pt;
  auto src = mmzero<8, T>();
  if constexpr (std::same_as<T, float>)
    return std::bit_cast<Vec<16, float>>(
      _mm512_mask_i32gather_ps(src, uint16_t(i.mask_), inds, p, 4));
  else if constexpr (sizeof(T) == 4)
    return std::bit_cast<Vec<16, T>>(
      _mm512_mask_i32gather_epi32(src, uint16_t(i.mask_), inds, p, 4));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Bit<16> i, int32_t stride) -> Vec<16, T> {
  return gather(p, i, range<16>() * stride);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
scatter(T *pt, mask::None<8>, Vec<8, T> x, Vec<8, int32_t> indv) {
  auto inds = std::bit_cast<__m256i>(indv);
  void *p = pt;
  if constexpr (std::same_as<T, double>)
    _mm512_i32scatter_pd(p, inds, std::bit_cast<__m512d>(x), 8);
  else if constexpr (sizeof(T) == 8)
    _mm512_i32scatter_epi64(p, inds, std::bit_cast<__m512i>(x), 8);
#ifdef __AVX512VL__
  else if constexpr (std::same_as<T, float>)
    _mm256_i32scatter_ps(p, inds, std::bit_cast<__m256>(x), 4);
  else if constexpr (sizeof(T) == 4)
    _mm256_i32scatter_epi32(p, inds, std::bit_cast<__m256i>(x), 4);
#endif
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::None<8>, Vec<8, T> x, int32_t stride) {
  scatter(p, mask::None<8>{}, x, range<8>() * stride);
}

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
scatter(T *pt, mask::None<16>, Vec<16, T> x, Vec<16, int32_t> indv) {
  auto inds = std::bit_cast<__m512i>(indv);
  void *p = pt;
  if constexpr (std::same_as<T, float>)
    _mm512_i32scatter_ps(p, inds, std::bit_cast<__m512>(x), 4);
  else if constexpr (sizeof(T) == 4)
    _mm512_i32scatter_epi32(p, inds, std::bit_cast<__m512i>(x), 4);
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::None<16>, Vec<16, T> x, int32_t stride) {
  scatter(p, mask::None<16>{}, x, range<16>() * stride);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
scatter(T *pt, mask::Bit<8> i, Vec<8, T> x, Vec<8, int32_t> indv) {
  auto inds = std::bit_cast<__m256i>(indv);
  void *p = pt;
  if constexpr (std::same_as<T, double>)
    _mm512_mask_i32scatter_pd(p, uint8_t(i.mask_), inds,
                              std::bit_cast<__m512d>(x), 8);
  else if constexpr (sizeof(T) == 8)
    _mm512_mask_i32scatter_epi64(p, uint8_t(i.mask_), inds,
                                 std::bit_cast<__m512i>(x), 8);
#ifdef __AVX512VL__
  else if constexpr (std::same_as<T, float>)
    _mm256_mask_i32scatter_ps(p, uint8_t(i.mask_), inds,
                              std::bit_cast<__m256>(x), 4);
  else if constexpr (sizeof(T) == 4)
    _mm256_mask_i32scatter_epi32(p, uint8_t(i.mask_), inds,
                                 std::bit_cast<__m256i>(x), 4);
#endif
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Bit<8> i, Vec<8, T> x, int32_t stride) {
  scatter(p, i, x, range<8>() * stride);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
scatter(T *pt, mask::Bit<16> i, Vec<16, T> x, Vec<16, int32_t> indv) {
  auto inds = std::bit_cast<__m512i>(indv);
  void *p = pt;
  if constexpr (std::same_as<T, float>)
    _mm512_mask_i32scatter_ps(p, uint16_t(i.mask_), inds,
                              std::bit_cast<__m512>(x), 4);
  else if constexpr (sizeof(T) == 4)
    _mm512_mask_i32scatter_epi32(p, uint16_t(i.mask_), inds,
                                 std::bit_cast<__m512i>(x), 4);
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Bit<16> i, Vec<16, T> x, int32_t stride) {
  scatter(p, i, x, range<16>() * stride);
}
template <typename T>
constexpr auto select(mask::Bit<8> m, Vec<8, T> x, Vec<8, T> y) -> Vec<8, T> {
  if constexpr (std::same_as<T, double>) {
    __m512d xd = std::bit_cast<__m512d>(x), yd = std::bit_cast<__m512d>(y);
    return std::bit_cast<Vec<8, T>>(_mm512_mask_mov_pd(yd, uint8_t(m), xd));
  } else if constexpr (sizeof(T) == 8) {
    __m512i xi = std::bit_cast<__m512i>(x), yi = std::bit_cast<__m512i>(y);
    return std::bit_cast<Vec<8, T>>(
      _mm512_mask_mov_epi64(yi, uint8_t(m.mask_), xi));
#ifdef __AVX512VL__
  } else if constexpr (std::same_as<T, float>) {
    __m256 xf = std::bit_cast<__m256>(x), yf = std::bit_cast<__m256>(y);
    return std::bit_cast<Vec<8, T>>(_mm256_mask_mov_ps(yf, uint8_t(m), xf));
  } else if constexpr (sizeof(T) == 4) {
    __m256i xi = std::bit_cast<__m256i>(x), yi = std::bit_cast<__m256i>(y);
    return std::bit_cast<Vec<8, T>>(
      _mm256_mask_mov_epi32(yi, uint8_t(m.mask_), xi));
#endif
  } else static_assert(false);
}

template <typename T>
constexpr auto select(mask::Bit<16> m, Vec<16, T> x,
                      Vec<16, T> y) -> Vec<16, T> {
  if constexpr (std::same_as<T, float>) {
    __m512 xf = std::bit_cast<__m512>(x), yf = std::bit_cast<__m512>(y);
    return std::bit_cast<Vec<16, T>>(_mm512_mask_mov_ps(yf, uint16_t(m), xf));
  } else if constexpr (sizeof(T) == 4) {
    __m512i xi = std::bit_cast<__m512i>(x), yi = std::bit_cast<__m512i>(y);
    return std::bit_cast<Vec<16, T>>(
      _mm512_mask_mov_epi32(yi, uint16_t(m.mask_), xi));
  } else static_assert(false);
}

#endif // AVX512F
#ifdef __AVX__
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::None<8>) -> Vec<8, T> {
  if constexpr (std::same_as<T, float>) {
    return std::bit_cast<Vec<8, float>>(_mm256_loadu_ps(p));
#ifdef __AVX512F__
  } else if constexpr (sizeof(T) == 4) {
    const void *vp = p;
    return std::bit_cast<Vec<8, T>>(_mm256_loadu_epi32(vp));
  } else if constexpr (sizeof(T) == 2) {
    const void *vp = p;
    return std::bit_cast<Vec<8, T>>(_mm_loadu_epi16(vp));
  } else if constexpr (std::same_as<T, double>) {
    const void *vp = p;
    return std::bit_cast<Vec<8, double>>(_mm512_loadu_pd(vp));
  } else if constexpr (sizeof(T) == 8) {
    const void *vp = p;
    return std::bit_cast<Vec<8, T>>(_mm512_loadu_epi64(vp));
#else
  } else if constexpr (sizeof(T) == 4) {
    const __m256i *pi = reinterpret_cast<const __m256i *>(p);
    return std::bit_cast<Vec<8, T>>(_mm256_loadu_si256(pi));
  } else if constexpr (sizeof(T) == 2) {
    const __m128i *pi = reinterpret_cast<const __m128i *>(p);
    return std::bit_cast<Vec<8, T>>(_mm_loadu_si128(pi));
#endif
  } else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void store(T *p, mask::None<8>,
                                                          Vec<8, T> x) {
  if constexpr (std::same_as<T, float>) {
    __m256 xf = std::bit_cast<__m256>(x);
    _mm256_storeu_ps(p, xf);
#ifdef __AVX512F__
  } else if constexpr (sizeof(T) == 4) {
    void *vp = p;
    __m256i xi = std::bit_cast<__m256i>(x);
    _mm256_storeu_epi32(vp, xi);
  } else if constexpr (sizeof(T) == 2) {
    void *vp = p;
    __m128i xi = std::bit_cast<__m128i>(x);
    _mm_storeu_epi16(vp, xi);
  } else if constexpr (std::same_as<T, double>) {
    void *vp = p;
    __m512d xd = std::bit_cast<__m512d>(x);
    _mm512_storeu_pd(vp, xd);
  } else if constexpr (sizeof(T) == 8) {
    void *vp = p;
    __m512i xi = std::bit_cast<__m512i>(x);
    _mm512_storeu_epi64(vp, xi);
#else
  } else if constexpr (sizeof(T) == 4) {
    __m256i xi = std::bit_cast<__m256i>(x);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(p), xi);
  } else if constexpr (sizeof(T) == 2) {
    __m128i xi = std::bit_cast<__m128i>(x);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(p), xi);
#endif
  } else static_assert(false);
}
#ifdef __AVX2__
// Non-masked gather is same with AVX512VL and AVX2
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::None<4>, Vec<4, int32_t> indv) -> Vec<4, T> {
  auto x = std::bit_cast<__m128i>(indv);
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<4, double>>(_mm256_i32gather_pd(p, x, 8));
  else if constexpr (std::same_as<T, float>)
    return std::bit_cast<Vec<4, float>>(_mm_i32gather_ps(p, x, 4));
  else if constexpr (sizeof(T) == 8) {
    const long long int *pi = reinterpret_cast<const long long int *>(p);
    return std::bit_cast<Vec<4, T>>(_mm256_i32gather_epi64(pi, x, 8));
  } else if constexpr (sizeof(T) == 4)
    return std::bit_cast<Vec<4, T>>(_mm_i32gather_epi32(p, x, 4));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::None<4>, Vec<4, int64_t> indv) -> Vec<4, T> {
  auto x = std::bit_cast<__m256i>(indv);
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<4, double>>(_mm256_i64gather_pd(p, x, 8));
  else if constexpr (sizeof(T) == 8) {
    const long long int *pi = reinterpret_cast<const long long int *>(p);
    return std::bit_cast<Vec<4, T>>(_mm256_i64gather_epi64(pi, x, 8));
  } else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::None<4>, int32_t stride) -> Vec<4, T> {
  return gather(p, mask::None<4>{}, range<4>() * stride);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::None<2>, Vec<2, int64_t> indv) -> Vec<2, T> {
  auto x = std::bit_cast<__m128i>(indv);
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<2, double>>(_mm_i64gather_pd(p, x, 8));
  else if constexpr (sizeof(T) == 8) {
    const long long int *pi = reinterpret_cast<const long long int *>(p);
    return std::bit_cast<Vec<2, T>>(_mm_i64gather_epi64(pi, x, 8));
  } else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::None<2>, int32_t stride) -> Vec<2, T> {
  return gather(p, mask::None<2>{}, range<2>() * stride);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::None<8>, Vec<8, int32_t> indv) -> Vec<8, T> {
  auto inds = std::bit_cast<__m256i>(indv);
  if constexpr (std::same_as<T, float>) {
    return std::bit_cast<Vec<8, T>>(_mm256_i32gather_ps(p, inds, 4));
  } else if constexpr (sizeof(T) == 4) {
    return std::bit_cast<Vec<8, T>>(_mm256_i32gather_epi32(p, inds, 4));
#ifdef __AVX512F__
  } else if constexpr (std::same_as<T, double>) {
    return std::bit_cast<Vec<8, T>>(_mm512_i32gather_pd(inds, p, 8));
  } else if constexpr (sizeof(T) == 8) {
    const long long int *pi = reinterpret_cast<const long long int *>(p);
    return std::bit_cast<Vec<8, T>>(_mm512_i32gather_epi64(inds, pi, 8));
#endif
  } else static_assert(false);
}
#ifdef __AVX512F__
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::None<8>, Vec<8, int64_t> indv) -> Vec<8, T> {
  auto inds = std::bit_cast<__m512i>(indv);
  if constexpr (std::same_as<T, double>) {
    return std::bit_cast<Vec<8, T>>(_mm512_i64gather_pd(inds, p, 8));
  } else if constexpr (sizeof(T) == 8) {
    const long long int *pi = reinterpret_cast<const long long int *>(p);
    return std::bit_cast<Vec<8, T>>(_mm512_i64gather_epi64(inds, pi, 8));
  } else static_assert(false);
}
#endif // AVX512F
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::None<8>, int32_t stride) -> Vec<8, T> {
  return gather(p, mask::None<8>{}, range<8>() * stride);
}
#endif // AVX2
#endif // AVX

// Here, we handle masked loads/stores
#ifdef __AVX512VL__
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *pt, mask::Bit<4> i) -> Vec<4, T> {
  const void *p = pt;
  if constexpr (std::same_as<T, double>) {
    return std::bit_cast<Vec<4, double>>(
      _mm256_maskz_loadu_pd(uint8_t(i.mask_), p));
  } else if constexpr (sizeof(T) == 8) {
    return std::bit_cast<Vec<4, T>>(
      _mm256_maskz_loadu_epi64(uint8_t(i.mask_), p));
  } else if constexpr (std::same_as<T, float>) {
    return std::bit_cast<Vec<4, float>>(
      _mm_maskz_loadu_ps(uint8_t(i.mask_), p));
  } else if constexpr (sizeof(T) == 4) {
    return std::bit_cast<Vec<4, T>>(_mm_maskz_loadu_epi32(uint8_t(i.mask_), p));
  } else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void store(T *pt, mask::Bit<4> i,
                                                          Vec<4, T> x) {
  void *p = pt;
  if constexpr (std::same_as<T, double>) {
    __m256d xd = std::bit_cast<__m256d>(x);
    _mm256_mask_storeu_pd(p, uint8_t(i.mask_), xd);
  } else if constexpr (sizeof(T) == 8) {
    __m256i xi = std::bit_cast<__m256i>(x);
    _mm256_mask_storeu_epi64(p, uint8_t(i.mask_), xi);
  } else if constexpr (std::same_as<T, float>) {
    __m128 xf = std::bit_cast<__m128>(x);
    _mm_mask_storeu_ps(p, uint8_t(i.mask_), xf);
  } else if constexpr (sizeof(T) == 4) {
    __m128i xi = std::bit_cast<__m128i>(x);
    _mm_mask_storeu_epi32(p, uint8_t(i.mask_), xi);
  } else static_assert(false);
}

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Bit<2> i) -> Vec<2, T> {
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<2, double>>(
      _mm_maskz_loadu_pd(uint8_t(i.mask_), p));
  else if constexpr (sizeof(T) == 8)
    return std::bit_cast<Vec<2, T>>(_mm_maskz_loadu_epi64(uint8_t(i.mask_), p));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void store(T *p, mask::Bit<2> i,
                                                          Vec<2, T> x) {
  if constexpr (std::same_as<T, double>)
    _mm_mask_storeu_pd(p, uint8_t(i.mask_), std::bit_cast<__m128d>(x));
  else if constexpr (sizeof(T) == 8)
    _mm_mask_storeu_epi64(p, uint8_t(i.mask_), std::bit_cast<__m128i>(x));
  else static_assert(false);
}
// gather/scatter
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::Bit<4> i, Vec<4, int32_t> indv) -> Vec<4, T> {
  auto inds = std::bit_cast<__m128i>(indv);
  auto src{mmzero<4, T>()};
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<4, double>>(
      _mm256_mmask_i32gather_pd(src, uint8_t(i.mask_), inds, p, 8));
  else if constexpr (sizeof(T) == 8)
    return std::bit_cast<Vec<4, T>>(
      _mm256_mmask_i32gather_epi64(src, uint8_t(i.mask_), inds, p, 8));
  else if constexpr (std::same_as<T, float>)
    return std::bit_cast<Vec<4, float>>(
      _mm_mmask_i32gather_ps(src, uint8_t(i.mask_), inds, p, 4));
  else if constexpr (sizeof(T) == 4)
    return std::bit_cast<Vec<4, T>>(
      _mm_mmask_i32gather_epi32(src, uint8_t(i.mask_), inds, p, 4));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::Bit<4> i, Vec<4, int64_t> indv) -> Vec<4, T> {
  auto inds = std::bit_cast<__m256i>(indv);
  auto src{mmzero<4, T>()};
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<4, double>>(
      _mm256_mmask_i64gather_pd(src, uint8_t(i.mask_), inds, p, 8));
  else if constexpr (sizeof(T) == 8)
    return std::bit_cast<Vec<4, T>>(
      _mm256_mmask_i64gather_epi64(src, uint8_t(i.mask_), inds, p, 8));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Bit<4> i, int32_t stride) -> Vec<4, T> {
  return gather(p, i, range<4>() * stride);
}

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
scatter(T *p, mask::None<4>, Vec<4, T> x, Vec<4, int32_t> indv) {
  auto inds = std::bit_cast<__m128i>(indv);
  if constexpr (std::same_as<T, double>)
    _mm256_i32scatter_pd(p, inds, std::bit_cast<__m256d>(x), 8);
  else if constexpr (sizeof(T) == 8)
    _mm256_i32scatter_epi64(p, inds, std::bit_cast<__m256i>(x), 8);
  else if constexpr (std::same_as<T, float>)
    _mm_i32scatter_ps(p, inds, std::bit_cast<__m128>(x), 4);
  else if constexpr (sizeof(T) == 4)
    _mm_i32scatter_epi32(p, inds, std::bit_cast<__m128i>(x), 4);
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::None<4>, Vec<4, T> x, int32_t stride) {
  scatter(p, mask::None<4>{}, x, range<4>() * stride);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
scatter(T *p, mask::Bit<4> i, Vec<4, T> x, Vec<4, int32_t> indv) {
  auto inds = std::bit_cast<__m128i>(indv);
  if constexpr (std::same_as<T, double>)
    _mm256_mask_i32scatter_pd(p, uint8_t(i.mask_), inds,
                              std::bit_cast<__m256d>(x), 8);
  else if constexpr (sizeof(T) == 8)
    _mm256_mask_i32scatter_epi64(p, uint8_t(i.mask_), inds,
                                 std::bit_cast<__m256i>(x), 8);
  else if constexpr (std::same_as<T, double>)
    _mm_mask_i32scatter_ps(p, uint8_t(i.mask_), inds, std::bit_cast<__m128>(x),
                           4);
  else if constexpr (sizeof(T) == 4)
    _mm_mask_i32scatter_epi32(p, uint8_t(i.mask_), inds,
                              std::bit_cast<__m128i>(x), 4);
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Bit<4> i, Vec<4, T> x, int32_t stride) {
  scatter(p, i, x, range<4>() * stride);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::Bit<2> i, Vec<2, int64_t> indv) -> Vec<2, T> {
  auto inds = std::bit_cast<__m128i>(indv);
  auto src{mmzero<2, T>()};
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<2, double>>(
      _mm_mmask_i64gather_pd(src, uint8_t(i.mask_), inds, p, 8));
  else if constexpr (sizeof(T) == 8)
    return std::bit_cast<Vec<2, T>>(
      _mm_mmask_i64gather_epi64(src, uint8_t(i.mask_), inds, p, 8));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Bit<2> i, int32_t stride) -> Vec<2, T> {
  return gather(p, i, range<2>() * stride);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
scatter(T *p, mask::None<2>, Vec<2, T> x, Vec<2, int64_t> indv) {
  auto inds = std::bit_cast<__m128i>(indv);
  if constexpr (std::same_as<T, double>)
    _mm_i64scatter_pd(p, inds, std::bit_cast<__m128d>(x), 8);
  else if constexpr (sizeof(T) == 8)
    _mm_i64scatter_epi64(p, inds, std::bit_cast<__m128i>(x), 8);
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::None<2>, Vec<2, T> x, int32_t stride) {
  scatter(p, mask::None<2>{}, x, range<2>() * stride);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
scatter(T *p, mask::Bit<2> i, Vec<2, T> x, Vec<2, int64_t> indv) {
  auto inds = std::bit_cast<__m128i>(indv);
  if constexpr (std::same_as<T, double>)
    _mm_mask_i64scatter_pd(p, uint8_t(i.mask_), inds, std::bit_cast<__m128d>(x),
                           8);
  else if constexpr (sizeof(T) == 8)
    _mm_mask_i64scatter_epi64(p, uint8_t(i.mask_), inds,
                              std::bit_cast<__m128i>(x), 8);
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Bit<2> i, Vec<2, T> x, int32_t stride) {
  scatter(p, i, x, range<2>() * stride);
}

template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
fmadd(Vec<W, T> a, Vec<W, T> b, Vec<W, T> c, mask::Bit<W> m) {
  if constexpr (std::same_as<T, double>) {
    if constexpr (W == 8) {
      return std::bit_cast<Vec<W, T>>(_mm512_mask3_fmadd_pd(
        std::bit_cast<__m512d>(a), std::bit_cast<__m512d>(b),
        std::bit_cast<__m512d>(c), uint8_t(m.mask_)));
    } else if constexpr (W == 4) {
      return std::bit_cast<Vec<W, T>>(_mm256_mask3_fmadd_pd(
        std::bit_cast<__m256d>(a), std::bit_cast<__m256d>(b),
        std::bit_cast<__m256d>(c), uint8_t(m.mask_)));
    } else {
      static_assert(W == 2);
      return std::bit_cast<Vec<W, T>>(
        _mm_mask3_fmadd_pd(std::bit_cast<__m128d>(a), std::bit_cast<__m128d>(b),
                           std::bit_cast<__m128d>(c), uint8_t(m.mask_)));
    }
  } else {
    static_assert(std::same_as<T, float>);
    if constexpr (W == 16) {
      return std::bit_cast<Vec<W, T>>(_mm512_mask3_fmadd_ps(
        std::bit_cast<__m512>(a), std::bit_cast<__m512>(b),
        std::bit_cast<__m512>(c), uint16_t(m.mask_)));
    } else if constexpr (W == 8) {
      return std::bit_cast<Vec<W, T>>(_mm256_mask3_fmadd_ps(
        std::bit_cast<__m256>(a), std::bit_cast<__m256>(b),
        std::bit_cast<__m256>(c), uint8_t(m.mask_)));
    } else {
      static_assert(W == 4);
      return std::bit_cast<Vec<W, T>>(
        _mm_mask3_fmadd_ps(std::bit_cast<__m128>(a), std::bit_cast<__m128>(b),
                           std::bit_cast<__m128>(c), uint8_t(m.mask_)));
    }
  }
}

template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
fnmadd(Vec<W, T> a, Vec<W, T> b, Vec<W, T> c, mask::Bit<W> m) {
  if constexpr (std::same_as<T, double>) {
    if constexpr (W == 8) {
      return std::bit_cast<Vec<W, T>>(_mm512_mask3_fnmadd_pd(
        std::bit_cast<__m512d>(a), std::bit_cast<__m512d>(b),
        std::bit_cast<__m512d>(c), uint8_t(m.mask_)));
    } else if constexpr (W == 4) {
      return std::bit_cast<Vec<W, T>>(_mm256_mask3_fnmadd_pd(
        std::bit_cast<__m256d>(a), std::bit_cast<__m256d>(b),
        std::bit_cast<__m256d>(c), uint8_t(m.mask_)));
    } else {
      static_assert(W == 2);
      return std::bit_cast<Vec<W, T>>(_mm_mask3_fnmadd_pd(
        std::bit_cast<__m128d>(a), std::bit_cast<__m128d>(b),
        std::bit_cast<__m128d>(c), uint8_t(m.mask_)));
    }
  } else {
    static_assert(std::same_as<T, float>);
    if constexpr (W == 16) {
      return std::bit_cast<Vec<W, T>>(_mm512_mask3_fnmadd_ps(
        std::bit_cast<__m512>(a), std::bit_cast<__m512>(b),
        std::bit_cast<__m512>(c), uint16_t(m.mask_)));
    } else if constexpr (W == 8) {
      return std::bit_cast<Vec<W, T>>(_mm256_mask3_fnmadd_ps(
        std::bit_cast<__m256>(a), std::bit_cast<__m256>(b),
        std::bit_cast<__m256>(c), uint8_t(m.mask_)));
    } else {
      static_assert(W == 4);
      return std::bit_cast<Vec<W, T>>(
        _mm_mask3_fnmadd_ps(std::bit_cast<__m128>(a), std::bit_cast<__m128>(b),
                            std::bit_cast<__m128>(c), uint8_t(m.mask_)));
    }
  }
}
template <typename T>
constexpr auto select(mask::Bit<4> m, Vec<4, T> x, Vec<4, T> y) -> Vec<4, T> {
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<4, T>>(_mm256_mask_mov_pd(
      std::bit_cast<__m256d>(y), uint8_t(m), std::bit_cast<__m256d>(x)));
  else if constexpr (sizeof(T) == 8)
    return std::bit_cast<Vec<4, T>>(_mm256_mask_mov_epi64(
      std::bit_cast<__m256i>(y), uint8_t(m.mask_), std::bit_cast<__m256i>(x)));
  else if constexpr (std::same_as<T, float>)
    return std::bit_cast<Vec<4, T>>(_mm_mask_mov_ps(
      std::bit_cast<__m128>(y), uint8_t(m), std::bit_cast<__m128>(x)));
  else if constexpr (sizeof(T) == 4)
    return std::bit_cast<Vec<4, T>>(_mm_mask_mov_epi32(
      std::bit_cast<__m128i>(y), uint8_t(m.mask_), std::bit_cast<__m128i>(x)));
  else static_assert(false);
}
template <typename T>
constexpr auto select(mask::Bit<2> m, Vec<2, T> x, Vec<2, T> y) -> Vec<2, T> {
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<2, T>>(_mm_mask_mov_pd(
      std::bit_cast<__m128d>(y), uint8_t(m), std::bit_cast<__m128d>(x)));
  else if constexpr (sizeof(T) == 8)
    return std::bit_cast<Vec<2, T>>(_mm_mask_mov_epi64(
      std::bit_cast<__m128i>(y), uint8_t(m.mask_), std::bit_cast<__m128i>(x)));
  else static_assert(false);
}

#else // No AVX512VL
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
fmadd(Vec<W, T> a, Vec<W, T> b, Vec<W, T> c, mask::Mask<W> m) {
  if constexpr ((W * sizeof(T)) != 64) return m.m ? (a * b + c) : c;
  else if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<W, T>>(_mm512_mask3_fmadd_pd(
      std::bit_cast<__m512d>(a), std::bit_cast<__m512d>(b),
      std::bit_cast<__m512d>(c), uint8_t(m.mask_)));
  else if constexpr (std::same_as<T, float>)
    return std::bit_cast<Vec<W, T>>(
      _mm512_mask3_fmadd_ps(std::bit_cast<__m512>(a), std::bit_cast<__m512>(b),
                            std::bit_cast<__m512>(c), uint16_t(m.mask_)));
  else static_assert(false);
}
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
fnmadd(Vec<W, T> a, Vec<W, T> b, Vec<W, T> c, mask::Mask<W> m) {
  if constexpr ((W * sizeof(T)) != 64) return m.m ? (c - a * b) : c;
  else if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<W, T>>(_mm512_mask3_fnmadd_pd(
      std::bit_cast<__m512d>(a), std::bit_cast<__m512d>(b),
      std::bit_cast<__m512d>(c), uint8_t(m.mask_)));
  else if constexpr (std::same_as<T, float>)
    return std::bit_cast<Vec<W, T>>(
      _mm512_mask3_fnmadd_ps(std::bit_cast<__m512>(a), std::bit_cast<__m512>(b),
                             std::bit_cast<__m512>(c), uint16_t(m.mask_)));
  else static_assert(false);
}

// We need [gather, scatter, load, store] * [unmasked, masked]

// 128 bit fallback scatters
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::None<2>, Vec<2, T> x, int32_t stride) {
  p[0] = x[0];
  p[stride] = x[1];
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
scatter(T *p, mask::None<2>, Vec<2, T> x, Vec<2, int64_t> i) {
  p[i[0]] = x[0];
  p[i[1]] = x[1];
}

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Vector<2, sizeof(T)> i, Vec<2, T> x, int32_t stride) {
  if (i.m[0] != 0) p[0] = x[0];
  if (i.m[1] != 0) p[stride] = x[1];
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
scatter(T *p, mask::Vector<2, sizeof(T)> i, Vec<2, T> x, Vec<2, int64_t> indv) {
  if (i.m[0] != 0) p[indv[0]] = x[0];
  if (i.m[1] != 0) p[indv[1]] = x[1];
}

#ifdef __AVX2__
// masked gathers
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::Vector<4, sizeof(T)> m,
       Vec<4, int32_t> indv) -> Vec<4, T> {
  auto x = std::bit_cast<__m128i>(indv);
  static constexpr auto z = mmzero<4, T>();
  if constexpr (std::same_as<T, double>) {
    return std::bit_cast<Vec<4, double>>(
      _mm256_mask_i32gather_pd(z, p, x, m, 8));
  } else if constexpr (sizeof(T) == 8) {
    const long long int *l = reinterpret_cast<const long long int *>(p);
    return std::bit_cast<Vec<4, T>>(_mm256_mask_i32gather_epi64(z, l, x, m, 8));
  } else if constexpr (std::same_as<T, float>) {
    return std::bit_cast<Vec<4, float>>(_mm_mask_i32gather_ps(z, p, x, m, 4));
  } else if constexpr (sizeof(T) == 4) {
    return std::bit_cast<Vec<4, T>>(_mm_mask_i32gather_epi32(z, p, x, m, 4));
  } else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::Vector<4, sizeof(T)> m,
       Vec<4, int64_t> indv) -> Vec<4, T> {
  auto x = std::bit_cast<__m256i>(indv);
  static constexpr auto z = mmzero<4, T>();
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<4, double>>(
      _mm256_mask_i64gather_pd(z, p, x, m, 8));
  else if constexpr (sizeof(T) == 8)
    return std::bit_cast<Vec<4, T>>(_mm256_mask_i64gather_epi64(z, p, x, m, 8));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Vector<4, sizeof(T)> m, int32_t stride) -> Vec<4, T> {
  return gather(p, m, range<4>() * stride);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::Vector<2, sizeof(T)> m,
       Vec<2, int64_t> indv) -> Vec<2, T> {
  auto x = std::bit_cast<__m128i>(indv);
  __m128i mask = __m128i(m);
  static constexpr auto z = mmzero<2, T>();
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<2, double>>(
      _mm_mask_i64gather_pd(z, p, x, mask, 8));
  else if constexpr (sizeof(T) == 8)
    return std::bit_cast<Vec<2, T>>(_mm_mask_i64gather_epi64(z, p, x, mask, 8));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Vector<2, sizeof(T)> m, int32_t stride) -> Vec<2, T> {
  return gather(p, m, range<2>() * stride);
}

#else // no AVX2
// fallback 128-bit gather
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::None<2>, int32_t stride) -> Vec<2, T> {
  return Vec<2, T>{p[0], p[stride]};
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::None<2>, Vec<2, int64_t> i) -> Vec<2, T> {
  return Vec<2, T>{p[i[0]], p[i[1]]};
}

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Vector<2, sizeof(T)> i, int32_t stride) -> Vec<2, T> {
  return Vec<2, T>{(i.m[0] != 0) ? p[0] : T{}, (i.m[1] != 0) ? p[stride] : T{}};
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::Vector<2, sizeof(T)> i,
       Vec<2, int64_t> indv) -> Vec<2, T> {
  return Vec<2, T>{(i.m[0] != 0) ? p[indv[0]] : T{},
                   (i.m[1] != 0) ? p[indv[1]] : T{}};
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::Vector<2, sizeof(T)> i,
       Vec<2, int32_t> indv) -> Vec<2, T> {
  return Vec<2, T>{(i.m[0] != 0) ? p[indv[0]] : T{},
                   (i.m[1] != 0) ? p[indv[1]] : T{}};
}

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::None<4>, int32_t stride) -> Vec<4, T> {
  return Vec<4, T>{p[0], p[stride], p[2 * stride], p[3 * stride]};
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::None<4>, Vec<4, int32_t> i) -> Vec<4, T> {
  return Vec<4, T>{p[i[0]], p[i[1]], p[i[2]], p[i[3]]};
}

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Vector<4, sizeof(T)> i, int32_t stride) -> Vec<4, T> {
  return Vec<4, T>{(i.m[0] != 0) ? p[0] : T{}, (i.m[1] != 0) ? p[stride] : T{},
                   (i.m[2] != 0) ? p[2 * stride] : T{},
                   (i.m[3] != 0) ? p[3 * stride] : T{}};
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::Vector<4, sizeof(T)> i,
       Vec<4, int> indv) -> Vec<4, T> {
  return Vec<4, T>{
    (i.m[0] != 0) ? p[indv[0]] : T{}, (i.m[1] != 0) ? p[indv[1]] : T{},
    (i.m[2] != 0) ? p[indv[2]] : T{}, (i.m[3] != 0) ? p[indv[3]] : T{}};
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
gather(const T *p, mask::None<4>, Vec<4, int32_t> i) -> Vec<4, T> {
  return Vec<4, T>{p[i[0]], p[i[1]], p[i[2]], p[i[3]]};
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::None<4>, Vec<4, int64_t> i) -> Vec<4, T> {
  return Vec<4, T>{p[i[0]], p[i[1]], p[i[2]], p[i[3]]};
}

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Vector<4, sizeof(T)> i, int32_t stride) -> Vec<4, T> {
  return Vec<4, T>{
    (i.m[0] != 0) ? p[0] : T{},
    (i.m[1] != 0) ? p[stride] : T{},
    (i.m[2] != 0) ? p[2 * stride] : T{},
    (i.m[3] != 0) ? p[3 * stride] : T{},
  };
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::Vector<4, sizeof(T)> i,
       Vec<4, int32_t> indv) -> Vec<4, T> {
  return Vec<4, T>{
    (i.m[0] != 0) ? p[indv[0]] : T{},
    (i.m[1] != 0) ? p[indv[1]] : T{},
    (i.m[2] != 0) ? p[indv[2]] : T{},
    (i.m[3] != 0) ? p[indv[3]] : T{},
  };
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::Vector<4, sizeof(T)> i,
       Vec<4, int64_t> indv) -> Vec<4, T> {
  return Vec<4, T>{
    (i.m[0] != 0) ? p[indv[0]] : T{},
    (i.m[1] != 0) ? p[indv[1]] : T{},
    (i.m[2] != 0) ? p[indv[2]] : T{},
    (i.m[3] != 0) ? p[indv[3]] : T{},
  };
}

#endif // AVX
#endif // no AVX2
#ifdef __AVX__
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Vector<4, sizeof(T)> i) -> Vec<4, T> {
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<4, double>>(_mm256_maskload_pd(p, i));
  else if constexpr (sizeof(T) == 8)
    return std::bit_cast<Vec<4, T>>(
      _mm256_maskload_epi64((const long long *)p, i));
  else if constexpr (std::same_as<T, float>)
    return std::bit_cast<Vec<4, float>>(_mm_maskload_ps(p, i));
  else if constexpr (sizeof(T) == 4)
    return std::bit_cast<Vec<4, T>>(_mm_maskload_epi32((const int *)p, i));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Vector<8, sizeof(T)> i) -> Vec<8, T> {
  if constexpr (std::same_as<T, float>)
    return std::bit_cast<Vec<8, float>>(_mm256_maskload_ps(p, i));
  else if constexpr (sizeof(T) == 4)
    return std::bit_cast<Vec<8, T>>(_mm256_maskload_epi32((const int *)p, i));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Vector<4, sizeof(T)> i, Vec<4, T> x) {
  if constexpr (std::same_as<T, double>)
    _mm256_maskstore_pd(p, i, std::bit_cast<__m256d>(x));
  else if constexpr (sizeof(T) == 8)
    _mm256_maskstore_epi64((long long *)p, i, std::bit_cast<__m256i>(x));
  else if constexpr (std::same_as<T, float>)
    _mm_maskstore_ps(p, i, std::bit_cast<__m128>(x));
  else if constexpr (sizeof(T) == 4)
    _mm_maskstore_epi32((int *)p, i, std::bit_cast<__m128i>(x));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Vector<8, sizeof(T)> i, Vec<8, T> x) {
  if constexpr (std::same_as<T, float>)
    _mm256_maskstore_ps(p, i, std::bit_cast<__m256>(x));
  else if constexpr (sizeof(T) == 4)
    _mm256_maskstore_epi32((int *)p, i, std::bit_cast<__m256i>(x));
  else static_assert(false);
}

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Vector<2, sizeof(T)> i) -> Vec<2, T> {
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<2, double>>(_mm_maskload_pd(p, i));
  else if constexpr (sizeof(T) == 8)
    return std::bit_cast<Vec<2, T>>(
      _mm_maskload_epi64((const long long *)p, i));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Vector<2, sizeof(T)> i, Vec<2, T> x) {
  if constexpr (std::same_as<T, double>)
    _mm_maskstore_pd(p, i, std::bit_cast<__m128d>(x));
  else if constexpr (sizeof(T) == 8)
    _mm_maskstore_epi64((long long *)p, i, std::bit_cast<__m128i>(x));
  else static_assert(false);
}

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
scatter(T *p, mask::None<4>, Vec<4, T> x, Vec<4, int32_t> i) {
  p[i[0]] = x[0];
  p[i[1]] = x[1];
  p[i[2]] = x[2];
  p[i[3]] = x[3];
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Vector<4, sizeof(T)> i, Vec<4, T> x, int32_t stride) {
  if (i.m[0] != 0) p[0] = x[0];
  if (i.m[1] != 0) p[stride] = x[1];
  if (i.m[2] != 0) p[2 * stride] = x[2];
  if (i.m[3] != 0) p[3 * stride] = x[3];
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
scatter(T *p, mask::Vector<4, sizeof(T)> i, Vec<4, T> x, Vec<4, int32_t> indv) {
  if (i.m[0] != 0) p[indv[0]] = x[0];
  if (i.m[1] != 0) p[indv[1]] = x[1];
  if (i.m[2] != 0) p[indv[2]] = x[2];
  if (i.m[3] != 0) p[indv[3]] = x[3];
}
#else // No AVX
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Vector<2, sizeof(T)> i) -> Vec<2, T> {
  return Vec<2, T>{(i.m[0] != 0) ? p[0] : T{}, (i.m[1] != 0) ? p[1] : T{}};
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Vector<4, sizeof(T)> i) -> Vec<4, T> {
  return Vec<4, T>{(i.m[0] != 0) ? p[0] : T{}, (i.m[1] != 0) ? p[1] : T{},
                   (i.m[2] != 0) ? p[2] : T{}, (i.m[3] != 0) ? p[3] : T{}};
}

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Vector<2, sizeof(T)> i, Vec<2, T> x) {
  if (i.m[0] != 0) p[0] = x[0];
  if (i.m[1] != 0) p[1] = x[1];
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Vector<4, sizeof(T)> i, Vec<4, T> x) {
  if (i.m[0] != 0) p[0] = x[0];
  if (i.m[1] != 0) p[1] = x[1];
  if (i.m[2] != 0) p[2] = x[2];
  if (i.m[3] != 0) p[3] = x[3];
}

#endif // No AVX
#endif // No AVX512VL
#ifdef __AVX__
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::None<4>) -> Vec<4, T> {
  if constexpr (std::same_as<T, double>) {
    return std::bit_cast<Vec<4, double>>(_mm256_loadu_pd(p));
  } else if constexpr (sizeof(T) == 8) {
#ifdef __AVX512VL__
    const void *vp = p;
    return std::bit_cast<Vec<4, T>>(_mm256_loadu_epi64(vp));
#else
    const __m256i *pi = reinterpret_cast<const __m256i *>(p);
    return std::bit_cast<Vec<4, T>>(_mm256_loadu_si256(pi));
#endif
  } else if constexpr (std::same_as<T, float>) {
    return std::bit_cast<Vec<4, float>>(_mm_loadu_ps(p));
  } else if constexpr (sizeof(T) == 4) {
#ifdef __AVX512VL__
    const void *vp = p;
    return std::bit_cast<Vec<4, T>>(_mm_loadu_epi32(vp));
#else
    const __m128i *pi = reinterpret_cast<const __m128i *>(p);
    return std::bit_cast<Vec<4, T>>(_mm_loadu_si128(pi));
#endif
  } else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void store(T *p, mask::None<4>,
                                                          Vec<4, T> x) {
  if constexpr (std::same_as<T, double>) {
    __m256d xd = std::bit_cast<__m256d>(x);
    _mm256_storeu_pd(p, xd);
  } else if constexpr (sizeof(T) == 8) {
    __m256i xi = std::bit_cast<__m256i>(x);
#ifdef __AVX512VL__
    void *vp = p;
    _mm256_storeu_epi64(vp, xi);
#else
    __m256i *pi = reinterpret_cast<__m256i *>(p);
    _mm256_storeu_si256(pi, xi);
#endif
  } else if constexpr (std::same_as<T, float>) {
    __m128 xf = std::bit_cast<__m128>(x);
    _mm_storeu_ps(p, xf);
  } else if constexpr (sizeof(T) == 4) {
    __m128i xi = std::bit_cast<__m128i>(x);
#ifdef __AVX512VL__
    void *vp = p;
    _mm_storeu_epi32(vp, xi);
#else
    __m128i *pi = reinterpret_cast<__m128i *>(p);
    _mm_storeu_si128(pi, xi);
#endif
  } else static_assert(false);
}

// // non-power-of-2 memory ops
// template <ptrdiff_t N, typename M>
// constexpr auto fixupnonpow2(index::Vector<N, M> i) {
//   static_assert(std::popcount(size_t(N)) > 1,
//                 "Shouldn't be calling this if not needed.");
//   static constexpr ptrdiff_t W = std::bit_ceil(size_t(N));
//   auto m = i.mask_ & mask(index::VectorMask<W>{N});
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
#else // NO AVX:

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::None<4>) -> Vec<4, T> {
  if constexpr (std::same_as<T, float>)
    return std::bit_cast<Vec<4, float>>(_mm_loadu_ps(p));
  else if constexpr (sizeof(T) == 4)
    return std::bit_cast<Vec<4, T>>(_mm_loadu_si128((const __m128i *)p));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void store(T *p, mask::None<4>,
                                                          Vec<4, T> x) {
  if constexpr (std::same_as<T, float>)
    _mm_storeu_pd(p, std::bit_cast<__m128d>(x));
  else if constexpr (sizeof(T) == 4) {
    __m128i *pi = reinterpret_cast<__m128i *>(p);
    __m128i xi = std::bit_cast<__m128i>(x);
    _mm_storeu_si128(pi, xi);
  } else static_assert(false);
}

#endif // AVX

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::None<2>) -> Vec<2, T> {
  if constexpr (std::same_as<T, double>) {
    double const *dp = p;
    return std::bit_cast<Vec<2, double>>(_mm_loadu_pd(dp));
    // return std::bit_cast<Vec<2, double>>(_mm_loadu_pd(p));
  } else if constexpr (sizeof(T) == 8)
#ifdef __AVX512VL__
    return std::bit_cast<Vec<2, T>>(_mm_loadu_epi64(p));
#else
    return std::bit_cast<Vec<2, T>>(_mm_loadu_si128((const __m128i *)p));
#endif
  else return Vec<2, T>{p[0], p[1]};
  // else static_assert(false);
}

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void store(T *p, mask::None<2>,
                                                          Vec<2, T> x) {
  if constexpr (std::same_as<T, double>) {
    double *dp = p;
    __m128d xpd = std::bit_cast<__m128d>(x);
    _mm_storeu_pd(dp, xpd);
  } else if constexpr (sizeof(T) == 8) {
    __m128i xi = std::bit_cast<__m128i>(x);
#ifdef __AVX512VL__
    void *vp = p;
    _mm_storeu_epi64(vp, xi);
#else
    __m128i *pi = reinterpret_cast<__m128i *>(p);
    _mm_storeu_si128(pi, xi);
#endif
  } else {
    p[0] = x[0];
    p[1] = x[1];
  }
  // else static_assert(false);
}

#ifdef __AVX512F__
template auto load<int64_t>(const int64_t *, mask::None<2>) -> Vec<2, int64_t>;
template auto load<int64_t>(const int64_t *, mask::Bit<2>) -> Vec<2, int64_t>;
template void store<int64_t>(int64_t *, mask::None<2>, Vec<2, int64_t>);
template void store<int64_t>(int64_t *, mask::Bit<2>, Vec<2, int64_t>);
template auto load<int64_t>(const int64_t *, mask::None<4>) -> Vec<4, int64_t>;
template auto load<int64_t>(const int64_t *, mask::Bit<4>) -> Vec<4, int64_t>;
template void store<int64_t>(int64_t *, mask::None<4>, Vec<4, int64_t>);
template void store<int64_t>(int64_t *, mask::Bit<4>, Vec<4, int64_t>);
template auto load<int64_t>(const int64_t *, mask::None<8>) -> Vec<8, int64_t>;
template auto load<int64_t>(const int64_t *, mask::Bit<8>) -> Vec<8, int64_t>;
template void store<int64_t>(int64_t *, mask::None<8>, Vec<8, int64_t>);
template void store<int64_t>(int64_t *, mask::Bit<8>, Vec<8, int64_t>);

template auto load<int64_t>(const int64_t *, mask::None<2>,
                            int32_t) -> Vec<2, int64_t>;
template auto load<int64_t>(const int64_t *, mask::Bit<2>,
                            int32_t) -> Vec<2, int64_t>;
template void store<int64_t>(int64_t *, mask::None<2>, Vec<2, int64_t>,
                             int32_t);
template void store<int64_t>(int64_t *, mask::Bit<2>, Vec<2, int64_t>, int32_t);
template auto load<int64_t>(const int64_t *, mask::None<4>,
                            int32_t) -> Vec<4, int64_t>;
template auto load<int64_t>(const int64_t *, mask::Bit<4>,
                            int32_t) -> Vec<4, int64_t>;
template void store<int64_t>(int64_t *, mask::None<4>, Vec<4, int64_t>,
                             int32_t);
template void store<int64_t>(int64_t *, mask::Bit<4>, Vec<4, int64_t>, int32_t);
template auto load<int64_t>(const int64_t *, mask::None<8>,
                            int32_t) -> Vec<8, int64_t>;
template auto load<int64_t>(const int64_t *, mask::Bit<8>,
                            int32_t) -> Vec<8, int64_t>;
template void store<int64_t>(int64_t *, mask::None<8>, Vec<8, int64_t>,
                             int32_t);
template void store<int64_t>(int64_t *, mask::Bit<8>, Vec<8, int64_t>, int32_t);
#endif

#else // not __x86_64__
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::None<W>) -> Vec<W, T> {
  Vec<W, T> ret;
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w) ret[w] = p[w];
  return ret;
}
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline void store(T *p, mask::None<W>,
                                                          Vec<W, T> x) {
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w) p[w] = x[w];
}

template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Vector<W, 4> i) -> Vec<W, T> {
  Vec<W, T> ret;
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w) ret[w] = (i.m[w] != 0) ? p[w] : T{};
  return ret;
}
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Vector<W, 8> i) -> Vec<W, T> {
  Vec<W, T> ret;
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w) ret[w] = (i.m[w] != 0) ? p[w] : T{};
  return ret;
}
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Vector<W, 4> i, Vec<W, T> x) {
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w)
    if (i.m[w] != 0) p[w] = x[w];
}
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Vector<W, 8> i, Vec<W, T> x) {
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w)
    if (i.m[w] != 0) p[w] = x[w];
}

template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::None<W>, int32_t stride) -> Vec<W, T> {
  Vec<W, T> ret;
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w) ret[w] = p[w * stride];
  return ret;
}
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::None<W>, Vec<W, int32_t> indv) -> Vec<W, T> {
  Vec<W, T> ret;
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w) ret[w] = p[indv[w]];
  return ret;
}
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::None<W>, Vec<W, int64_t> indv) -> Vec<W, T> {
  Vec<W, T> ret;
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w) ret[w] = p[indv[w]];
  return ret;
}
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::None<W>, Vec<W, T> x, int32_t stride) {
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w) p[w * stride] = x[w];
}
template <typename T, ptrdiff_t W, std::integral I>
[[gnu::always_inline, gnu::artificial]] inline void
scatter(T *p, mask::None<W>, Vec<W, T> x, Vec<W, I> indv) {
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w) p[indv[w]] = x[w];
}

template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Vector<W, 4> i, int32_t stride) -> Vec<W, T> {
  Vec<W, T> ret;
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w)
    ret[w] = (i.m[w] != 0) ? p[w * stride] : T{};
  return ret;
}
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Vector<W, 8> i, int32_t stride) -> Vec<W, T> {
  Vec<W, T> ret;
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w)
    ret[w] = (i.m[w] != 0) ? p[w * stride] : T{};
  return ret;
}
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::Vector<W, sizeof(T)> i,
       Vec<W, int32_t> indv) -> Vec<W, T> {
  Vec<W, T> ret;
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w) ret[w] = (i.m[w] != 0) ? p[indv[w]] : T{};
  return ret;
}
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
gather(const T *p, mask::Vector<W, sizeof(T)> i,
       Vec<W, int64_t> indv) -> Vec<W, T> {
  Vec<W, T> ret;
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w) ret[w] = (i.m[w] != 0) ? p[indv[w]] : T{};
  return ret;
}
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Vector<W, 4> i, Vec<W, T> x, int32_t stride) {
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w)
    if (i.m[w] != 0) p[w * stride] = x[w];
}
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Vector<W, 8> i, Vec<W, T> x, int32_t stride) {
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w)
    if (i.m[w] != 0) p[w * stride] = x[w];
}
template <typename T, ptrdiff_t W, std::integral I>
[[gnu::always_inline, gnu::artificial]] inline void
scatter(T *p, mask::Vector<W, sizeof(T)> i, Vec<W, T> x, Vec<W, I> indv) {
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w)
    if (i.m[w] != 0) p[indv[w]] = x[w];
}
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
fmadd(Vec<W, T> a, Vec<W, T> b, Vec<W, T> c, mask::Vector<W, sizeof(T)> m) {
  return m.m ? (a * b + c) : c;
}
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
fnmadd(Vec<W, T> a, Vec<W, T> b, Vec<W, T> c, mask::Vector<W, sizeof(T)> m) {
  return m.m ? (c - a * b) : c;
}

#endif
#ifndef __AVX512F__
template <typename T, ptrdiff_t W>
constexpr auto select(mask::Vector<W, sizeof(T)> m, Vec<W, T> x,
                      Vec<W, T> y) -> Vec<W, T> {
  return m.m ? x : y;
}
#endif
#ifdef __AVX512CD__

// count left zeros
template <ptrdiff_t W, std::integral T>
[[gnu::always_inline, gnu::artificial]] inline constexpr auto clz(Vec<W, T> v) {
  static_assert((sizeof(T) == 4) || (sizeof(T) == 8));
  if constexpr (W == 16) {
    static_assert(sizeof(T) == 4);
    __m512i vi = std::bit_cast<__m512i>(v);
    return std::bit_cast<Vec<W, T>>(_mm512_lzcnt_epi32(vi));
  } else if constexpr (W == 8) {
    if constexpr (sizeof(T) == 8) {
      __m512i vi = std::bit_cast<__m512i>(v);
      return std::bit_cast<Vec<W, T>>(_mm512_lzcnt_epi64(vi));
    } else {
      __m256i vi = std::bit_cast<__m256i>(v);
      return std::bit_cast<Vec<W, T>>(_mm256_lzcnt_epi32(vi));
    }
  } else if constexpr (W == 4) {
    if constexpr (sizeof(T) == 8) {
      __m256i vi = std::bit_cast<__m256i>(v);
      return std::bit_cast<Vec<W, T>>(_mm256_lzcnt_epi64(vi));
    } else {
      __m128i vi = std::bit_cast<__m128i>(v);
      return std::bit_cast<Vec<W, T>>(_mm_lzcnt_epi32(vi));
    }
  } else {
    static_assert(sizeof(T) == 8);
    __m128i vi = std::bit_cast<__m128i>(v);
    return std::bit_cast<Vec<W, T>>(_mm_lzcnt_epi64(vi));
  }
}
// count right zeros
template <ptrdiff_t W, std::integral T>
[[gnu::always_inline, gnu::artificial]] inline constexpr auto crz(Vec<W, T> v) {
  return T(8 * sizeof(T)) - clz<W, T>((~v) & (v - T(1)));
}

#else

template <ptrdiff_t W, std::integral T> constexpr auto clz(Vec<W, T> v) {
  Vec<W, T> ret;
  for (ptrdiff_t w = 0; w < W; ++w)
    ret[w] = T(std::countl_zero(std::make_unsigned_t<T>(v[w])));
  return ret;
}
template <ptrdiff_t W, std::integral T> constexpr auto crz(Vec<W, T> v) {
  Vec<W, T> ret;
  for (ptrdiff_t w = 0; w < W; ++w)
    ret[w] = T(std::countr_zero(std::make_unsigned_t<T>(v[w])));
  return ret;
}

#endif

template <typename T>
inline constexpr ptrdiff_t Width =
  SIMDSupported<T> ? VECTORWIDTH / sizeof(T) : 1;
template <ptrdiff_t N, typename T>
constexpr ptrdiff_t VecLen =
  (N < Width<T>) ? ptrdiff_t(std::bit_ceil(size_t(N)))
                 : std::max(Width<T>, ptrdiff_t(1));

// returns { vector_size, num_vectors, remainder }
template <ptrdiff_t L, typename T>
consteval auto VectorDivRem() -> std::array<ptrdiff_t, 3> {
  constexpr ptrdiff_t W = Width<T>;
  if constexpr (L <= W) {
    constexpr auto V = ptrdiff_t(std::bit_ceil(size_t(L)));
    return {V, L / V, L % V};
  } else return {W, L / W, L % W};
};
template <typename T, ptrdiff_t R, ptrdiff_t C> constexpr auto VecWidth() {
  if constexpr (C > 1) return VectorDivRem<C, T>()[0];
  else if constexpr (C == 1)
    if constexpr (R >= 1) return VectorDivRem<R, T>()[0];
    else return Width<T>;
  else return Width<T>;
}

} // namespace simd
