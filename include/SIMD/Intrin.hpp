#pragma once
#ifndef MATH_SIMD_INTRIN_HPP_INCLUDED
#define MATH_SIMD_INTRIN_HPP_INCLUDED

#include <Math/AxisTypes.hpp>
#include <SIMD/Indexing.hpp>
#include <SIMD/Masks.hpp>
#include <SIMD/Vec.hpp>
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
namespace poly::simd {

// Supported means by this library currently; more types may be added in the
// future as needed.
template <typename T>
concept SIMDSupported = std::same_as<T, int64_t> || std::same_as<T, double>;

template <ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto vbroadcast(Vec<W, T> v) -> Vec<W, T> {
  if constexpr (W == 2) return __builtin_shufflevector(v, v, 0, 0);
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
  return vbroadcast<W, T>(Vec<W, T>{x});
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

// TODO: make `consteval` when clang supports it
#ifdef __clang__
template <ptrdiff_t W, typename T> constexpr auto mmzero() {
#else
template <ptrdiff_t W, typename T> consteval auto mmzero() {
#endif
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
static constexpr ptrdiff_t REGISTERS = 32;
static constexpr ptrdiff_t VECTORWIDTH = 64;
// namespace hw {
// typedef double __m512d_u __attribute__((__vector_size__(64),
// __aligned__(8)));

// typedef int64_t __m512i_u __attribute__((__vector_size__(64),
// __aligned__(8))); } // namespace hw

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto load(const T *p,
                                                         mask::None<8>)
  -> Vec<8, T> {
  if constexpr (std::same_as<T, double>) {
    // hw::__m512d_u v = *reinterpret_cast<const hw::__m512d_u *>(p);
    // return std::bit_cast<Vec<8, double>>(v);
    return std::bit_cast<Vec<8, double>>(_mm512_loadu_pd(p));
  } else if constexpr (std::same_as<T, int64_t>) {
    // hw::__m512i_u v = *reinterpret_cast<const hw::__m512i_u *>(p);
    // return std::bit_cast<Vec<8, int64_t>>(v);
    return std::bit_cast<Vec<8, int64_t>>(_mm512_loadu_epi64(p));
  } else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto load(const T *p,
                                                         mask::Bit<8> i)
  -> Vec<8, T> {
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<8, double>>(
      _mm512_maskz_loadu_pd(uint8_t(i.mask), p));
  else if constexpr (std::same_as<T, int64_t>)
    return std::bit_cast<Vec<8, int64_t>>(
      _mm512_maskz_loadu_epi64(uint8_t(i.mask), p));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void store(T *p, mask::None<8>,
                                                          Vec<8, T> x) {
  if constexpr (std::same_as<T, double>) {
    // hw::__m512d_u v = std::bit_cast<Vec<8, double>>(x);
    // *reinterpret_cast<hw::__m512d_u *>(p) = v;
    _mm512_storeu_pd(p, std::bit_cast<__m512d>(x));
  } else if constexpr (std::same_as<T, int64_t>) {
    // hw::__m512i_u v = std::bit_cast<Vec<8, int64_t>>(x);
    // *reinterpret_cast<hw::__m512i_u *>(p) = v;
    _mm512_storeu_epi64(p, std::bit_cast<__m512i>(x));
  } else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void store(T *p, mask::Bit<8> i,
                                                          Vec<8, T> x) {
  if constexpr (std::same_as<T, double>)
    _mm512_mask_storeu_pd(p, uint8_t(i.mask), std::bit_cast<__m512d>(x));
  else if constexpr (std::same_as<T, int64_t>)
    _mm512_mask_storeu_epi64(p, uint8_t(i.mask), std::bit_cast<__m512i>(x));
  else static_assert(false);
}

// strided memory accesses
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::None<8>, int32_t stride) -> Vec<8, T> {
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<8, T>>(
      _mm512_i32gather_pd(vindex<8>(stride), p, 8));
  else if constexpr (std::same_as<T, int64_t>)
    return std::bit_cast<Vec<8, int64_t>>(
      _mm512_i32gather_pd(vindex<8>(stride), p, 8));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Bit<8> i, int32_t stride) -> Vec<8, T> {
  auto src = mmzero<8, T>();
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<8, double>>(
      _mm512_mask_i32gather_pd(src, uint8_t(i.mask), vindex<8>(stride), p, 8));
  else if constexpr (std::same_as<T, int64_t>)
    return std::bit_cast<Vec<8, int64_t>>(_mm512_mask_i32gather_epi64(
      src, uint8_t(i.mask), vindex<8>(stride), p, 8));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::None<8>, Vec<8, T> x, int32_t stride) {
  if constexpr (std::same_as<T, double>)
    _mm512_i32scatter_pd(p, vindex<8>(stride), std::bit_cast<__m512d>(x), 8);
  else if constexpr (std::same_as<T, int64_t>)
    _mm512_i32scatter_epi64(p, vindex<8>(stride), std::bit_cast<__m512i>(x), 8);
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Bit<8> i, Vec<8, T> x, int32_t stride) {
  if constexpr (std::same_as<T, double>)
    _mm512_mask_i32scatter_pd(p, uint8_t(i.mask), vindex<8>(stride),
                              std::bit_cast<__m512d>(x), 8);
  else if constexpr (std::same_as<T, int64_t>)
    _mm512_mask_i32scatter_epi64(p, uint8_t(i.mask), vindex<8>(stride),
                                 std::bit_cast<__m512i>(x), 8);
  else static_assert(false);
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
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::None<4>, int32_t stride) -> Vec<4, T> {
  auto x{vindex<4>(stride)};
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<4, double>>(_mm256_i32gather_pd(p, x, 8));
  else if constexpr (std::same_as<T, int64_t>)
    return std::bit_cast<Vec<4, int64_t>>(_mm256_i32gather_pd(p, x, 8));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::None<2>, int32_t stride) -> Vec<2, T> {
  auto x{vindex<2>(stride)};
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<2, double>>(_mm_i64gather_pd(p, x, 8));
  else if constexpr (std::same_as<T, int64_t>)
    return std::bit_cast<Vec<2, int64_t>>(_mm_i64gather_pd(p, x, 8));
  else static_assert(false);
}
#endif // AVX2

// Here, we handle masked loads/stores
#ifdef __AVX512VL__
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto load(const T *p,
                                                         mask::Bit<4> i)
  -> Vec<4, T> {
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<4, double>>(
      _mm256_maskz_loadu_pd(uint8_t(i.mask), p));
  else if constexpr (std::same_as<T, int64_t>)
    return std::bit_cast<Vec<4, int64_t>>(
      _mm256_maskz_loadu_epi64(uint8_t(i.mask), p));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void store(T *p, mask::Bit<4> i,
                                                          Vec<4, T> x) {
  if constexpr (std::same_as<T, double>)
    _mm256_mask_storeu_pd(p, uint8_t(i.mask), std::bit_cast<__m256d>(x));
  else if constexpr (std::same_as<T, int64_t>)
    _mm256_mask_storeu_epi64(p, uint8_t(i.mask), std::bit_cast<__m256i>(x));
  else static_assert(false);
}

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto load(const T *p,
                                                         mask::Bit<2> i)
  -> Vec<2, T> {
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<2, double>>(
      _mm_maskz_loadu_pd(uint8_t(i.mask), p));
  else if constexpr (std::same_as<T, int64_t>)
    return std::bit_cast<Vec<2, int64_t>>(
      _mm_maskz_loadu_epi64(uint8_t(i.mask), p));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void store(T *p, mask::Bit<2> i,
                                                          Vec<2, T> x) {
  if constexpr (std::same_as<T, double>)
    _mm_mask_storeu_pd(p, uint8_t(i.mask), std::bit_cast<__m128d>(x));
  else if constexpr (std::same_as<T, int64_t>)
    _mm_mask_storeu_epi64(p, uint8_t(i.mask), std::bit_cast<__m128i>(x));
  else static_assert(false);
}
// gather/scatter
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Bit<4> i, int32_t stride) -> Vec<4, T> {
  auto src{mmzero<4, T>()};
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<4, double>>(
      _mm256_mmask_i32gather_pd(src, uint8_t(i.mask), vindex<4>(stride), p, 8));
  else if constexpr (std::same_as<T, int64_t>)
    return std::bit_cast<Vec<4, int64_t>>(_mm256_mmask_i32gather_epi64(
      src, uint8_t(i.mask), vindex<4>(stride), p, 8));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::None<4>, Vec<4, T> x, int32_t stride) {
  if constexpr (std::same_as<T, double>)
    _mm256_i32scatter_pd(p, vindex<4>(stride), std::bit_cast<__m256d>(x), 8);
  else if constexpr (std::same_as<T, int64_t>)
    _mm256_i32scatter_epi64(p, vindex<4>(stride), std::bit_cast<__m256i>(x), 8);
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Bit<4> i, Vec<4, T> x, int32_t stride) {
  if constexpr (std::same_as<T, double>)
    _mm256_mask_i32scatter_pd(p, uint8_t(i.mask), vindex<4>(stride),
                              std::bit_cast<__m256d>(x), 8);
  else if constexpr (std::same_as<T, int64_t>)
    _mm256_mask_i32scatter_epi64(p, uint8_t(i.mask), vindex<4>(stride),
                                 std::bit_cast<__m256i>(x), 8);
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Bit<2> i, int32_t stride) -> Vec<2, T> {
  auto src{mmzero<2, T>()};
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<2, double>>(
      _mm_mmask_i64gather_pd(src, uint8_t(i.mask), vindex<2>(stride), p, 8));
  else if constexpr (std::same_as<T, int64_t>)
    return std::bit_cast<Vec<2, int64_t>>(
      _mm_mmask_i64gather_epi64(src, uint8_t(i.mask), vindex<2>(stride), p, 8));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::None<2>, Vec<2, T> x, int32_t stride) {
  if constexpr (std::same_as<T, double>)
    _mm_i64scatter_pd(p, vindex<2>(stride), std::bit_cast<__m128d>(x), 8);
  else if constexpr (std::same_as<T, int64_t>)
    _mm_i64scatter_epi64(p, vindex<2>(stride), std::bit_cast<__m128i>(x), 8);
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Bit<2> i, Vec<2, T> x, int32_t stride) {
  if constexpr (std::same_as<T, double>)
    _mm_mask_i64scatter_pd(p, uint8_t(i.mask), vindex<2>(stride),
                           std::bit_cast<__m128d>(x), 8);
  else if constexpr (std::same_as<T, int64_t>)
    _mm_mask_i64scatter_epi64(p, uint8_t(i.mask), vindex<2>(stride),
                              std::bit_cast<__m128i>(x), 8);
  else static_assert(false);
}

template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
fmadd(Vec<W, T> a, Vec<W, T> b, Vec<W, T> c, mask::Bit<W> m) {
  if constexpr (std::same_as<T, double>) {
    if constexpr (W == 8) {
      return std::bit_cast<Vec<W, T>>(_mm512_mask3_fmadd_pd(
        std::bit_cast<__m512d>(a), std::bit_cast<__m512d>(b),
        std::bit_cast<__m512d>(c), uint8_t(m.mask)));
    } else if constexpr (W == 4) {
      return std::bit_cast<Vec<W, T>>(_mm256_mask3_fmadd_pd(
        std::bit_cast<__m256d>(a), std::bit_cast<__m256d>(b),
        std::bit_cast<__m256d>(c), uint8_t(m.mask)));
    } else {
      static_assert(W == 2);
      return std::bit_cast<Vec<W, T>>(
        _mm_mask3_fmadd_pd(std::bit_cast<__m128d>(a), std::bit_cast<__m128d>(b),
                           std::bit_cast<__m128d>(c), uint8_t(m.mask)));
    }
  } else {
    static_assert(std::same_as<T, float>);
    if constexpr (W == 16) {
      return std::bit_cast<Vec<W, T>>(_mm512_mask3_fmadd_ps(
        std::bit_cast<__m512>(a), std::bit_cast<__m512>(b),
        std::bit_cast<__m512>(c), uint16_t(m.mask)));
    } else if constexpr (W == 8) {
      return std::bit_cast<Vec<W, T>>(_mm256_mask3_fmadd_ps(
        std::bit_cast<__m256>(a), std::bit_cast<__m256>(b),
        std::bit_cast<__m256>(c), uint8_t(m.mask)));
    } else {
      static_assert(W == 4);
      return std::bit_cast<Vec<W, T>>(
        _mm_mask3_fmadd_ps(std::bit_cast<__m128>(a), std::bit_cast<__m128>(b),
                           std::bit_cast<__m128>(c), uint8_t(m.mask)));
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
        std::bit_cast<__m512d>(c), uint8_t(m.mask)));
    } else if constexpr (W == 4) {
      return std::bit_cast<Vec<W, T>>(_mm256_mask3_fnmadd_pd(
        std::bit_cast<__m256d>(a), std::bit_cast<__m256d>(b),
        std::bit_cast<__m256d>(c), uint8_t(m.mask)));
    } else {
      static_assert(W == 2);
      return std::bit_cast<Vec<W, T>>(_mm_mask3_fnmadd_pd(
        std::bit_cast<__m128d>(a), std::bit_cast<__m128d>(b),
        std::bit_cast<__m128d>(c), uint8_t(m.mask)));
    }
  } else {
    static_assert(std::same_as<T, float>);
    if constexpr (W == 16) {
      return std::bit_cast<Vec<W, T>>(_mm512_mask3_fnmadd_ps(
        std::bit_cast<__m512>(a), std::bit_cast<__m512>(b),
        std::bit_cast<__m512>(c), uint16_t(m.mask)));
    } else if constexpr (W == 8) {
      return std::bit_cast<Vec<W, T>>(_mm256_mask3_fnmadd_ps(
        std::bit_cast<__m256>(a), std::bit_cast<__m256>(b),
        std::bit_cast<__m256>(c), uint8_t(m.mask)));
    } else {
      static_assert(W == 4);
      return std::bit_cast<Vec<W, T>>(
        _mm_mask3_fnmadd_ps(std::bit_cast<__m128>(a), std::bit_cast<__m128>(b),
                            std::bit_cast<__m128>(c), uint8_t(m.mask)));
    }
  }
}

#else // No AVX512VL
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
fmadd(Vec<W, T> a, Vec<W, T> b, Vec<W, T> c, mask::Mask<W> m) {
  if constexpr ((W * sizeof(T)) != 64) return m.m ? (a * b + c) : c;
  else if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<W, T>>(_mm512_mask3_fmadd_pd(
      std::bit_cast<__m512d>(a), std::bit_cast<__m512d>(b),
      std::bit_cast<__m512d>(c), uint8_t(m.mask)));
  else if constexpr (std::same_as<T, float>)
    return std::bit_cast<Vec<W, T>>(
      _mm512_mask3_fmadd_ps(std::bit_cast<__m512>(a), std::bit_cast<__m512>(b),
                            std::bit_cast<__m512>(c), uint16_t(m.mask)));
  else static_assert(false);
}
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
fnmadd(Vec<W, T> a, Vec<W, T> b, Vec<W, T> c, mask::Mask<W> m) {
  if constexpr ((W * sizeof(T)) != 64) return m.m ? (c - a * b) : c;
  else if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<W, T>>(_mm512_mask3_fnmadd_pd(
      std::bit_cast<__m512d>(a), std::bit_cast<__m512d>(b),
      std::bit_cast<__m512d>(c), uint8_t(m.mask)));
  else if constexpr (std::same_as<T, float>)
    return std::bit_cast<Vec<W, T>>(
      _mm512_mask3_fnmadd_ps(std::bit_cast<__m512>(a), std::bit_cast<__m512>(b),
                             std::bit_cast<__m512>(c), uint16_t(m.mask)));
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
store(T *p, mask::Vector<2> i, Vec<2, T> x, int32_t stride) {
  if (i.m[0] != 0) p[0] = x[0];
  if (i.m[1] != 0) p[stride] = x[1];
}

#ifdef __AVX2__
// masked gathers
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Vector<4> m, int32_t stride) -> Vec<4, T> {
  __m128i x = vindex<4>(stride);
  __m256i mask = __m256i(m);
  if constexpr (std::same_as<T, double>) {
    constexpr __m256d z = mmzero<4, double>();
    return std::bit_cast<Vec<4, double>>(
      _mm256_mask_i32gather_pd(z, p, x, mask, 8));
  } else if constexpr (std::same_as<T, int64_t>) {
    constexpr __m256i z = mmzero<4, int64_t>();
    return std::bit_cast<Vec<4, int64_t>>(
      _mm256_mask_i32gather_epi64(z, p, x, mask, 8));
  } else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Vector<2> m, int32_t stride) -> Vec<2, T> {
  __m128i x = vindex<2>(stride), mask = __m128i(m);
  if constexpr (std::same_as<T, double>) {
    constexpr __m128d z = mmzero<2, double>();
    return std::bit_cast<Vec<2, double>>(
      _mm_mask_i64gather_pd(z, p, x, mask, 8));
  } else if constexpr (std::same_as<T, int64_t>) {
    constexpr __m128i z = mmzero<2, int64_t>();
    return std::bit_cast<Vec<2, int64_t>>(
      _mm_mask_i64gather_epi64(z, p, x, mask, 8));
  } else static_assert(false);
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
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto load(const T *p,
                                                         mask::Vector<4> i)
  -> Vec<4, T> {
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<4, double>>(_mm256_maskload_pd(p, i));
  else if constexpr (std::same_as<T, int64_t>)
    return std::bit_cast<Vec<4, int64_t>>(
      _mm256_maskload_epi64((const long long *)p, i));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Vector<4> i, Vec<4, T> x) {
  if constexpr (std::same_as<T, double>)
    _mm256_maskstore_pd(p, i, std::bit_cast<__m256d>(x));
  else if constexpr (std::same_as<T, int64_t>)
    _mm256_maskstore_epi64((long long *)p, i, std::bit_cast<__m256i>(x));
  else static_assert(false);
}

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto load(const T *p,
                                                         mask::Vector<2> i)
  -> Vec<2, T> {
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<2, double>>(_mm_maskload_pd(p, i));
  else if constexpr (std::same_as<T, int64_t>)
    return std::bit_cast<Vec<2, int64_t>>(
      _mm_maskload_epi64((const long long *)p, i));
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Vector<2> i, Vec<2, T> x) {
  if constexpr (std::same_as<T, double>)
    _mm_maskstore_pd(p, i, std::bit_cast<__m128d>(x));
  else if constexpr (std::same_as<T, int64_t>)
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
store(T *p, mask::Vector<4> i, Vec<4, T> x, int32_t stride) {
  if (i.m[0] != 0) p[0] = x[0];
  if (i.m[1] != 0) p[stride] = x[1];
  if (i.m[2] != 0) p[2 * stride] = x[2];
  if (i.m[3] != 0) p[3 * stride] = x[3];
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
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto load(const T *p,
                                                         mask::None<4>)
  -> Vec<4, T> {
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<4, double>>(_mm256_loadu_pd(p));
  else if constexpr (std::same_as<T, int64_t>)
#ifdef __AVX512VL__
    return std::bit_cast<Vec<4, int64_t>>(_mm256_loadu_epi64(p));
#else
    return std::bit_cast<Vec<4, int64_t>>(
      _mm256_loadu_si256((const __m256i *)p));
#endif
  else static_assert(false);
}
template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void store(T *p, mask::None<4>,
                                                          Vec<4, T> x) {
  if constexpr (std::same_as<T, double>)
    _mm256_storeu_pd(p, std::bit_cast<__m256d>(x));
  else if constexpr (std::same_as<T, int64_t>)
#ifdef __AVX512VL__
    _mm256_storeu_epi64(p, std::bit_cast<__m256i>(x));
#else
    _mm256_storeu_si256((__m256i *)p, std::bit_cast<__m256i>(x));
#endif
  else static_assert(false);
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

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline auto load(const T *p,
                                                         mask::None<2>)
  -> Vec<2, T> {
  if constexpr (std::same_as<T, double>)
    return std::bit_cast<Vec<2, double>>(_mm_loadu_pd(p));
  else if constexpr (std::same_as<T, int64_t>)
#ifdef __AVX512VL__
    return std::bit_cast<Vec<2, int64_t>>(_mm_loadu_epi64(p));
#else
    return std::bit_cast<Vec<2, int64_t>>(_mm_loadu_si128((const __m128i *)p));
#endif
  else static_assert(false);
}

template <typename T>
[[gnu::always_inline, gnu::artificial]] inline void store(T *p, mask::None<2>,
                                                          Vec<2, T> x) {
  if constexpr (std::same_as<T, double>)
    _mm_storeu_pd(p, std::bit_cast<__m128d>(x));
  else if constexpr (std::same_as<T, int64_t>)
#ifdef __AVX512VL__
    _mm_storeu_epi64(p, std::bit_cast<__m128i>(x));
#else
    _mm_storeu_si128((__m128i *)p, std::bit_cast<__m128i>(x));
#endif
  else static_assert(false);
}

#else // not __x86_64__
static constexpr ptrdiff_t REGISTERS = 32;
static constexpr ptrdiff_t VECTORWIDTH = 16;
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto load(const T *p,
                                                         mask::None<W>)
  -> Vec<W, T> {
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
[[gnu::always_inline, gnu::artificial]] inline auto load(const T *p,
                                                         mask::Vector<W> i)
  -> Vec<W, T> {
  Vec<W, T> ret;
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w) ret[w] = (i.m[w] != 0) ? p[w] : T{};
  return ret;
}
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Vector<W> i, Vec<W, T> x) {
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
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::None<W>, Vec<W, T> x, int32_t stride) {
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w) p[w * stride] = x[w];
}

template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
load(const T *p, mask::Vector<W> i, int32_t stride) -> Vec<W, T> {
  Vec<W, T> ret;
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w)
    ret[w] = (i.m[w] != 0) ? p[w * stride] : T{};
  return ret;
}
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline void
store(T *p, mask::Vector<W> i, Vec<W, T> x, int32_t stride) {
  POLYMATHFULLUNROLL
  for (ptrdiff_t w = 0; w < W; ++w)
    if (i.m[w] != 0) p[w * stride] = x[w];
}
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
fmadd(Vec<W, T> a, Vec<W, T> b, Vec<W, T> c, mask::Vector<W> m) {
  return m.m ? (a * b + c) : c;
}
template <typename T, ptrdiff_t W>
[[gnu::always_inline, gnu::artificial]] inline auto
fnmadd(Vec<W, T> a, Vec<W, T> b, Vec<W, T> c, mask::Vector<W> m) {
  return m.m ? (c - a * b) : c;
}

#endif
#ifdef __AVX512CD__

// count left zeros
template <ptrdiff_t W, std::integral T> constexpr auto clz(Vec<W, T> v) {
  static_assert((sizeof(T) == 4) || (sizeof(T) == 8));
  if constexpr (W == 16) {
    static_assert(sizeof(T) == 4);
    return std::bit_cast<Vec<W, T>>(
      _mm512_lzcnt_epi32(std::bit_cast<__m512i>(v)));
  } else if constexpr (W == 8) {
    if constexpr (sizeof(T) == 8)
      return std::bit_cast<Vec<W, T>>(
        _mm512_lzcnt_epi64(std::bit_cast<__m512i>(v)));
    else
      return std::bit_cast<Vec<W, T>>(
        _mm256_lzcnt_epi32(std::bit_cast<__m256i>(v)));
  } else if constexpr (W == 4) {
    if constexpr (sizeof(T) == 8)
      return std::bit_cast<Vec<W, T>>(
        _mm256_lzcnt_epi64(std::bit_cast<__m256i>(v)));
    else
      return std::bit_cast<Vec<W, T>>(
        _mm_lzcnt_epi32(std::bit_cast<__m128i>(v)));
  } else {
    static_assert(sizeof(T) == 8);
    return std::bit_cast<Vec<W, T>>(_mm_lzcnt_epi64(std::bit_cast<__m128i>(v)));
  }
}
// count right zeros
template <ptrdiff_t W, std::integral T> constexpr auto crz(Vec<W, T> v) {
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
static constexpr ptrdiff_t Width =
  SIMDSupported<T> ? VECTORWIDTH / sizeof(T) : 1;

// returns { vector_size, num_vectors, remainder }
template <ptrdiff_t L, typename T>
consteval auto VectorDivRem() -> std::array<ptrdiff_t, 3> {
  constexpr ptrdiff_t W = Width<T>;
  if constexpr (L <= W) {
    constexpr auto V = ptrdiff_t(std::bit_ceil(size_t(L)));
    return {V, L / V, L % V};
  } else return {W, L / W, L % W};
};

} // namespace poly::simd
#endif // SIMD_hpp_INCLUDED
