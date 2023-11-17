#pragma once
#ifndef MATH_SIMD_MASKS_HPP_INCLUDED
#define MATH_SIMD_MASKS_HPP_INCLUDED
#include "SIMD/Vec.hpp"
#include <cstddef>
#include <cstdint>
#ifdef __x86_64__
#include <immintrin.h>
#endif
namespace poly::simd {

template <typename T>
using IntegerOfSize = std::conditional_t<
  sizeof(T) == 8, int64_t,
  std::conditional_t<sizeof(T) == 4, int32_t,
                     std::conditional_t<sizeof(T) == 2, int16_t, int8_t>>>;

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

namespace mask {
template <ptrdiff_t W> struct None {
  static constexpr auto lastUnmasked() -> ptrdiff_t { return W; }
};

// Alternatives we can have: BitMask and VectorMask
// We use `BitMask` on AVX512, VectorMask elsewhere.
// ARM SVE(2) will eventually want bitmasks as well.
#ifdef __x86_64__
#ifdef __AVX512F__
template <ptrdiff_t W> struct Bit {
  uint64_t mask;
  template <std::unsigned_integral U> explicit constexpr operator U() {
    return U(mask);
  }
  explicit constexpr operator bool() const { return mask; }
  [[nodiscard]] constexpr auto lastUnmasked() const -> ptrdiff_t {
    // could make this `countr_ones` if we decide to only
    // support leading masks
    uint64_t m = mask & ((uint64_t(1) << W) - uint64_t(1));
    return 64 - ptrdiff_t(std::countl_zero(m));
  }
};
template <ptrdiff_t W> constexpr auto operator&(Bit<W> a, Bit<W> b) -> Bit<W> {
  return {a.mask & b.mask};
}
template <ptrdiff_t W> constexpr auto operator&(None<W>, Bit<W> b) -> Bit<W> {
  return b;
}
template <ptrdiff_t W> constexpr auto operator&(Bit<W> a, None<W>) -> Bit<W> {
  return a;
}
#endif // AVX512F
#ifdef __AVX512VL__
// In: iteration count `i.i` is the total length of the loop
// Out: mask for the final iteration. Zero indicates no masked iter.
template <ptrdiff_t W> constexpr auto create(ptrdiff_t i) -> Bit<W> {
  static_assert(std::popcount(size_t(W)) == 1);
  utils::invariant(i >= 0);
  return {_bzhi_u64(0xffffffffffffffff, uint64_t(i) & uint64_t(W - 1))};
};
// In: index::Vector where `i.i` is for the current iteration, and total loop
// length. Out: mask for the current iteration, 0 indicates exit loop.
template <ptrdiff_t W>
constexpr auto create(ptrdiff_t i, ptrdiff_t len) -> Bit<W> {
  static_assert(std::popcount(size_t(W)) == 1);
  uint64_t x;
  if (__builtin_sub_overflow(len, i, &x)) x = 0;
  else x = std::min(x, uint64_t(255));
  return {_bzhi_u64(0xffffffffffffffff, x)};
};
#else // ifdef __AVX512VL__

template <ptrdiff_t W, typename I = int64_t> struct Vector {
  static_assert((W == 2) || (W == 4));
  // TODO: add support for smaller mask types, we we can use smaller eltypes
  Vec<W, I> m;
  [[nodiscard]] constexpr auto intmask() const -> int32_t {
    if constexpr (sizeof(I) == 8)
      if constexpr (W == 2) return _mm_movemask_pd(std::bit_cast<__m128d>(m));
      else return _mm256_movemask_pd(std::bit_cast<__m256d>(m));
    else if constexpr (sizeof(I) == 4)
      if constexpr (W == 4) return _mm_movemask_ps(std::bit_cast<__m128>(m));
      else return _mm256_movemask_ps(std::bit_cast<__m256>(m));
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
// but no VL!!! xeon phi
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
#else  // ifdef __AVX512F__
template <ptrdiff_t W> constexpr auto create(ptrdiff_t i) -> Vector<W> {
  return {range<W, int64_t>() < (i & (W - 1))};
}
template <ptrdiff_t W>
constexpr auto create(ptrdiff_t i, ptrdiff_t len) -> Vector<W> {
  return {range<W, int64_t>() + i < len};
}
#endif // ifdef __AVX512F__; else

#endif // ifdef __AVX512VL__; else
#else  // ifdef __x86_64__

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
#endif // ifdef __x86_64__; else
} // namespace mask
} // namespace poly::simd
#endif // Masks_hpp_INCLUDED
