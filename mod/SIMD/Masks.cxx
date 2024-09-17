#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifdef __x86_64__
#include <immintrin.h>
#endif

#ifndef USE_MODULE
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "SIMD/Vec.cxx"
#include "Utilities/Invariant.cxx"
#include "Utilities/Widen.cxx"
#else
export module SIMD:Mask;

import :Vec;
import Invariant;
import STL;
import Widen;
#endif

#ifdef USE_MODULE
export namespace simd {
#else
namespace simd {
#endif
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

#if defined(__x86_64__) && defined(__AVX512VL__)
template <ptrdiff_t W>
[[gnu::always_inline]] constexpr auto
sextelts(Vec<W, int32_t> v) -> Vec<W, int64_t> {
  if constexpr (W == 2) {
    return std::bit_cast<Vec<2, int64_t>>(
      _mm_cvtepi32_epi64(std::bit_cast<__m128i>(v)));
  } else if constexpr (W == 4) {
    return std::bit_cast<Vec<4, int64_t>>(
      _mm256_cvtepi32_epi64(std::bit_cast<__m128i>(v)));
  } else if constexpr (W == 8) {
    return std::bit_cast<Vec<8, int64_t>>(
      _mm512_cvtepi32_epi64(std::bit_cast<__m256i>(v)));
  } else static_assert(false);
}
template <ptrdiff_t W>
[[gnu::always_inline]] constexpr auto
zextelts(Vec<W, int32_t> v) -> Vec<W, int64_t> {
  if constexpr (W == 2) {
    return std::bit_cast<Vec<2, int64_t>>(
      _mm_cvtepu32_epi64(std::bit_cast<__m128i>(v)));
  } else if constexpr (W == 4) {
    return std::bit_cast<Vec<4, int64_t>>(
      _mm256_cvtepu32_epi64(std::bit_cast<__m128i>(v)));
  } else if constexpr (W == 8) {
    return std::bit_cast<Vec<8, int64_t>>(
      _mm512_cvtepu32_epi64(std::bit_cast<__m256i>(v)));
  } else static_assert(false);
}
template <ptrdiff_t W>
[[gnu::always_inline]] constexpr auto
truncelts(Vec<W, int64_t> v) -> Vec<W, int32_t> {
  if constexpr (W == 2) {
    return std::bit_cast<Vec<2, int32_t>>(
      _mm_cvtepi64_epi32(std::bit_cast<__m128i>(v)));
  } else if constexpr (W == 4) {
    return std::bit_cast<Vec<4, int32_t>>(
      _mm256_cvtepi64_epi32(std::bit_cast<__m256i>(v)));
  } else if constexpr (W == 8) {
    return std::bit_cast<Vec<8, int32_t>>(
      _mm512_cvtepi64_epi32(std::bit_cast<__m512i>(v)));
  } else static_assert(false);
}
#else
template <ptrdiff_t W>
[[gnu::always_inline]] constexpr auto
sextelts(Vec<W, int32_t> v) -> Vec<W, int64_t> {
  if constexpr (W != 1) {
    Vec<W, int64_t> r;
    for (ptrdiff_t w = 0; w < W; ++w) r[w] = static_cast<int64_t>(v[w]);
    return r;
  } else return static_cast<int64_t>(v);
}
template <ptrdiff_t W>
[[gnu::always_inline]] constexpr auto
zextelts(Vec<W, int32_t> v) -> Vec<W, int64_t> {
  using R = Vec<W, int64_t>;
  static constexpr Vec<W, int32_t> z{};
  if constexpr (W == 1)
    return static_cast<int64_t>(
      static_cast<uint64_t>(static_cast<uint32_t>(v)));
  else if constexpr (W == 2)
    return std::bit_cast<R>(__builtin_shufflevector(v, z, 0, 2, 1, 3));
  else if constexpr (W == 4)
    return std::bit_cast<R>(
      __builtin_shufflevector(v, z, 0, 4, 1, 5, 2, 6, 3, 7));
  else if constexpr (W == 8)
    return std::bit_cast<R>(__builtin_shufflevector(
      v, z, 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15));
  else static_assert(false);
}
template <ptrdiff_t W>
[[gnu::always_inline]] constexpr auto
truncelts(Vec<W, int64_t> v) -> Vec<W, int32_t> {
  using R = Vec<W, int64_t>;
  if constexpr (W == 1) return static_cast<R>(v);
  else {
    Vec<2 * W, int32_t> x = std::bit_cast<Vec<2 * W, int32_t>>(v);
    if constexpr (W == 2) return __builtin_shufflevector(x, x, 0, 2);
    else if constexpr (W == 4) return __builtin_shufflevector(x, x, 0, 2, 4, 6);
    else if constexpr (W == 8)
      return __builtin_shufflevector(x, x, 0, 2, 4, 6, 8, 10, 12, 14);
    else static_assert(false);
  }
}
#endif

namespace mask {
template <ptrdiff_t W> struct None {
  static constexpr auto firstMasked() -> ptrdiff_t { return 0; }
  static constexpr auto lastUnmasked() -> ptrdiff_t { return W; }
};

// Alternatives we can have: BitMask and VectorMask
// We use `BitMask` on AVX512, VectorMask elsewhere.
// ARM SVE(2) will eventually want bitmasks as well.
#ifdef __x86_64__
#ifdef __AVX512F__
template <ptrdiff_t W> struct Bit {
  uint64_t mask_;
  template <std::unsigned_integral U> explicit constexpr operator U() {
    return U(mask_);
  }
  explicit constexpr operator bool() const { return mask_; }
  [[nodiscard]] constexpr auto firstMasked() const -> ptrdiff_t {
    return std::countr_zero(mask_);
  }
  [[nodiscard]] constexpr auto lastUnmasked() const -> ptrdiff_t {
    if constexpr (W < 64) {
      // could make this `countr_ones` if we decide to only
      // support leading masks
      uint64_t m = mask_ & ((uint64_t(1) << W) - uint64_t(1));
      return 64 - ptrdiff_t(std::countl_zero(m));
    } else return 64 - ptrdiff_t(std::countl_zero(mask_));
  }
  template <ptrdiff_t S> [[nodiscard]] constexpr auto sub() -> Bit<S> {
    static_assert(S <= W);
    uint64_t s = mask_;
    mask_ >>= S;
    return {s};
  }

private:
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator&(Bit<W> a, Bit<W> b) -> Bit<W> {
    return {a.mask_ & b.mask_};
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator&(None<W>, Bit<W> b) -> Bit<W> {
    return b;
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator&(Bit<W> a, None<W>) -> Bit<W> {
    return a;
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator|(Bit<W> a, Bit<W> b) -> Bit<W> {
    return {a.mask_ | b.mask_};
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator|(None<W>, Bit<W>) -> Bit<W> {
    return None<W>{};
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator|(Bit<W>, None<W>) -> Bit<W> {
    return None<W>{};
  }
};
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
  if (__builtin_usubl_overflow(len, i, &x)) return {0};
  if (x >= 64) return {0xffffffffffffffff};
  return {_bzhi_u64(0xffffffffffffffff, x)};
};
// Requires: 0 <= m <= 255
template <ptrdiff_t W>
constexpr auto createSmallPositive(ptrdiff_t m) -> Bit<W> {
  static_assert(std::popcount(size_t(W)) == 1);
  utils::invariant(0 <= m);
  utils::invariant(m <= 255);
  return {_bzhi_u64(0xffffffffffffffff, uint64_t(m))};
};

template <ptrdiff_t W> using Mask = Bit<W>;

#else // ifdef __AVX512VL__

template <ptrdiff_t W, size_t Bytes> struct Vector {
  static_assert(Bytes <= 8, "Only at most 8 bytes per element supported.");
  using I = utils::signed_integer_t<Bytes>;
  static_assert(sizeof(I) == Bytes);
  // static_assert(sizeof(I) * W <= VECTORWIDTH);
  // TODO: add support for smaller mask types, we we can use smaller eltypes
  Vec<W, I> m;
  template <size_t newBytes> constexpr operator Vector<W, newBytes>() {
    if constexpr (newBytes == Bytes) return *this;
    else if constexpr (newBytes == 2 * Bytes) return {sextelts<W>(m)};
    else if constexpr (2 * newBytes == Bytes) return {truncelts<W>(m)};
    else static_assert(false);
  }
  [[nodiscard]] constexpr auto intmask() const -> int32_t {
    if constexpr (sizeof(I) == 8)
      if constexpr (W == 2) {
        __m128d arg = std::bit_cast<__m128d>(m);
        return _mm_movemask_pd(arg);
      } else return _mm256_movemask_pd(std::bit_cast<__m256d>(m));
    else if constexpr (sizeof(I) == 4)
      if constexpr (W == 4) {
        __m128 mm = std::bit_cast<__m128>(m);
        return _mm_movemask_ps(mm);
      } else return _mm256_movemask_ps(std::bit_cast<__m256>(m));
    else if constexpr (W == 16)
      return _mm_movemask_epi8(std::bit_cast<__m128i>(m));
    else return _mm256_movemask_epi8(std::bit_cast<__m256i>(m));
  }
  explicit constexpr operator bool() const { return intmask(); }
  [[nodiscard]] constexpr auto firstMasked() const -> ptrdiff_t {
    return std::countr_zero(uint32_t(intmask()));
  }
  [[nodiscard]] constexpr auto lastUnmasked() const -> ptrdiff_t {
    return 32 - std::countl_zero(uint32_t(intmask()));
  }
  constexpr operator __m128i()
  requires(sizeof(I) * W == 16)
  {
    return std::bit_cast<__m128i>(m);
  }
  constexpr operator __m128d()
  requires(sizeof(I) * W == 16)
  {
    return std::bit_cast<__m128d>(m);
  }
  constexpr operator __m128()
  requires(sizeof(I) * W == 16)
  {
    return std::bit_cast<__m128>(m);
  }
  constexpr operator __m256i()
  requires(sizeof(I) * W == 32)
  {
    return std::bit_cast<__m256i>(m);
  }
  constexpr operator __m256d()
  requires(sizeof(I) * W == 32)
  {
    return std::bit_cast<__m256d>(m);
  }
  constexpr operator __m256()
  requires(sizeof(I) * W == 32)
  {
    return std::bit_cast<__m256>(m);
  }

private:
  friend constexpr auto operator&(Vector a, Vector b) -> Vector {
    return {a.m & b.m};
  }
  friend constexpr auto operator&(mask::None<W>, Vector b) -> Vector {
    return b;
  }
  friend constexpr auto operator&(Vector a, mask::None<W>) -> Vector {
    return a;
  }
  friend constexpr auto operator|(Vector a, Vector b) -> Vector {
    return {a.m | b.m};
  }
  friend constexpr auto operator|(mask::None<W>, Vector) -> None<W> {
    return {};
  }
  friend constexpr auto operator|(Vector, mask::None<W>) -> None<W> {
    return {};
  }
};
static_assert(!std::convertible_to<Vector<2, 8>, Vector<4, 8>>);
static_assert(!std::convertible_to<Vector<4, 4>, Vector<8, 4>>);
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
template <ptrdiff_t W, typename I = int64_t>
using Mask = std::conditional_t<sizeof(I) * W == 64, Bit<W>, Vector<W, I>>;
#else  // ifdef __AVX512F__

template <ptrdiff_t W>
constexpr auto create(ptrdiff_t i) -> Vector<W, std::min(8z, VECTORWIDTH / W)> {
  static constexpr ptrdiff_t R = VECTORWIDTH / W;
  using I = utils::signed_integer_t<R >= 8 ? 8 : R>;
  return {range<W, I>() < static_cast<I>(i & (W - 1))};
}
template <ptrdiff_t W>
constexpr auto
create(ptrdiff_t i, ptrdiff_t len) -> Vector<W, std::min(8z, VECTORWIDTH / W)> {
  static constexpr ptrdiff_t R = VECTORWIDTH / W;
  using I = utils::signed_integer_t<R >= 8 ? 8 : R>;
  return {range<W, I>() + static_cast<I>(i) < static_cast<I>(len)};
}
template <ptrdiff_t W, typename I = int64_t> using Mask = Vector<W, sizeof(I)>;
#endif // ifdef __AVX512F__; else

#endif // ifdef __AVX512VL__; else
#else  // ifdef __x86_64__

template <ptrdiff_t W, size_t Bytes> struct Vector {
  using I = utils::signed_integer_t<Bytes>;
  static_assert(sizeof(I) == Bytes);
  Vec<W, I> m;
  template <size_t newBytes> constexpr operator Vector<W, newBytes>() {
    if constexpr (newBytes == Bytes) return *this;
    else if constexpr (newBytes == 2 * Bytes) return {sextelts<W>(m)};
    else if constexpr (2 * newBytes == Bytes) return {truncelts<W>(m)};
    else static_assert(false);
  }
  explicit constexpr operator bool() {
    bool any{false};
    for (ptrdiff_t w = 0; w < W; ++w) any |= m[w];
    return any;
  }
  [[nodiscard]] constexpr auto firstMasked() const -> ptrdiff_t {
    for (ptrdiff_t w = 0; w < W; ++w)
      if (m[w]) return w;
    return W;
  }
  [[nodiscard]] constexpr auto lastUnmasked() const -> ptrdiff_t {
    ptrdiff_t l = 0;
    for (ptrdiff_t w = 0; w < W; ++w)
      if (m[w]) l = w;
    return l;
  }
  [[nodiscard]] explicit constexpr operator bool() const {
    if constexpr (W == 2) {
      return m[0] || m[1];
    } else {
      for (ptrdiff_t w = 0; w < W; ++w)
        if (m[w]) return true;
      return false;
    }
  }

private:
  friend constexpr auto operator&(Vector a, Vector b) -> Vector {
    return {a.m & b.m};
  }
  friend constexpr auto operator&(mask::None<W>, Vector b) -> Vector {
    return b;
  }
  friend constexpr auto operator&(Vector a, mask::None<W>) -> Vector {
    return a;
  }
  friend constexpr auto operator|(Vector a, Vector b) -> Vector {
    return {a.m | b.m};
  }
  friend constexpr auto operator|(mask::None<W>, Vector) -> None<W> {
    return {};
  }
  friend constexpr auto operator|(Vector, mask::None<W>) -> None<W> {
    return {};
  }
};

template <ptrdiff_t W>
constexpr auto create(ptrdiff_t i) -> Vector<W, VECTORWIDTH / W> {
  using I = utils::signed_integer_t<VECTORWIDTH / W>;
  return {range<W, I>() < static_cast<I>(i & (W - 1))};
}
template <ptrdiff_t W>
constexpr auto create(ptrdiff_t i,
                      ptrdiff_t len) -> Vector<W, VECTORWIDTH / W> {
  using I = utils::signed_integer_t<VECTORWIDTH / W>;
  return {range<W, I>() + static_cast<I>(i) < static_cast<I>(len)};
}
#endif // ifdef __x86_64__; else
} // namespace mask

namespace cmp {
#ifdef __AVX512VL__

template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
eq(Vec<W, T> x, Vec<W, T> y) -> mask::Bit<W> {
  if constexpr (W == 16) {
    if constexpr (std::same_as<T, float>) // UQ (unordered quiet?)
      return {_mm512_cmp_ps_mask(std::bit_cast<__m512>(x),
                                 std::bit_cast<__m512>(y), 8)};
    else if constexpr (sizeof(T) == 4)
      return {_mm512_cmp_epi32_mask(std::bit_cast<__m512i>(x),
                                    std::bit_cast<__m512i>(y), 0)};
    else static_assert(false);
  } else if constexpr (W == 8) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm512_cmp_pd_mask(std::bit_cast<__m512d>(x),
                                 std::bit_cast<__m512d>(y), 8)};
    else if constexpr (sizeof(T) == 8)
      return {_mm512_cmp_epi64_mask(std::bit_cast<__m512i>(x),
                                    std::bit_cast<__m512i>(y), 0)};
    else if constexpr (std::same_as<T, float>) // UQ (unordered quiet?)
      return {_mm256_cmp_ps_mask(std::bit_cast<__m256>(x),
                                 std::bit_cast<__m256>(y), 8)};
    else if constexpr (sizeof(T) == 4)
      return {_mm256_cmp_epi32_mask(std::bit_cast<__m256i>(x),
                                    std::bit_cast<__m256i>(y), 0)};
    else static_assert(false);
  } else if constexpr (W == 4) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm256_cmp_pd_mask(std::bit_cast<__m256d>(x),
                                 std::bit_cast<__m256d>(y), 8)};
    else if constexpr (sizeof(T) == 8)
      return {_mm256_cmp_epi64_mask(std::bit_cast<__m256i>(x),
                                    std::bit_cast<__m256i>(y), 0)};
    else if constexpr (std::same_as<T, float>) // UQ (unordered quiet?)
      return {
        _mm_cmp_ps_mask(std::bit_cast<__m128>(x), std::bit_cast<__m128>(y), 8)};
    else if constexpr (sizeof(T) == 4)
      return {_mm_cmp_epi32_mask(std::bit_cast<__m128i>(x),
                                 std::bit_cast<__m128i>(y), 0)};
    else static_assert(false);
  } else if constexpr (W == 2) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm_cmp_pd_mask(std::bit_cast<__m128d>(x),
                              std::bit_cast<__m128d>(y), 8)};
    else if constexpr (sizeof(T) == 8)
      return {_mm_cmp_epi64_mask(std::bit_cast<__m128i>(x),
                                 std::bit_cast<__m128i>(y), 0)};
    else static_assert(false);
  } else static_assert(false);
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
ne(Vec<W, T> x, Vec<W, T> y) -> mask::Bit<W> {
  if constexpr (W == 16) {
    if constexpr (std::same_as<T, float>) // UQ (unordered quiet?)
      return {_mm512_cmp_ps_mask(std::bit_cast<__m512>(x),
                                 std::bit_cast<__m512>(y), 4)};
    else if constexpr (sizeof(T) == 4)
      return {_mm512_cmp_epi32_mask(std::bit_cast<__m512i>(x),
                                    std::bit_cast<__m512i>(y), 4)};
    else static_assert(false);
  } else if constexpr (W == 8) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm512_cmp_pd_mask(std::bit_cast<__m512d>(x),
                                 std::bit_cast<__m512d>(y), 4)};
    else if constexpr (sizeof(T) == 8)
      return {_mm512_cmp_epi64_mask(std::bit_cast<__m512i>(x),
                                    std::bit_cast<__m512i>(y), 4)};
    else if constexpr (std::same_as<T, float>) // UQ (unordered quiet?)
      return {_mm256_cmp_ps_mask(std::bit_cast<__m256>(x),
                                 std::bit_cast<__m256>(y), 4)};
    else if constexpr (sizeof(T) == 4)
      return {_mm256_cmp_epi32_mask(std::bit_cast<__m256i>(x),
                                    std::bit_cast<__m256i>(y), 4)};
    else static_assert(false);
  } else if constexpr (W == 4) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm256_cmp_pd_mask(std::bit_cast<__m256d>(x),
                                 std::bit_cast<__m256d>(y), 4)};
    else if constexpr (sizeof(T) == 8)
      return {_mm256_cmp_epi64_mask(std::bit_cast<__m256i>(x),
                                    std::bit_cast<__m256i>(y), 4)};
    else if constexpr (std::same_as<T, float>) // UQ (unordered quiet?)
      return {
        _mm_cmp_ps_mask(std::bit_cast<__m128>(x), std::bit_cast<__m128>(y), 4)};
    else if constexpr (sizeof(T) == 4)
      return {_mm_cmp_epi32_mask(std::bit_cast<__m128i>(x),
                                 std::bit_cast<__m128i>(y), 4)};
    else static_assert(false);
  } else if constexpr (W == 2) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm_cmp_pd_mask(std::bit_cast<__m128d>(x),
                              std::bit_cast<__m128d>(y), 4)};
    else if constexpr (sizeof(T) == 8)
      return {_mm_cmp_epi64_mask(std::bit_cast<__m128i>(x),
                                 std::bit_cast<__m128i>(y), 4)};
    else static_assert(false);
  } else static_assert(false);
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
lt(Vec<W, T> x, Vec<W, T> y) -> mask::Bit<W> {
  if constexpr (W == 16) {
    if constexpr (std::same_as<T, float>) // UQ (unordered quiet?)
      return {_mm512_cmp_ps_mask(std::bit_cast<__m512>(x),
                                 std::bit_cast<__m512>(y), 25)};
    else if constexpr (sizeof(T) == 4)
      return {_mm512_cmp_epi32_mask(std::bit_cast<__m512i>(x),
                                    std::bit_cast<__m512i>(y), 1)};
    else static_assert(false);
  } else if constexpr (W == 8) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm512_cmp_pd_mask(std::bit_cast<__m512d>(x),
                                 std::bit_cast<__m512d>(y), 25)};
    else if constexpr (sizeof(T) == 8)
      return {_mm512_cmp_epi64_mask(std::bit_cast<__m512i>(x),
                                    std::bit_cast<__m512i>(y), 1)};
    else if constexpr (std::same_as<T, float>) // UQ (unordered quiet?)
      return {_mm256_cmp_ps_mask(std::bit_cast<__m256>(x),
                                 std::bit_cast<__m256>(y), 25)};
    else if constexpr (sizeof(T) == 4)
      return {_mm256_cmp_epi32_mask(std::bit_cast<__m256i>(x),
                                    std::bit_cast<__m256i>(y), 1)};
    else static_assert(false);
  } else if constexpr (W == 4) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm256_cmp_pd_mask(std::bit_cast<__m256d>(x),
                                 std::bit_cast<__m256d>(y), 25)};
    else if constexpr (sizeof(T) == 8)
      return {_mm256_cmp_epi64_mask(std::bit_cast<__m256i>(x),
                                    std::bit_cast<__m256i>(y), 1)};
    else if constexpr (std::same_as<T, float>) // UQ (unordered quiet?)
      return {_mm_cmp_ps_mask(std::bit_cast<__m128>(x),
                              std::bit_cast<__m128>(y), 25)};
    else if constexpr (sizeof(T) == 4)
      return {_mm_cmp_epi32_mask(std::bit_cast<__m128i>(x),
                                 std::bit_cast<__m128i>(y), 1)};
    else static_assert(false);
  } else if constexpr (W == 2) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm_cmp_pd_mask(std::bit_cast<__m128>(x),
                              std::bit_cast<__m128d>(y), 25)};
    else if constexpr (sizeof(T) == 8)
      return {_mm_cmp_epi64_mask(std::bit_cast<__m128i>(x),
                                 std::bit_cast<__m128i>(y), 1)};
    else static_assert(false);
  } else static_assert(false);
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gt(Vec<W, T> x, Vec<W, T> y) -> mask::Bit<W> {
  if constexpr (W == 16) {
    if constexpr (std::same_as<T, float>) // UQ (unordered quiet?)
      return {_mm512_cmp_ps_mask(std::bit_cast<__m512>(x),
                                 std::bit_cast<__m512>(y), 22)};
    else if constexpr (sizeof(T) == 4)
      return {_mm512_cmp_epi32_mask(std::bit_cast<__m512i>(x),
                                    std::bit_cast<__m512i>(y), 6)};
    else static_assert(false);
  } else if constexpr (W == 8) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm512_cmp_pd_mask(std::bit_cast<__m512d>(x),
                                 std::bit_cast<__m512d>(y), 22)};
    else if constexpr (sizeof(T) == 8)
      return {_mm512_cmp_epi64_mask(std::bit_cast<__m512i>(x),
                                    std::bit_cast<__m512i>(y), 6)};
    else if constexpr (std::same_as<T, float>) // UQ (unordered quiet?)
      return {_mm256_cmp_ps_mask(std::bit_cast<__m256>(x),
                                 std::bit_cast<__m256>(y), 22)};
    else if constexpr (sizeof(T) == 4)
      return {_mm256_cmp_epi32_mask(std::bit_cast<__m256i>(x),
                                    std::bit_cast<__m256i>(y), 6)};
    else static_assert(false);
  } else if constexpr (W == 4) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm256_cmp_pd_mask(std::bit_cast<__m256d>(x),
                                 std::bit_cast<__m256d>(y), 22)};
    else if constexpr (sizeof(T) == 8)
      return {_mm256_cmp_epi64_mask(std::bit_cast<__m256i>(x),
                                    std::bit_cast<__m256i>(y), 6)};
    else if constexpr (std::same_as<T, float>) // UQ (unordered quiet?)
      return {_mm_cmp_ps_mask(std::bit_cast<__m128>(x),
                              std::bit_cast<__m128>(y), 22)};
    else if constexpr (sizeof(T) == 4)
      return {_mm_cmp_epi32_mask(std::bit_cast<__m128i>(x),
                                 std::bit_cast<__m128i>(y), 6)};
    else static_assert(false);
  } else if constexpr (W == 2) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm_cmp_pd_mask(std::bit_cast<__m128d>(x),
                              std::bit_cast<__m128d>(y), 22)};
    else if constexpr (sizeof(T) == 8)
      return {_mm_cmp_epi64_mask(std::bit_cast<__m128i>(x),
                                 std::bit_cast<__m128i>(y), 6)};
    else static_assert(false);
  } else static_assert(false);
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
le(Vec<W, T> x, Vec<W, T> y) -> mask::Bit<W> {
  if constexpr (W == 16) {
    if constexpr (std::same_as<T, float>) // UQ (unordered quiet?)
      return {_mm512_cmp_ps_mask(std::bit_cast<__m512>(x),
                                 std::bit_cast<__m512>(y), 26)};
    else if constexpr (sizeof(T) == 4)
      return {_mm512_cmp_epi32_mask(std::bit_cast<__m512i>(x),
                                    std::bit_cast<__m512i>(y), 2)};
    else static_assert(false);
  } else if constexpr (W == 8) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm512_cmp_pd_mask(std::bit_cast<__m512d>(x),
                                 std::bit_cast<__m512d>(y), 26)};
    else if constexpr (sizeof(T) == 8)
      return {_mm512_cmp_epi64_mask(std::bit_cast<__m512i>(x),
                                    std::bit_cast<__m512i>(y), 2)};
    else if constexpr (std::same_as<T, float>) // UQ (unordered quiet?)
      return {_mm256_cmp_ps_mask(std::bit_cast<__m256>(x),
                                 std::bit_cast<__m256>(y), 26)};
    else if constexpr (sizeof(T) == 4)
      return {_mm256_cmp_epi32_mask(std::bit_cast<__m256i>(x),
                                    std::bit_cast<__m256i>(y), 2)};
    else static_assert(false);
  } else if constexpr (W == 4) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm256_cmp_pd_mask(std::bit_cast<__m256d>(x),
                                 std::bit_cast<__m256d>(y), 26)};
    else if constexpr (sizeof(T) == 8)
      return {_mm256_cmp_epi64_mask(std::bit_cast<__m256i>(x),
                                    std::bit_cast<__m256i>(y), 2)};
    else if constexpr (std::same_as<T, float>) // UQ (unordered quiet?)
      return {_mm_cmp_ps_mask(std::bit_cast<__m128>(x),
                              std::bit_cast<__m128>(y), 26)};
    else if constexpr (sizeof(T) == 4)
      return {_mm_cmp_epi32_mask(std::bit_cast<__m128i>(x),
                                 std::bit_cast<__m128i>(y), 2)};
    else static_assert(false);
  } else if constexpr (W == 2) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm_cmp_pd_mask(std::bit_cast<__m128d>(x),
                              std::bit_cast<__m128d>(y), 26)};
    else if constexpr (sizeof(T) == 8)
      return {_mm_cmp_epi64_mask(std::bit_cast<__m128i>(x),
                                 std::bit_cast<__m128i>(y), 2)};
    else static_assert(false);
  } else static_assert(false);
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
ge(Vec<W, T> x, Vec<W, T> y) -> mask::Bit<W> {
  if constexpr (W == 16) {
    if constexpr (std::same_as<T, float>) // UQ (unordered quiet?)
      return {_mm512_cmp_ps_mask(std::bit_cast<__m512>(x),
                                 std::bit_cast<__m512>(y), 21)};
    else if constexpr (sizeof(T) == 4)
      return {_mm512_cmp_epi32_mask(std::bit_cast<__m512i>(x),
                                    std::bit_cast<__m512i>(y), 5)};
    else static_assert(false);
  } else if constexpr (W == 8) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm512_cmp_pd_mask(std::bit_cast<__m512d>(x),
                                 std::bit_cast<__m512d>(y), 21)};
    else if constexpr (sizeof(T) == 8)
      return {_mm512_cmp_epi64_mask(std::bit_cast<__m512i>(x),
                                    std::bit_cast<__m512i>(y), 5)};
    else if constexpr (std::same_as<T, float>) // UQ (unordered quiet?)
      return {_mm256_cmp_ps_mask(std::bit_cast<__m256>(x),
                                 std::bit_cast<__m256>(y), 21)};
    else if constexpr (sizeof(T) == 4)
      return {_mm256_cmp_epi32_mask(std::bit_cast<__m256i>(x),
                                    std::bit_cast<__m256i>(y), 5)};
    else static_assert(false);
  } else if constexpr (W == 4) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm256_cmp_pd_mask(std::bit_cast<__m256d>(x),
                                 std::bit_cast<__m256d>(y), 21)};
    else if constexpr (sizeof(T) == 8)
      return {_mm256_cmp_epi64_mask(std::bit_cast<__m256i>(x),
                                    std::bit_cast<__m256i>(y), 5)};
    else if constexpr (std::same_as<T, float>) // UQ (unordered quiet?)
      return {_mm_cmp_ps_mask(std::bit_cast<__m128>(x),
                              std::bit_cast<__m128>(y), 21)};
    else if constexpr (sizeof(T) == 4)
      return {_mm_cmp_epi32_mask(std::bit_cast<__m128i>(x),
                                 std::bit_cast<__m128i>(y), 5)};
    else static_assert(false);
  } else if constexpr (W == 2) {

    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm_cmp_pd_mask(std::bit_cast<__m128d>(x),
                              std::bit_cast<__m128d>(y), 21)};
    else if constexpr (sizeof(T) == 8)
      return {_mm_cmp_epi64_mask(std::bit_cast<__m128i>(x),
                                 std::bit_cast<__m128i>(y), 5)};
    else static_assert(false);
  } else static_assert(false);
}

#elif defined(__AVX512F__)

template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto eq(Vec<W, T> x,
                                                       Vec<W, T> y) {
  if constexpr (W == 8) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm512_cmp_pd_mask(std::bit_cast<__m512d>(x),
                                 std::bit_cast<__m512d>(y), 8)};
    else if constexpr (sizeof(T) == 8)
      return {_mm512_cmp_epi64_mask(std::bit_cast<__m512i>(x),
                                    std::bit_cast<__m512i>(y), 0)};
    else static_assert(false);
  } else {
    return mask::Vector<W>{x == y};
  }
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto ne(Vec<W, T> x,
                                                       Vec<W, T> y) {
  if constexpr (W == 8) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm512_cmp_pd_mask(std::bit_cast<__m512d>(x),
                                 std::bit_cast<__m512d>(y), 4)};
    else if constexpr (sizeof(T) == 8)
      return {_mm512_cmp_epi64_mask(std::bit_cast<__m512i>(x),
                                    std::bit_cast<__m512i>(y), 4)};
    else static_assert(false);
  } else {
    return mask::Vector<W>{x != y};
  }
}

template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto lt(Vec<W, T> x,
                                                       Vec<W, T> y) {
  if constexpr (W == 8) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm512_cmp_pd_mask(std::bit_cast<__m512d>(x),
                                 std::bit_cast<__m512d>(y), 25)};
    else if constexpr (sizeof(T) == 8)
      return {_mm512_cmp_epi64_mask(std::bit_cast<__m512i>(x),
                                    std::bit_cast<__m512i>(y), 1)};
    else static_assert(false);
  } else {
    return mask::Vector<W>{x < y};
  }
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto gt(Vec<W, T> x,
                                                       Vec<W, T> y) {
  if constexpr (W == 8) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm512_cmp_pd_mask(std::bit_cast<__m512d>(x),
                                 std::bit_cast<__m512d>(y), 22)};
    else if constexpr (sizeof(T) == 8)
      return {_mm512_cmp_epi64_mask(std::bit_cast<__m512i>(x),
                                    std::bit_cast<__m512i>(y), 6)};
    else static_assert(false);
  } else {
    return mask::Vector<W>{x > y};
  }
}

template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto le(Vec<W, T> x,
                                                       Vec<W, T> y) {
  if constexpr (W == 8) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm512_cmp_pd_mask(std::bit_cast<__m512d>(x),
                                 std::bit_cast<__m512d>(y), 26)};
    else if constexpr (sizeof(T) == 8)
      return {_mm512_cmp_epi64_mask(std::bit_cast<__m512i>(x),
                                    std::bit_cast<__m512i>(y), 2)};
    else static_assert(false);
  } else {
    return mask::Vector<W>{x <= y};
  }
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto ge(Vec<W, T> x,
                                                       Vec<W, T> y) {
  if constexpr (W == 8) {
    if constexpr (std::same_as<T, double>) // UQ (unordered quiet?)
      return {_mm512_cmp_pd_mask(std::bit_cast<__m512d>(x),
                                 std::bit_cast<__m512d>(y), 21)};
    else if constexpr (sizeof(T) == 8)
      return {_mm512_cmp_epi64_mask(std::bit_cast<__m512i>(x),
                                    std::bit_cast<__m512i>(y), 5)};
    else static_assert(false);
  } else {
    return mask::Vector<W>{x >= y};
  }
}

#else  // ifdef __AVX512VL__

template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
eq(Vec<W, T> x, Vec<W, T> y) -> mask::Vector<W, sizeof(T)> {
  return {x == y};
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
ne(Vec<W, T> x, Vec<W, T> y) -> mask::Vector<W, sizeof(T)> {
  return {x != y};
}

template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
lt(Vec<W, T> x, Vec<W, T> y) -> mask::Vector<W, sizeof(T)> {
  return {x < y};
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
gt(Vec<W, T> x, Vec<W, T> y) -> mask::Vector<W, sizeof(T)> {
  return {x > y};
}

template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
le(Vec<W, T> x, Vec<W, T> y) -> mask::Vector<W, sizeof(T)> {
  return {x <= y};
}
template <ptrdiff_t W, typename T>
[[gnu::always_inline, gnu::artificial]] inline auto
ge(Vec<W, T> x, Vec<W, T> y) -> mask::Vector<W, sizeof(T)> {
  return {x >= y};
}
#endif // ifdef __AVX512VL__; else
} // namespace cmp
template <ptrdiff_t W,
          typename I = std::conditional_t<W == 2, int64_t, int32_t>>
[[gnu::always_inline]] inline auto firstoff() {
  return cmp::ne<W, I>(range<W, I>(), Vec<W, I>{});
}

} // namespace simd
