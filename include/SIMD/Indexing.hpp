#pragma once
#ifndef MATH_SIMD_INDEXING_HPP_INCLUDED
#define MATH_SIMD_INDEXING_HPP_INCLUDED

#include "Math/AxisTypes.hpp"
#include "SIMD/Masks.hpp"
namespace poly::simd::index {
// Unroll rows by a factor of `R` and cols by `C`, vectorizing with width `W`
template <ptrdiff_t U, ptrdiff_t W = 1, typename M = mask::None<W>>
struct Unroll {
  ptrdiff_t index;
  [[no_unique_address]] M mask{};
  explicit constexpr operator ptrdiff_t() const { return index; }
  explicit constexpr operator bool() const { return bool(mask); }
  constexpr auto operator+(ptrdiff_t b) -> Unroll { return {b + index, mask}; }
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

/// UnrollDims<R,C,W,M,Transposed=false>
/// Transposed means that the `W` dim indexes across the stride
/// !Tranposed means that the `W` dim is contiguous.
/// Note that `UnrollDims<R,C,1,mask::None<1>,false>` is thus morally
/// equivalent to `UnrollDims<C,R,1,mask::None<1>,true>`
///
template <ptrdiff_t R, ptrdiff_t C = 1, ptrdiff_t W = 1,
          typename M = mask::None<W>, bool Transposed = false, ptrdiff_t X = -1>
struct UnrollDims {
  static_assert(W != 1 || std::same_as<M, mask::None<1>>,
                "Only mask vector dims");
  static_assert(W != 1 || !Transposed,
                "Canonicalize scalar with Tranpose=false");
  [[no_unique_address]] M mask;
  [[no_unique_address]] math::RowStride<X> rs;
};

template <typename T> static constexpr bool issimd = false;

template <ptrdiff_t U, ptrdiff_t W, typename M>
static constexpr bool issimd<Unroll<U, W, M>> = true;
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename M, bool Transposed,
          ptrdiff_t X>
static constexpr bool issimd<UnrollDims<R, C, W, M, Transposed, X>> = true;
} // namespace poly::simd::index

#endif
