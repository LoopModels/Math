#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifndef USE_MODULE
#include <concepts>
#include <cstddef>

#include "Math/AxisTypes.cxx"
#include "SIMD/Masks.cxx"
#else
export module SIMD:Index;

import :Mask;
import AxisTypes;
import STL;
#endif

#ifdef USE_MODULE
export namespace simd::index {
#else
namespace simd::index {
#endif

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
  [[no_unique_address]] M mask_;
  [[no_unique_address]] math::RowStride<X> rs_;
};

template <typename T> inline constexpr bool issimd = false;

template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename M, bool Transposed,
          ptrdiff_t X>
inline constexpr bool issimd<UnrollDims<R, C, W, M, Transposed, X>> = true;
} // namespace simd::index
