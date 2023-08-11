#pragma once

#include "Math/MatrixDimensions.hpp"
#include "Utilities/TypePromotion.hpp"
#include <eve/module/core.hpp>

namespace poly::math::simd {

template <typename T, typename S>
inline constexpr auto vecWidth() -> ptrdiff_t {
  if constexpr (PrimitiveScalar<T>) {
    constexpr ptrdiff_t W = eve::wide<T>::size();
    if constexpr (StaticInt<S>) {
      if constexpr (S::value < W) {
        constexpr size_t L = S::value;
        return 1 << (8 * sizeof(size_t) - std::countl_zero(L - 1));
      }
    }
    return W;
  } else return 1;
}
} // namespace poly::math::simd
