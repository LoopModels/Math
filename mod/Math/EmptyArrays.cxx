#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifndef USE_MODULE
#include <cstddef>

#include "Math/ArrayConcepts.cxx"
#include "Math/AxisTypes.cxx"
#else
export module EmptyMatrix;

import ArrayConcepts;
import AxisTypes;
import STL;
#endif

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif

template <typename T> struct EmptyMatrix {
  using value_type = T;
  static constexpr auto begin() -> T * { return nullptr; }
  static constexpr auto end() -> T * { return nullptr; }

  static constexpr auto numRow() -> Row<0> { return {}; }
  static constexpr auto numCol() -> Col<0> { return {}; }
  static constexpr auto rowStride() -> RowStride<0> { return {}; }
  static constexpr auto getConstCol() -> ptrdiff_t { return 0; }

  static constexpr auto data() -> T * { return nullptr; }
  static constexpr auto operator[](ptrdiff_t, ptrdiff_t) -> T { return 0; }
  static constexpr auto shape() -> CartesianIndex<ptrdiff_t, ptrdiff_t> {
    return {0, 0};
  }
  static constexpr auto view() -> EmptyMatrix<T> { return EmptyMatrix<T>{}; }
  static constexpr auto dim() -> SquareDims<0> { return {numRow()}; }
};

static_assert(AbstractMatrix<EmptyMatrix<ptrdiff_t>>);

// template <typename T>
// constexpr auto matmul(EmptyMatrix<T>, PtrMatrix<T>) -> EmptyMatrix<T> {
//   return EmptyMatrix<T>{};
// }
// template <typename T>
// constexpr auto matmul(PtrMatrix<T>, EmptyMatrix<T>) -> EmptyMatrix<T> {
//   return EmptyMatrix<T>{};
// }

template <typename T> struct EmptyVector {
  static constexpr auto size() -> ptrdiff_t { return 0; };
  static constexpr auto begin() -> T * { return nullptr; }
  static constexpr auto end() -> T * { return nullptr; }
};
} // namespace math
