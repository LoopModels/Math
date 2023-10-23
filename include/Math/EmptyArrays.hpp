#pragma once
#include "Math/Matrix.hpp"
#include "Math/MatrixDimensions.hpp"
#include <cstddef>

namespace poly::math {

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
  static constexpr auto size() -> CartesianIndex<ptrdiff_t, ptrdiff_t> {
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
} // namespace poly::math
