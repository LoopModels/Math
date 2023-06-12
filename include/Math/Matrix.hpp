#pragma once
#include "Math/AxisTypes.hpp"
#include "Math/MatrixDimensions.hpp"
#include "Utilities/TypePromotion.hpp"
#include <concepts>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace poly::math {
template <typename T>
concept AbstractMatrixCore = utils::HasEltype<T> && requires(T t, ptrdiff_t i) {
  { t(i, i) } -> std::convertible_to<utils::eltype_t<T>>;
  { t.numRow() } -> SameOrBroadcast<Row>;
  { t.numCol() } -> SameOrBroadcast<Col>;
  { t.size() } -> SameOrBroadcast<CartesianIndex<Row, Col>>;
  { t.dim() } -> std::convertible_to<StridedDims>;
  // {
  //     std::remove_reference_t<T>::canResize
  //     } -> std::same_as<const bool &>;
  // {t.extendOrAssertSize(i, i)};
};
template <typename T>
concept AbstractMatrix = AbstractMatrixCore<T> && requires(T t, ptrdiff_t i) {
  { t.view() } -> AbstractMatrixCore;
};
template <typename T>
concept HasDataPtr = requires(T t) {
  { t.data() } -> std::same_as<utils::eltype_t<T> *>;
};
template <typename T>
concept DataMatrix = AbstractMatrix<T> && HasDataPtr<T>;
template <typename T>
concept TemplateMatrix = AbstractMatrix<T> && (!HasDataPtr<T>);

template <typename T>
concept AbstractRowMajorMatrix = AbstractMatrix<T> && requires(T t) {
  { t.rowStride() } -> std::same_as<RowStride>;
};

template <typename A> struct Transpose {
  static_assert(AbstractMatrix<A>, "Argument to transpose is not a matrix.");
  static_assert(std::is_trivially_copyable_v<A>,
                "Argument to transpose is not trivially copyable.");

  using value_type = utils::eltype_t<A>;
  [[no_unique_address]] A a;
  constexpr auto operator()(ptrdiff_t i, ptrdiff_t j) const { return a(j, i); }
  [[nodiscard]] constexpr auto numRow() const -> Row {
    return Row{ptrdiff_t{a.numCol()}};
  }
  [[nodiscard]] constexpr auto numCol() const -> Col {
    return Col{ptrdiff_t{a.numRow()}};
  }
  [[nodiscard]] constexpr auto view() const -> auto & { return *this; };
  [[nodiscard]] constexpr auto size() const -> CartesianIndex<Row, Col> {
    return {numRow(), numCol()};
  }
  [[nodiscard]] constexpr auto dim() const -> DenseDims {
    return {numRow(), numCol()};
  }
  constexpr Transpose(A b) : a(b) {}
};
template <typename A> Transpose(A) -> Transpose<A>;

} // namespace poly::math
