#pragma once
#include "Math/AxisTypes.hpp"
#include "Math/MatrixDimensions.hpp"
#include "Math/Vector.hpp"
#include "Utilities/TypePromotion.hpp"
#include <cstddef>
#include <type_traits>

namespace poly::math {

template <typename T, typename S = utils::eltype_t<T>>
concept CartesianIndexable = requires(T t, ptrdiff_t i) {
  { t[i, i] } -> std::convertible_to<S>;
};
template <typename T, typename S>
concept CartesianIndexableOrConvertible =
  CartesianIndexable<T, S> || std::convertible_to<T, S>;

template <typename T>
concept AbstractMatrixCore =
  utils::HasEltype<T> && CartesianIndexable<T> && requires(T t) {
    { t.numRow() } -> std::convertible_to<Row<>>;
    { t.numCol() } -> std::convertible_to<Col<>>;
    { t.size() } -> std::same_as<CartesianIndex<ptrdiff_t, ptrdiff_t>>;
    { t.dim() } -> std::convertible_to<StridedDims<>>;
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
  { t.rowStride() } -> std::convertible_to<RowStride<>>;
};

template <ptrdiff_t M>
constexpr auto transpose_dim(Col<M> c){
    if constexpr (M==-1) return Row<>{ptrdiff_t(c)};
    else return Row<M>{};
}
template <ptrdiff_t M>
constexpr auto transpose_dim(Row<M> r){
    if constexpr (M==-1) return Col<>{ptrdiff_t(r)};
    else return Col<M>{};
}

template <typename A> struct Transpose {
  static_assert(AbstractMatrix<A> || AbstractVector<A>,
                "Argument to transpose is not a matrix or vector.");
  static_assert(std::is_trivially_copyable_v<A>,
                "Argument to transpose is not trivially copyable.");

  using value_type = utils::eltype_t<A>;
  [[no_unique_address]] A a;
  constexpr auto operator[](ptrdiff_t i, ptrdiff_t j) const -> value_type {
    if constexpr (AbstractMatrix<A>) return a[j, i];
    else {
      invariant(i == 0);
      return a[j];
    }
  }
  [[nodiscard]] constexpr auto numRow() const {
    if constexpr (AbstractMatrix<A>) return transpose_dim(a.numCol());
    else return Row<1>{};
  }
  [[nodiscard]] constexpr auto numCol() const {
    if constexpr (AbstractMatrix<A>) return transpose_dim(a.numRow());
    else return col(a.size());
  }
  [[nodiscard]] constexpr auto view() const -> auto & { return *this; };
  [[nodiscard]] constexpr auto size() const -> CartesianIndex<ptrdiff_t, ptrdiff_t> {
    return {ptrdiff_t(numRow()), ptrdiff_t(numCol())};
  }
  [[nodiscard]] constexpr auto dim() const {
    return DenseDims(numRow(), numCol());
  }
  constexpr Transpose(A b) : a(b) {}
  constexpr auto transpose() const -> A { return a; }
};
template <typename A> Transpose(A) -> Transpose<A>;

} // namespace poly::math
