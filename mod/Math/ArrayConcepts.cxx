#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifndef USE_MODULE
#include "Math/AxisTypes.cxx"
#include "Math/MatrixDimensions.cxx"
#include <concepts>
#include <cstddef>
#else
export module ArrayConcepts;

export import MatDim;
import AxisTypes;
import STL;
#endif

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif

template <typename T, typename S = utils::eltype_t<T>>
concept LinearlyIndexable = requires(T t, ptrdiff_t i) {
  { t[i] } -> std::convertible_to<S>;
};
template <typename T, typename S = utils::eltype_t<T>>
concept CartesianIndexable = requires(T t, ptrdiff_t i) {
  { t[i, i] } -> std::convertible_to<S>;
};
// The `OrConvertible` concepts are for expression templates
template <typename T, typename S>
concept LinearlyIndexableOrConvertible =
  LinearlyIndexable<T, S> || std::convertible_to<T, S>;
template <typename T, typename S>
concept CartesianIndexableOrConvertible =
  CartesianIndexable<T, S> || std::convertible_to<T, S>;

template <typename T>
concept DefinesSize = requires(T t) {
  { t.size() } -> std::convertible_to<ptrdiff_t>;
};

template <typename T>
concept DefinesShape = requires(T t) {
  { t.numRow() } -> std::convertible_to<Row<>>;
  { t.numCol() } -> std::convertible_to<Col<>>;
};

template <typename T>
concept ShapelessSize = DefinesSize<T> && (!DefinesShape<T>);
template <typename T>
concept PrimitiveScalar = std::integral<T> || std::floating_point<T>;

template <typename T> constexpr auto numRows(const T &A) {
  if constexpr (DefinesShape<T>) return A.numRow();
  else return Row<1>{};
}
template <typename T> constexpr auto numCols(const T &A) {
  if constexpr (DefinesShape<T>) return A.numCol();
  else if constexpr (ShapelessSize<T>) return col(A.size());
  else return Col<1>{};
}
constexpr auto shape(const auto &x) {
  return CartesianIndex(unwrapRow(numRows(x)), unwrapCol(numCols(x)));
}

template <typename T>
concept LinearlyIndexableTensor =
  DefinesSize<T> && utils::HasEltype<T> && LinearlyIndexable<T>;

template <typename T>
concept RowVectorCore = LinearlyIndexableTensor<T> && requires(T t) {
  { numRows(t) } -> std::same_as<Row<1>>;
};
template <typename T>
concept ColVectorCore =
  LinearlyIndexableTensor<T> && !RowVectorCore<T> && requires(T t) {
    { numCols(t) } -> std::same_as<Col<1>>;
  };
template <typename T>
concept RowVector = RowVectorCore<T> && requires(T t) {
  { t.view() } -> RowVectorCore;
};
template <typename T>
concept ColVector = ColVectorCore<T> && requires(T t) {
  { t.view() } -> ColVectorCore;
};
template <typename T>
concept AbstractVector = RowVector<T> || ColVector<T>;

template <typename T>
concept AbstractMatrixCore =
  DefinesShape<T> && utils::HasEltype<T> && CartesianIndexable<T>;
template <typename T>
concept AbstractMatrix =
  AbstractMatrixCore<T> && !AbstractVector<T> && requires(T t) {
    { t.view() } -> AbstractMatrixCore;
  };

template <typename T>
concept AbstractTensor = AbstractVector<T> || AbstractMatrix<T>;

template <typename T>
concept HasDataPtr = requires(T t) {
  { t.data() } -> std::same_as<utils::eltype_t<T> *>;
};
template <typename T>
concept DenseTensor = AbstractTensor<T> && requires(T t) {
  { t.begin() } -> std::convertible_to<const utils::eltype_t<T> *>;
};

template <ptrdiff_t M> constexpr auto transpose_dim(Col<M> c) {
  if constexpr (M == -1) return row(ptrdiff_t(c));
  else return Row<M>{};
}
template <ptrdiff_t M> constexpr auto transpose_dim(Row<M> r) {
  if constexpr (M == -1) return col(ptrdiff_t(r));
  else return Col<M>{};
}

template <typename T, typename U> constexpr auto reinterpret(U x) {
  if constexpr (std::same_as<T, U>) return x;
  else return x.template reinterpretImpl<T>();
}

} // namespace math
