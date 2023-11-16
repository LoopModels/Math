#pragma once
#include "Math/AxisTypes.hpp"
#include "Math/MatrixDimensions.hpp"
#include "Utilities/TypePromotion.hpp"
#include <cstddef>
#include <type_traits>

namespace poly::math {

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

constexpr auto numRows(const DefinesShape auto &A) { return A.numRow(); }
constexpr auto numCols(const DefinesShape auto &A) { return A.numCol(); }

constexpr auto numRows(const ShapelessSize auto &) -> Row<1> { return {}; }
constexpr auto numCols(const ShapelessSize auto &A) { return col(A.size()); }

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
concept ColVectorCore = LinearlyIndexableTensor<T> && requires(T t) {
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
  if constexpr (M == -1) return Row<>{ptrdiff_t(c)};
  else return Row<M>{};
}
template <ptrdiff_t M> constexpr auto transpose_dim(Row<M> r) {
  if constexpr (M == -1) return Col<>{ptrdiff_t(r)};
  else return Col<M>{};
}

template <typename A> struct Transpose {
  static_assert(AbstractTensor<A>,
                "Argument to transpose is not a matrix or vector.");
  static_assert(std::is_trivially_copyable_v<A>,
                "Argument to transpose is not trivially copyable.");

  using value_type = utils::eltype_t<A>;
  static constexpr bool has_reduction_loop = HasInnerReduction<A>;
  [[no_unique_address]] A a;
  constexpr auto operator[](auto i) const
  requires(AbstractVector<A>)
  {
    return a[i];
  }
  constexpr auto operator[](auto i, auto j) const
  requires(AbstractMatrix<A>)
  {
    return a[j, i];
  }
  [[nodiscard]] constexpr auto numRow() const {
    return transpose_dim(a.numCol());
  }
  [[nodiscard]] constexpr auto numCol() const {
    return transpose_dim(a.numRow());
  }
  [[nodiscard]] constexpr auto view() const -> auto & { return *this; };
  [[nodiscard]] constexpr auto size() const { return a.size(); }
  [[nodiscard]] constexpr auto dim() const {
    return DenseDims(numRow(), numCol());
  }
  constexpr Transpose(A b) : a(b) {}
  constexpr auto t() const -> A { return a; }
  constexpr auto operator<<(const auto &b) -> Transpose<A> & {
    a << transpose(b);
    return *this;
  }
  constexpr auto operator+=(const auto &b) -> Transpose<A> & {
    a += transpose(b);
    return *this;
  }
  constexpr auto operator-=(const auto &b) -> Transpose<A> & {
    a -= transpose(b);
    return *this;
  }
  constexpr auto operator*=(const auto &b) -> Transpose<A> & {
    a *= transpose(b);
    return *this;
  }
  constexpr auto operator/=(const auto &b) -> Transpose<A> & {
    a /= transpose(b);
    return *this;
  }
};
template <typename A> Transpose(A) -> Transpose<A>;

template <typename T> constexpr auto transpose(const T &a) {
  if constexpr (requires(T t) {
                  { t.t() } -> AbstractTensor;
                })
    return a.t();
  else return Transpose{view(a)};
}

} // namespace poly::math
