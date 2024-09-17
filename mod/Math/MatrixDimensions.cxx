#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifndef USE_MODULE
#include "Math/AxisTypes.cxx"
#include "Utilities/Invariant.cxx"
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <ostream>
#include <type_traits>
#else
export module MatDim;

import AxisTypes;
import Invariant;
import STL;
#endif

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif

using utils::invariant;

template <ptrdiff_t R = -1> struct SquareDims;
template <ptrdiff_t R = -1, ptrdiff_t C = -1> struct DenseDims;
template <ptrdiff_t R = -1, ptrdiff_t C = -1, ptrdiff_t X = -1>
struct StridedDims;

template <class R, class C> struct CartesianIndex {
  [[no_unique_address]] R row_idx_;
  [[no_unique_address]] C col_idx_;
  explicit constexpr operator Row<>() const { return row(row_idx_); }
  explicit constexpr operator Col<>() const { return col(col_idx_); }
  constexpr auto operator==(const CartesianIndex &) const -> bool = default;
  constexpr operator CartesianIndex<ptrdiff_t, ptrdiff_t>() const
  requires(std::convertible_to<R, ptrdiff_t> &&
           std::convertible_to<C, ptrdiff_t> &&
           (!(std::same_as<R, ptrdiff_t> && std::same_as<C, ptrdiff_t>)))
  {
    invariant(row_idx_ >= 0);
    invariant(col_idx_ >= 0);
    return {.row_idx_ = ptrdiff_t(row_idx_), .col_idx_ = ptrdiff_t(col_idx_)};
  }
  // FIXME: Do we need this??
  // Either document why, or delete this method.
  constexpr operator ptrdiff_t() const
  requires(std::same_as<R, std::integral_constant<ptrdiff_t, 1>>)
  {
    invariant(col_idx_ >= 0);
    return col_idx_;
  }
  constexpr operator Length<>() const
  requires(std::same_as<R, std::integral_constant<ptrdiff_t, 1>>)
  {
    invariant(col_idx_ >= 0);
    return length(col_idx_);
  }
  constexpr operator SquareDims<>() const
  requires(std::convertible_to<R, ptrdiff_t> &&
           std::convertible_to<C, ptrdiff_t>);
  constexpr operator DenseDims<>() const
  requires(std::convertible_to<R, ptrdiff_t> &&
           std::convertible_to<C, ptrdiff_t>);
  constexpr operator StridedDims<>() const
  requires(std::convertible_to<R, ptrdiff_t> &&
           std::convertible_to<C, ptrdiff_t>);
};
template <class R, class C> CartesianIndex(R, C) -> CartesianIndex<R, C>;

template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t X> struct StridedDims {
  static constexpr ptrdiff_t nrow = R;
  static constexpr ptrdiff_t ncol = C;
  static constexpr ptrdiff_t nstride = X;
  [[no_unique_address]] Row<R> m_;
  [[no_unique_address]] Col<C> n_;
  [[no_unique_address]] RowStride<X> stride_m_;
  explicit constexpr StridedDims() = default;
  constexpr StridedDims(Row<R> m, Col<C> n)
    : m_{m}, n_{n}, stride_m_{stride(ptrdiff_t(n))} {}
  constexpr StridedDims(Row<R> m, Col<C> n, RowStride<X> x)
    : m_{m}, n_{n}, stride_m_{x} {
    invariant(n_ <= stride_m_);
  }
  template <ptrdiff_t N>
  constexpr StridedDims(Length<N>)
    : m_{Row<R>(row(std::integral_constant<ptrdiff_t, 1>{}))},
      n_{Col<C>(col(std::integral_constant<ptrdiff_t, N>{}))},
      stride_m_{RowStride<X>(stride(std::integral_constant<ptrdiff_t, N>{}))} {}

  template <ptrdiff_t A, ptrdiff_t B, ptrdiff_t S>
  constexpr StridedDims(StridedDims<A, B, S> other)
  requires((R != -1 && A == -1) || (C != -1 && B == -1) || (X != -1 && S == -1))
    : m_{}, n_{}, stride_m_{} {
    if constexpr (R != -1) utils::invariant(row(other) == R);
    else m_ = other.m_;
    if constexpr (C != -1) utils::invariant(col(other) == C);
    else n_ = other.n_;
    if constexpr (X != -1) utils::invariant(stride(other) == X);
    else stride_m_ = other.stride_m_;
  }
  constexpr explicit operator int() const {
    return int(ptrdiff_t(m_) * ptrdiff_t(stride_m_));
  }
  constexpr explicit operator long() const {
    return long(ptrdiff_t(m_) * ptrdiff_t(stride_m_));
  }
  constexpr explicit operator long long() const {
    return (long long)(ptrdiff_t(m_) * ptrdiff_t(stride_m_));
  }
  constexpr explicit operator unsigned int() const {
    return (unsigned int)(ptrdiff_t(m_) * ptrdiff_t(stride_m_));
  }
  constexpr explicit operator unsigned long() const {
    return (unsigned long)(ptrdiff_t(m_) * ptrdiff_t(stride_m_));
  }
  constexpr explicit operator unsigned long long() const {
    return (unsigned long long)(ptrdiff_t(m_) * ptrdiff_t(stride_m_));
  }
  constexpr auto operator=(DenseDims<R, C> D) -> StridedDims &requires(C == X);
  constexpr auto operator=(SquareDims<R> D) -> StridedDims &requires((R == C) &&
                                                                     (C == X));
  [[nodiscard]] constexpr auto operator==(const StridedDims &D) const -> bool {
    invariant(n_ <= stride_m_);
    return (m_ == D.m_) && (n_ == D.n_) && (stride_m_ == D.stride_m_);
  }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto
  truncate(Row<S> r) const -> StridedDims<S, C, X> {
    invariant(r <= m_);
    return similar(r);
  }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto
  truncate(Col<S> c) const -> StridedDims<R, S, X> {
    invariant(c <= n_);
    return similar(c);
  }
  constexpr auto set(Row<> r) -> StridedDims &
  requires(R == -1)
  {
    invariant(n_ <= stride_m_);
    m_ = r;
    return *this;
  }
  constexpr auto set(Col<> c) -> StridedDims &
  requires(C == -1)
  {
    n_ = c;
    stride_m_ = stride(std::max(ptrdiff_t(stride_m_), ptrdiff_t(n_)));
    return *this;
  }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto similar(Row<S> r) const -> StridedDims {
    invariant(n_ <= stride_m_);
    return {r, n_, stride_m_};
  }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto similar(Col<S> c) const -> StridedDims {
    invariant(n_ <= stride_m_);
    invariant(c <= Col{stride_m_});
    return {m_, c, stride_m_};
  }
  constexpr operator StridedDims<-1, -1, -1>() const
  requires((R != -1) || (C != -1) || (X != -1))
  {
    return {m_, n_, stride_m_};
  }
  constexpr explicit operator Row<R>() const { return m_; }
  constexpr explicit operator Col<C>() const {
    invariant(n_ <= stride_m_);
    return n_;
  }
  constexpr explicit operator RowStride<X>() const {
    invariant(n_ <= stride_m_);
    return stride_m_;
  }

private:
  friend constexpr auto row(StridedDims d) -> Row<R> { return d.m_; }
  friend constexpr auto col(StridedDims d) -> Col<C> {
    invariant(d.n_ <= d.stride_m_);
    return d.n_;
  }
  friend constexpr auto stride(StridedDims d) -> RowStride<X> {
    invariant(d.n_ <= d.stride_m_);
    return d.stride_m_;
  }
  friend auto operator<<(std::ostream &os, StridedDims x) -> std::ostream & {
    return os << x.m_ << " x " << x.n_ << " (stride " << x.stride_m_ << ")";
  }
}; // namespace math
static_assert(sizeof(StridedDims<-1, 8, 8>) == sizeof(ptrdiff_t));

template <ptrdiff_t R, ptrdiff_t C> struct DenseDims {
  static constexpr ptrdiff_t nrow = R;
  static constexpr ptrdiff_t ncol = C;
  static constexpr ptrdiff_t nstride = C;
  [[no_unique_address]] Row<R> m_;
  [[no_unique_address]] Col<C> n_;
  explicit constexpr DenseDims() = default;
  constexpr DenseDims(Row<R> m, Col<C> n) : m_{m}, n_{n} {}
  constexpr DenseDims(Length<1>)
    : m_{Row<R>(row(std::integral_constant<ptrdiff_t, 1>{}))},
      n_{Col<C>(col(std::integral_constant<ptrdiff_t, 1>{}))} {}
  template <ptrdiff_t A, ptrdiff_t B>
  constexpr DenseDims(DenseDims<A, B> other)
  requires((R != -1 && A == -1) || (C != -1 && B == -1))
    : m_{}, n_{} {
    if constexpr (R != -1) utils::invariant(row(other) == R);
    else m_ = other.m_;
    if constexpr (C != -1) utils::invariant(col(other) == C);
    else n_ = other.n_;
  }
  constexpr explicit operator int() const {
    return int(ptrdiff_t(m_) * ptrdiff_t(n_));
  }
  constexpr explicit operator long() const {
    return long(ptrdiff_t(m_) * ptrdiff_t(n_));
  }
  constexpr explicit operator long long() const {
    return (long long)(ptrdiff_t(m_) * ptrdiff_t(n_));
  }
  constexpr explicit operator unsigned int() const {
    return (unsigned int)(ptrdiff_t(m_) * ptrdiff_t(n_));
  }
  constexpr explicit operator unsigned long() const {
    return (unsigned long)(ptrdiff_t(m_) * ptrdiff_t(n_));
  }
  constexpr explicit operator unsigned long long() const {
    return (unsigned long long)(ptrdiff_t(m_) * ptrdiff_t(n_));
  }
  // constexpr DenseDims() = default;
  // constexpr DenseDims(Row<R> m, Col<C> n) : M(unsigned(m)), N(unsigned(n)) {}
  // template <ptrdiff_t X>
  // constexpr explicit DenseDims(StridedDims<R, C, X> d) : M(d.M), N(d.N) {}
  // constexpr DenseDims(CartesianIndex<R, C> ind)
  //   : M(unsigned(ind.row)), N(unsigned(ind.col)) {}
  constexpr auto operator=(SquareDims<R> D) -> DenseDims &requires(R == C);
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto truncate(Row<S> r) const -> DenseDims {
    invariant(r <= Row{m_});
    return similar(r);
  }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto
  truncate(Col<S> c) const -> StridedDims<R, S, C> {
    invariant(c <= Col{m_});
    return {m_, c, {ptrdiff_t(n_)}};
  }
  constexpr auto set(Row<> r) -> DenseDims &
  requires(R == -1)
  {
    m_ = r;
    return *this;
  }
  constexpr auto set(Col<> c) -> DenseDims &
  requires(C == -1)
  {
    n_ = c;
    return *this;
  }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto similar(Row<S> r) const -> DenseDims {
    return {r, n_};
  }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto similar(Col<S> c) const -> DenseDims {
    return {m_, c};
  }
  constexpr operator StridedDims<R, C, C>() const
  requires((R != -1) || (C != -1))
  {
    return {m_, n_, n_};
  }
  constexpr operator StridedDims<>() const {
    return {m_, n_, stride(ptrdiff_t(n_))};
  }
  constexpr operator DenseDims<>() const
  requires((R != -1) || (C != -1))
  {
    return {m_, n_};
  }
  constexpr operator ptrdiff_t() const
  requires(R == 1)
  {
    return ptrdiff_t(n_);
  }
  constexpr operator std::integral_constant<ptrdiff_t, C>() const
  requires((R == 1) && (C != -1))
  {
    return {};
  }
  constexpr explicit operator Row<R>() const { return m_; }
  constexpr explicit operator Col<C>() const { return n_; }
  constexpr explicit operator RowStride<C>() const {
    if constexpr (C == -1) return stride(ptrdiff_t{n_});
    else return {};
  }

  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  flat() const -> Length<(R == -1) || (C == -1) ? -1 : R * C> {
    if constexpr ((R == -1) || (C == -1))
      return length(ptrdiff_t(m_) * ptrdiff_t(n_));
    else return {};
  }

private:
  friend constexpr auto row(DenseDims d) -> Row<R> { return d.m_; }
  friend constexpr auto col(DenseDims d) -> Col<C> { return d.n_; }
  friend constexpr auto stride(DenseDims d) -> RowStride<C> {
    if constexpr (C == -1) return stride(ptrdiff_t{d.n_});
    else return {};
  }

  friend auto operator<<(std::ostream &os, DenseDims x) -> std::ostream & {
    return os << x.m_ << " x " << x.n_;
  }
};
static_assert(std::is_trivially_default_constructible_v<DenseDims<>>);

template <ptrdiff_t R> struct SquareDims {
  static constexpr ptrdiff_t nrow = R;
  static constexpr ptrdiff_t ncol = R;
  static constexpr ptrdiff_t nstride = R;
  [[no_unique_address]] Row<R> m_;
  constexpr explicit operator int() const {
    return int(ptrdiff_t(m_) * ptrdiff_t(m_));
  }
  constexpr explicit operator long() const {
    return long(ptrdiff_t(m_) * ptrdiff_t(m_));
  }
  constexpr explicit operator long long() const {
    return (long long)(ptrdiff_t(m_) * ptrdiff_t(m_));
  }
  constexpr explicit operator unsigned int() const {
    return (unsigned int)(ptrdiff_t(m_) * ptrdiff_t(m_));
  }
  constexpr explicit operator unsigned long() const {
    return (unsigned long)(ptrdiff_t(m_) * ptrdiff_t(m_));
  }
  constexpr explicit operator unsigned long long() const {
    return (unsigned long long)(ptrdiff_t(m_) * ptrdiff_t(m_));
  }
  // constexpr SquareDims() = default;
  // constexpr SquareDims(ptrdiff_t d) : M{d} {}
  // constexpr SquareDims(Row<R> d) : M{d} {}
  // constexpr SquareDims(Col<R> d) : M{ptrdiff_t(d)} {}
  // constexpr SquareDims(CartesianIndex<R, R> ind) : M(ind.row) {
  //   invariant(ptrdiff_t(ind.row), ptrdiff_t(ind.col));
  // }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto truncate(Row<S> r) const -> DenseDims<S, R> {
    invariant(r <= Row{m_});
    return {r, {ptrdiff_t(m_)}};
  }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto
  truncate(Col<S> c) const -> StridedDims<R, S, R> {
    invariant(c <= Col{m_});
    return {m_, unsigned(c), {ptrdiff_t(m_)}};
  }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto similar(Row<S> r) const -> DenseDims<S, R> {
    return {r, {ptrdiff_t(m_)}};
  }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto similar(Col<S> c) const -> DenseDims<R, S> {
    return {m_, c};
  }
  constexpr operator StridedDims<R, R, R>() const
  requires(R != -1)
  {
    return {m_, col(ptrdiff_t(m_)), stride(ptrdiff_t(m_))};
  }
  constexpr operator StridedDims<>() const {
    return {m_, col(ptrdiff_t(m_)), stride(ptrdiff_t(m_))};
  }
  constexpr operator DenseDims<R, R>() const
  requires(R != -1)
  {
    return {m_, {ptrdiff_t(m_)}};
  }
  constexpr operator DenseDims<>() const { return {m_, ascol(m_)}; }
  constexpr operator SquareDims<>() const
  requires((R != -1))
  {
    return {m_};
  }
  constexpr explicit operator Row<R>() const { return m_; }
  constexpr explicit operator Col<R>() const { return col(ptrdiff_t(m_)); }
  constexpr explicit operator RowStride<R>() const {
    if constexpr (R == -1) return stride(ptrdiff_t(m_));
    else return {};
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  flat() const -> Length<R == -1 ? -1 : R * R> {
    if constexpr (R == -1) {
      ptrdiff_t m = ptrdiff_t(m_);
      return length(m * m);
    } else return {};
  }

private:
  friend constexpr auto row(SquareDims d) -> Row<R> { return d.m_; }
  friend constexpr auto col(SquareDims d) -> Col<R> {
    return col(ptrdiff_t(d.m_));
  }
  friend constexpr auto stride(SquareDims d) -> RowStride<R> {
    if constexpr (R == -1) return stride(ptrdiff_t(d.m_));
    else return {};
  }
  friend auto operator<<(std::ostream &os, SquareDims x) -> std::ostream & {
    return os << x.m_ << " x " << x.m_;
  }
};

static_assert(std::is_convertible_v<StridedDims<>, StridedDims<2>>);

template <ptrdiff_t R> Row(SquareDims<R>) -> Row<R>;
template <ptrdiff_t R> Col(SquareDims<R>) -> Col<R>;
template <ptrdiff_t R> RowStride(SquareDims<R>) -> RowStride<R>;
template <ptrdiff_t R, ptrdiff_t C> Row(DenseDims<R, C>) -> Row<R>;
template <ptrdiff_t R, ptrdiff_t C> Col(DenseDims<R, C>) -> Col<C>;
template <ptrdiff_t R, ptrdiff_t C> RowStride(DenseDims<R, C>) -> RowStride<R>;
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t X>
Row(StridedDims<R, C, X>) -> Row<R>;
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t X>
Col(StridedDims<R, C, X>) -> Col<C>;
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t X>
RowStride(StridedDims<R, C, X>) -> RowStride<X>;

template <ptrdiff_t R> SquareDims(Row<R>) -> SquareDims<R>;
template <ptrdiff_t R, ptrdiff_t C>
DenseDims(Row<R>, Col<C>) -> DenseDims<R, C>;
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t X>
StridedDims(Row<R>, Col<C>, RowStride<X>) -> StridedDims<R, C, X>;
// [[nodiscard]] constexpr auto capacity(std::integral auto c) { return c; }
// [[nodiscard]] constexpr auto capacity(auto c) -> unsigned int { return c; }
// [[nodiscard]] constexpr auto capacity(CapDims c) -> unsigned int {
//   return c.rowCapacity * c.strideM;
// }

template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t X>
constexpr inline auto StridedDims<R, C, X>::operator=(DenseDims<R, C> D)
  -> StridedDims &requires(C == X) {
  m_ = D.M;
  n_ = D.N;
  stride_m_ = n_;
  return *this;
};
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t X>
constexpr inline auto StridedDims<R, C, X>::operator=(SquareDims<R> D)
  -> StridedDims &requires((R == C) && (C == X)) {
  m_ = D.M;
  n_ = m_;
  stride_m_ = m_;
  return *this;
};
template <ptrdiff_t R, ptrdiff_t C>
constexpr inline auto
DenseDims<R, C>::operator=(SquareDims<R> D) -> DenseDims &requires(R == C) {
  m_ = D.M;
  n_ = m_;
  return *this;
};

template <class R, class C>
constexpr inline CartesianIndex<R, C>::operator SquareDims<>() const
requires(std::convertible_to<R, ptrdiff_t> && std::convertible_to<C, ptrdiff_t>)
{
  invariant(row_idx_, col_idx_);
  return SquareDims{row(row_idx_)};
}
template <class R, class C>
constexpr inline CartesianIndex<R, C>::operator DenseDims<>() const
requires(std::convertible_to<R, ptrdiff_t> && std::convertible_to<C, ptrdiff_t>)
{
  return DenseDims{row(row_idx_), col(col_idx_)};
}
template <class R, class C>
constexpr inline CartesianIndex<R, C>::operator StridedDims<>() const
requires(std::convertible_to<R, ptrdiff_t> && std::convertible_to<C, ptrdiff_t>)
{
  return StridedDims{row(row_idx_), col(col_idx_), stride(col_idx_)};
}

template <typename T, typename S>
concept different = !std::same_as<T, S>;

template <typename D>
concept MatrixDimension = requires(D d) {
  { d } -> std::convertible_to<StridedDims<-1, -1, -1>>;
  { row(d) } -> different<Row<1>>;
  { col(d) } -> different<Col<1>>;
};
static_assert(MatrixDimension<SquareDims<>>);
static_assert(MatrixDimension<DenseDims<>>);
static_assert(!MatrixDimension<DenseDims<1>>);
static_assert(!MatrixDimension<DenseDims<-1, 1>>);
static_assert(MatrixDimension<StridedDims<>>);
static_assert(MatrixDimension<SquareDims<8>>);
static_assert(MatrixDimension<DenseDims<8, 8>>);
static_assert(MatrixDimension<StridedDims<8, 8, 16>>);
static_assert(!MatrixDimension<unsigned>);

static_assert(std::convertible_to<const DenseDims<8, 8>, DenseDims<>>);

template <typename T, typename S>
concept PromoteDimTo = (!std::same_as<T, S>) && std::convertible_to<T, S>;
template <typename T, typename S>
concept PromoteDimFrom = (!std::same_as<T, S>) && std::convertible_to<S, T>;

template <typename T>
concept HasInnerReduction = bool(T::has_reduction_loop);

template <typename T>
concept RowVectorDimension = requires(T t) {
  { Length(t) } -> std::same_as<T>;
};
static_assert(RowVectorDimension<Length<3>>);
static_assert(RowVectorDimension<Length<>>);
static_assert(!RowVectorDimension<ptrdiff_t>);
template <typename D>
concept ColVectorDimension =
  std::same_as<decltype(Col(std::declval<D>())), Col<1>>;
template <typename D>
concept VectorDimension = RowVectorDimension<D> || ColVectorDimension<D>;

template <typename S>
concept Dimension = VectorDimension<S> != MatrixDimension<S>;

template <typename T>
concept StaticInt =
  std::is_same_v<T, std::integral_constant<typename T::value_type, T::value>>;

} // namespace math

#ifdef USE_MODULE
export namespace utils {
#else
namespace utils {
#endif
template <typename T>
concept HasEltype = requires(T) {
  typename T::value_type;
  // std::is_scalar_v<typename std::remove_reference_t<T>::value_type>;
};

namespace detail {
template <typename A> struct GetEltype {
  // static_assert(!HasEltype<A>);
  using value_type = A;
};
template <utils::HasEltype A> struct GetEltype<A> {
  using value_type = typename A::value_type;
};
} // namespace detail

template <typename T>
using eltype_t =
  typename detail::GetEltype<std::remove_reference_t<T>>::value_type;

template <class T, class C>
concept ElementOf = std::convertible_to<T, eltype_t<C>>;
} // namespace utils
template <typename A, typename B> struct PromoteEltype {

  using elta = utils::eltype_t<A>;
  using eltb = utils::eltype_t<B>;
  using value_type =
    std::conditional_t<std::convertible_to<A, eltb>, eltb,
                       std::conditional_t<std::convertible_to<B, elta>, elta,
                                          std::common_type_t<elta, eltb>>>;
};
