#pragma once

#include "Math/AxisTypes.hpp"
#include <cstddef>
#include <cstdint>

namespace poly::math {
template <ptrdiff_t R = -1> struct SquareDims;
template <ptrdiff_t R = -1, ptrdiff_t C = -1> struct DenseDims;
template <ptrdiff_t R = -1, ptrdiff_t C = -1, ptrdiff_t X = -1>
struct StridedDims;

template <class R, class C> struct CartesianIndex {
  [[no_unique_address]] R row;
  [[no_unique_address]] C col;
  explicit constexpr operator Row<>() const { return {row}; }
  explicit constexpr operator Col<>() const { return {col}; }
  constexpr auto operator==(const CartesianIndex &) const -> bool = default;
  constexpr operator SquareDims<>()
  requires(std::same_as<R, ptrdiff_t> && std::same_as<C, ptrdiff_t>);
  constexpr operator DenseDims<>()
  requires(std::same_as<R, ptrdiff_t> && std::same_as<C, ptrdiff_t>);
  constexpr operator StridedDims<>()
  requires(std::same_as<R, ptrdiff_t> && std::same_as<C, ptrdiff_t>);
};
template <class R, class C> CartesianIndex(R, C) -> CartesianIndex<R, C>;

template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t X> struct StridedDims {
  [[no_unique_address]] Row<R> M{};
  [[no_unique_address]] Col<C> N{};
  [[no_unique_address]] RowStride<X> strideM{};
  // constexpr StridedDims() = default;
  // constexpr StridedDims(Row<R> m, Col<C> n)
  //   : M{m}, N{n}, strideM{ptrdiff_t(n)} {}
  // constexpr StridedDims(Row<R> m, Col<C> n, RowStride<X> x)
  //   : M{m}, N{n}, strideM{x} {
  //   invariant(N <= strideM);
  // }
  constexpr explicit operator int32_t() const {
    return int32_t(ptrdiff_t(M) * ptrdiff_t(strideM));
  }
  constexpr explicit operator int64_t() const {
    return int64_t(ptrdiff_t(M) * ptrdiff_t(strideM));
  }
  constexpr explicit operator uint32_t() const {
    return uint32_t(ptrdiff_t(M) * ptrdiff_t(strideM));
  }
  constexpr explicit operator uint64_t() const {
    return uint64_t(ptrdiff_t(M) * ptrdiff_t(strideM));
  }
  constexpr auto operator=(DenseDims<R, C> D) -> StridedDims &requires(C == X);
  constexpr auto operator=(SquareDims<R> D)
    -> StridedDims &requires((R == C) && (C == X));
  constexpr explicit operator Row<R>() const { return M; }
  constexpr explicit operator Col<C>() const {
    invariant(N <= strideM);
    return N;
  }
  constexpr explicit operator RowStride<X>() const {
    invariant(N <= strideM);
    return strideM;
  }
  [[nodiscard]] constexpr auto operator==(const StridedDims &D) const -> bool {
    invariant(N <= strideM);
    return (M == D.M) && (N == D.N) && (strideM == D.strideM);
  }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto truncate(Row<S> r) const
    -> StridedDims<S, C, X> {
    invariant(r <= M);
    return similar(r);
  }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto truncate(Col<S> c) const
    -> StridedDims<R, S, X> {
    invariant(c <= N);
    return similar(c);
  }
  constexpr auto set(Row<> r) -> StridedDims &
  requires(R == -1)
  {
    invariant(N <= strideM);
    M = r;
    return *this;
  }
  constexpr auto set(Col<> c) -> StridedDims &
  requires(C == -1)
  {
    N = c;
    strideM = RowStride{std::max<ptrdiff_t>(strideM, N)};
    return *this;
  }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto similar(Row<S> r) const -> StridedDims {
    invariant(N <= strideM);
    return {r, N, strideM};
  }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto similar(Col<S> c) const -> StridedDims {
    invariant(N <= strideM);
    invariant(c <= Col{strideM});
    return {M, c, strideM};
  }
  constexpr operator StridedDims<-1, -1, -1>() const
  requires((R != -1) || (C != -1) || (X != -1))
  {
    return {M, N, strideM};
  }
  friend inline auto operator<<(std::ostream &os, StridedDims x)
    -> std::ostream & {
    return os << x.M << " x " << x.N << " (stride " << x.strideM << ")";
  }
};
static_assert(sizeof(StridedDims<-1, 8, 8>) == sizeof(ptrdiff_t));
template <ptrdiff_t R, ptrdiff_t C> struct DenseDims {
  [[no_unique_address]] Row<R> M{};
  [[no_unique_address]] Col<C> N{};
  constexpr explicit operator int32_t() const {
    return int32_t(ptrdiff_t(M) * ptrdiff_t(N));
  }
  constexpr explicit operator int64_t() const {
    return int64_t(ptrdiff_t(M) * ptrdiff_t(N));
  }
  constexpr explicit operator uint32_t() const {
    return uint32_t(ptrdiff_t(M) * ptrdiff_t(N));
  }
  constexpr explicit operator uint64_t() const {
    return uint64_t(ptrdiff_t(M) * ptrdiff_t(N));
  }
  // constexpr DenseDims() = default;
  // constexpr DenseDims(Row<R> m, Col<C> n) : M(unsigned(m)), N(unsigned(n)) {}
  // template <ptrdiff_t X>
  // constexpr explicit DenseDims(StridedDims<R, C, X> d) : M(d.M), N(d.N) {}
  // constexpr DenseDims(CartesianIndex<R, C> ind)
  //   : M(unsigned(ind.row)), N(unsigned(ind.col)) {}
  constexpr auto operator=(SquareDims<R> D) -> DenseDims &requires(R == C);
  constexpr explicit operator Row<R>() const { return M; }
  constexpr explicit operator Col<C>() const { return N; }
  constexpr explicit operator RowStride<C>() const { return {ptrdiff_t{N}}; }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto truncate(Row<S> r) const -> DenseDims {
    invariant(r <= Row{M});
    return similar(r);
  }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto truncate(Col<S> c) const
    -> StridedDims<R, S, C> {
    invariant(c <= Col{M});
    return {M, c, {ptrdiff_t(N)}};
  }
  constexpr auto set(Row<> r) -> DenseDims &
  requires(R == -1)
  {
    M = r;
    return *this;
  }
  constexpr auto set(Col<> c) -> DenseDims &
  requires(C == -1)
  {
    N = c;
    return *this;
  }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto similar(Row<S> r) const -> DenseDims {
    return {r, N};
  }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto similar(Col<S> c) const -> DenseDims {
    return {M, c};
  }
  constexpr operator StridedDims<R, C, C>() const
  requires((R != -1) || (C != -1))
  {
    return {M, N, N};
  }
  constexpr operator StridedDims<>() const { return {M, N, {ptrdiff_t(N)}}; }
  constexpr operator DenseDims<>() const
  requires((R != -1) || (C != -1))
  {
    return {M, N};
  }
  friend inline auto operator<<(std::ostream &os, DenseDims x)
    -> std::ostream & {
    return os << x.M << " x " << x.N;
  }
};
template <ptrdiff_t R> struct SquareDims {
  [[no_unique_address]] Row<R> M{};
  constexpr explicit operator int32_t() const {
    return int32_t(ptrdiff_t(M) * ptrdiff_t(M));
  }
  constexpr explicit operator int64_t() const {
    return int64_t(ptrdiff_t(M) * ptrdiff_t(M));
  }
  constexpr explicit operator uint32_t() const {
    return uint32_t(ptrdiff_t(M) * ptrdiff_t(M));
  }
  constexpr explicit operator uint64_t() const {
    return uint64_t(ptrdiff_t(M) * ptrdiff_t(M));
  }
  // constexpr SquareDims() = default;
  // constexpr SquareDims(ptrdiff_t d) : M{d} {}
  // constexpr SquareDims(Row<R> d) : M{d} {}
  // constexpr SquareDims(Col<R> d) : M{ptrdiff_t(d)} {}
  // constexpr SquareDims(CartesianIndex<R, R> ind) : M(ind.row) {
  //   invariant(ptrdiff_t(ind.row), ptrdiff_t(ind.col));
  // }
  constexpr explicit operator Row<R>() const { return M; }
  constexpr explicit operator Col<R>() const { return {ptrdiff_t(M)}; }
  constexpr explicit operator RowStride<R>() const { return {ptrdiff_t(M)}; }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto truncate(Row<S> r) const -> DenseDims<S, R> {
    invariant(r <= Row{M});
    return {r, {ptrdiff_t(M)}};
  }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto truncate(Col<S> c) const
    -> StridedDims<R, S, R> {
    invariant(c <= Col{M});
    return {M, unsigned(c), {ptrdiff_t(M)}};
  }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto similar(Row<S> r) const -> DenseDims<S, R> {
    return {r, {ptrdiff_t(M)}};
  }
  template <ptrdiff_t S>
  [[nodiscard]] constexpr auto similar(Col<S> c) const -> DenseDims<R, S> {
    return {M, c};
  }
  constexpr operator StridedDims<R, R, R>() const
  requires(R != -1)
  {
    return {M, {ptrdiff_t(M)}, {ptrdiff_t(M)}};
  }
  constexpr operator StridedDims<>() const {
    return {M, {ptrdiff_t(M)}, {ptrdiff_t(M)}};
  }
  constexpr operator DenseDims<R, R>() const
  requires(R != -1)
  {
    return {M, {ptrdiff_t(M)}};
  }
  constexpr operator DenseDims<>() const { return {M, {ptrdiff_t(M)}}; }
  constexpr operator SquareDims<>() const
  requires((R != -1))
  {
    return {M};
  }
  friend inline auto operator<<(std::ostream &os, SquareDims x)
    -> std::ostream & {
    return os << x.M << " x " << x.M;
  }
};
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
    M = D.M;
    N = D.N;
    strideM = N;
    return *this;
  };
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t X>
constexpr inline auto StridedDims<R, C, X>::operator=(SquareDims<R> D)
  -> StridedDims &requires((R == C) && (C == X)) {
    M = D.M;
    N = M;
    strideM = M;
    return *this;
  };
template <ptrdiff_t R, ptrdiff_t C>
constexpr inline auto DenseDims<R, C>::operator=(SquareDims<R> D)
  -> DenseDims &requires(R == C) {
    M = D.M;
    N = M;
    return *this;
  };

template <class R, class C>
constexpr inline CartesianIndex<R, C>::operator SquareDims<>()
requires(std::same_as<R, ptrdiff_t> && std::same_as<C, ptrdiff_t>)
{
  invariant(row, col);
  return SquareDims{Row<>{row}};
}
template <class R, class C>
constexpr inline CartesianIndex<R, C>::operator DenseDims<>()
requires(std::same_as<R, ptrdiff_t> && std::same_as<C, ptrdiff_t>)
{
  return DenseDims{Row<>{row}, Col<>{col}};
}
template <class R, class C>
constexpr inline CartesianIndex<R, C>::operator StridedDims<>()
requires(std::same_as<R, ptrdiff_t> && std::same_as<C, ptrdiff_t>)
{
  return StridedDims{Row<>{row}, Col<>{col}, RowStride<>{col}};
}

template <typename D>
concept MatrixDimension = requires(D d) {
  { d } -> std::convertible_to<StridedDims<-1, -1, -1>>;
};
static_assert(MatrixDimension<SquareDims<>>);
static_assert(MatrixDimension<DenseDims<>>);
static_assert(MatrixDimension<StridedDims<>>);
static_assert(MatrixDimension<SquareDims<8>>);
static_assert(MatrixDimension<DenseDims<8, 8>>);
static_assert(MatrixDimension<StridedDims<8, 8, 16>>);
static_assert(!MatrixDimension<unsigned>);

static_assert(std::convertible_to<const DenseDims<8, 8>, DenseDims<>>);

template <typename T, typename S>
concept PromoteDimTo = (!std::same_as<T, S>)&&std::convertible_to<T, S>;
template <typename T, typename S>
concept PromoteDimFrom = (!std::same_as<T, S>)&&std::convertible_to<S, T>;

constexpr auto row(MatrixDimension auto s) { return Row(s); }
constexpr auto col(MatrixDimension auto s) { return Col(s); }
constexpr auto rowStride(MatrixDimension auto s) { return RowStride(s); }

} // namespace poly::math
