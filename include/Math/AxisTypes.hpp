#pragma once
#include "Utilities/Invariant.hpp"
#include <cstddef>
#include <cstdint>
#include <iostream>

/// LinAlg
///
/// This is the namespace for all mathematical functions.
/// Semantics:
/// We generally around structs holding pointer/size/stride/capacity information
/// by value.
/// For assignments that copy the underlying data, use `<<`
/// E.g., `A << B + C;`
/// Updating assignment operators like `+=`, `-=`, and, `*=` are supported.
///
/// The choice of `<<` over `=` for copying data is so that `operator=`
/// can be the usual copy assignment operator, which is useful to use when
/// we pass arrays/pointers by value to functions that truncate their size.
///
/// Operations like `+` and `-` are elementwise, while `*` performs matrix
/// multiplication. All operations are lazy, building up expression templates
/// that are evaluated upon assignments, e.g. `<<` or `+=`.
///
/// All the PtrVector/PtrMatrix types are trivially destructible, copyable, etc
/// Their lifetimes are governed by the Arena or RAII type used to back
/// them.
namespace poly::math {

using utils::invariant;
template <ptrdiff_t M = -1> struct Row {
  static_assert(M >= 0);
  explicit constexpr operator ptrdiff_t() const { return M; }
  explicit constexpr operator bool() const { return M; }
  constexpr operator Row<-1>() const;
};
template <> struct Row<-1> {
  [[no_unique_address]] ptrdiff_t M;
  explicit constexpr operator ptrdiff_t() const {
    invariant(M >= 0);
    return M;
  }
  explicit constexpr operator bool() const { return M; }
  constexpr auto operator++() -> Row & {
    ++M;
    return *this;
  }
  constexpr auto operator--() -> Row & {
    --M;
    return *this;
  }
  constexpr auto operator++(int) -> Row {
    Row tmp{*this};
    ++M;
    return tmp;
  }
  constexpr auto operator--(int) -> Row {
    Row tmp{*this};
    --M;
    return tmp;
  }
};
static_assert(sizeof(Row<>) == sizeof(ptrdiff_t));
template <ptrdiff_t M> constexpr Row<M>::operator Row<-1>() const {
  return {M};
}
constexpr auto operator==(ptrdiff_t x, Row<> y) -> bool {
  return x == ptrdiff_t(y);
}
constexpr auto operator==(Row<> y, ptrdiff_t x) -> bool {
  return x == ptrdiff_t(y);
}
constexpr auto operator==(Row<> x, Row<> y) -> bool {
  return ptrdiff_t(x) == ptrdiff_t(y);
}
constexpr auto operator<=>(ptrdiff_t x, Row<> y) -> std::strong_ordering {
  return x <=> ptrdiff_t(y);
}
constexpr auto operator<=>(Row<> x, ptrdiff_t y) -> std::strong_ordering {
  return ptrdiff_t(x) <=> y;
}
constexpr auto operator<=>(Row<> x, Row<> y) -> std::strong_ordering {
  return ptrdiff_t(x) <=> ptrdiff_t(y);
}
inline auto operator<<(std::ostream &os, Row<> x) -> std::ostream & {
  return os << "Rows{" << ptrdiff_t(x) << "}";
}
template <ptrdiff_t M = -1> struct Col {
  static_assert(M >= 0);
  explicit constexpr operator ptrdiff_t() const { return M; }
  explicit constexpr operator bool() const { return M; }
  constexpr operator Col<-1>() const;
};
template <> struct Col<-1> {
  [[no_unique_address]] ptrdiff_t M;
  explicit constexpr operator ptrdiff_t() const {
    invariant(M >= 0);
    return M;
  }
  explicit constexpr operator bool() const { return M; }
  constexpr auto operator++() -> Col & {
    ++M;
    return *this;
  }
  constexpr auto operator--() -> Col & {
    --M;
    return *this;
  }
  constexpr auto operator++(int) -> Col {
    Col tmp{*this};
    ++M;
    return tmp;
  }
  constexpr auto operator--(int) -> Col {
    Col tmp{*this};
    --M;
    return tmp;
  }
};
static_assert(sizeof(Col<>) == sizeof(ptrdiff_t));
template <ptrdiff_t M> constexpr Col<M>::operator Col<-1>() const {
  return {M};
}
constexpr auto operator==(ptrdiff_t x, Col<> y) -> bool {
  return x == ptrdiff_t(y);
}
constexpr auto operator==(Col<> y, ptrdiff_t x) -> bool {
  return x == ptrdiff_t(y);
}
constexpr auto operator==(Col<> x, Col<> y) -> bool {
  return ptrdiff_t(x) == ptrdiff_t(y);
}
constexpr auto operator<=>(ptrdiff_t x, Col<> y) -> std::strong_ordering {
  return x <=> ptrdiff_t(y);
}
constexpr auto operator<=>(Col<> x, ptrdiff_t y) -> std::strong_ordering {
  return ptrdiff_t(x) <=> y;
}
constexpr auto operator<=>(Col<> x, Col<> y) -> std::strong_ordering {
  return ptrdiff_t(x) <=> ptrdiff_t(y);
}
constexpr auto operator*(Row<> r, Col<> c) -> ptrdiff_t {
  return ptrdiff_t(r) * ptrdiff_t(c);
}
inline auto operator<<(std::ostream &os, Col<> x) -> std::ostream & {
  return os << "Rows{" << ptrdiff_t(x) << "}";
}
template <ptrdiff_t M = -1> struct RowStride {
  static_assert(M >= 0);
  explicit constexpr operator ptrdiff_t() const { return M; }
  explicit constexpr operator bool() const { return M; }
  constexpr operator RowStride<-1>() const;
};
template <> struct RowStride<-1> {
  [[no_unique_address]] ptrdiff_t M;
  explicit constexpr operator ptrdiff_t() const {
    invariant(M >= 0);
    return M;
  }
  explicit constexpr operator bool() const { return M; }
};
static_assert(sizeof(RowStride<>) == sizeof(ptrdiff_t));
template <ptrdiff_t M> constexpr RowStride<M>::operator RowStride<-1>() const {
  return {M};
}
constexpr auto operator==(ptrdiff_t x, RowStride<> y) -> bool {
  return x == ptrdiff_t(y);
}
constexpr auto operator==(RowStride<> y, ptrdiff_t x) -> bool {
  return x == ptrdiff_t(y);
}
constexpr auto operator==(RowStride<> x, RowStride<> y) -> bool {
  return ptrdiff_t(x) == ptrdiff_t(y);
}
constexpr auto operator<=>(ptrdiff_t x, RowStride<> y)
  -> std::strong_ordering {
  return x <=> ptrdiff_t(y);
}
constexpr auto operator<=>(RowStride<> x, ptrdiff_t y)
  -> std::strong_ordering {
  return ptrdiff_t(x) <=> y;
}
constexpr auto operator<=>(RowStride<> x, RowStride<> y)
  -> std::strong_ordering {
  return ptrdiff_t(x) <=> ptrdiff_t(y);
}
inline auto operator<<(std::ostream &os, RowStride<> x) -> std::ostream & {
  return os << "RowStrides{" << ptrdiff_t(x) << "}";
}

// constexpr auto max(Row M, Col N) -> ptrdiff_t {
//   return std::max(ptrdiff_t(M), ptrdiff_t(N));
// }
// constexpr auto max(Col N, RowStride X) -> RowStride {
//   return RowStride{std::max(ptrdiff_t(N), ptrdiff_t(X))};
// }
// constexpr auto min(Col N, Col X) -> Col {
//   return Col{std::max(Col::V(N), Col::V(X))};
// }
// constexpr auto min(Row N, Col X) -> ptrdiff_t {
//   return std::min(ptrdiff_t(N), ptrdiff_t(X));
// }

template <ptrdiff_t M> constexpr auto standardizeRangeBound(Row<M> x) {
  if constexpr (M == -1) return ptrdiff_t(x);
  else return std::integral_constant<ptrdiff_t, M>{};
}
template <ptrdiff_t M> constexpr auto standardizeRangeBound(Col<M> x) {
  if constexpr (M == -1) return ptrdiff_t(x);
  else return std::integral_constant<ptrdiff_t, M>{};
}

template <ptrdiff_t M> constexpr auto unwrapRow(Row<M> x) -> ptrdiff_t {
  return ptrdiff_t(x);
}
template <ptrdiff_t M> constexpr auto unwrapCol(Col<M> x) -> ptrdiff_t {
  return ptrdiff_t(x);
}
constexpr auto unwrapRow(auto x) { return x; }
constexpr auto unwrapCol(auto x) { return x; }

template <ptrdiff_t C, ptrdiff_t X>
constexpr auto operator==(Col<C> c, RowStride<X> x) -> bool {
  return ptrdiff_t(c) == ptrdiff_t(x);
}
template <ptrdiff_t C, ptrdiff_t X>
constexpr auto operator<=>(Col<C> c, RowStride<X> x) -> std::strong_ordering {
  return ptrdiff_t(c) <=> ptrdiff_t(x);
}

constexpr auto row(ptrdiff_t x) -> Row<> {
  invariant(x >= 0);
  return Row<-1>{x};
}
constexpr auto col(ptrdiff_t x) -> Col<> {
  invariant(x >= 0);
  return Col<-1>{x};
}
constexpr auto rowStride(ptrdiff_t x) -> RowStride<> {
  invariant(x >= 0);
  return RowStride<-1>{x};
}
template <std::integral I, I x>
constexpr auto row(std::integral_constant<I, x>) -> Row<ptrdiff_t(x)> {
  static_assert(x >= 0);
  return {};
}
template <std::integral I, I x>
constexpr auto col(std::integral_constant<I, x>) -> Col<ptrdiff_t(x)> {
  static_assert(x >= 0);
  return {};
}
template <std::integral I, I x>
constexpr auto rowStride(std::integral_constant<I, x>)
  -> RowStride<ptrdiff_t(x)> {
  static_assert(x >= 0);
  return {};
}
constexpr auto operator+(Row<> a, Row<> b) -> Row<> {
  return {ptrdiff_t(a) + ptrdiff_t(b)};
}
constexpr auto operator+(Col<> a, Col<> b) -> Col<> {
  return {ptrdiff_t(a) + ptrdiff_t(b)};
}
constexpr auto operator-(Row<> a, Row<> b) -> Row<> {
  return {ptrdiff_t(a) - ptrdiff_t(b)};
}
constexpr auto operator-(Col<> a, Col<> b) -> Col<> {
  return {ptrdiff_t(a) - ptrdiff_t(b)};
}

} // namespace poly::math
