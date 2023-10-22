#pragma once
#include "Utilities/Invariant.hpp"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>

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
};
static_assert(sizeof(Row<>) == sizeof(ptrdiff_t));
template <ptrdiff_t M> constexpr Row<M>::operator Row<-1>() const {
  return {M};
}
template <ptrdiff_t M>
constexpr auto operator==(ptrdiff_t x, Row<M> y) -> bool {
  return x == ptrdiff_t(y);
}
template <ptrdiff_t M>
constexpr auto operator==(Row<M> y, ptrdiff_t x) -> bool {
  return x == ptrdiff_t(y);
}
template <ptrdiff_t M, ptrdiff_t N>
constexpr auto operator==(Row<M> x, Row<N> y) -> bool {
  return ptrdiff_t(x) == ptrdiff_t(y);
}
template <ptrdiff_t M>
constexpr auto operator<=>(ptrdiff_t x, Row<M> y) -> std::strong_ordering {
  return x <=> ptrdiff_t(y);
}
template <ptrdiff_t M>
constexpr auto operator<=>(Row<M> y, ptrdiff_t x) -> std::strong_ordering {
  return x <=> ptrdiff_t(y);
}
template <ptrdiff_t M, ptrdiff_t N>
constexpr auto operator<=>(Row<M> x, Row<N> y) -> std::strong_ordering {
  return ptrdiff_t(x) <=> ptrdiff_t(y);
}
template <ptrdiff_t M>
inline auto operator<<(std::ostream &os, Row<M> x) -> std::ostream & {
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
};
static_assert(sizeof(Col<>) == sizeof(ptrdiff_t));
template <ptrdiff_t M> constexpr Col<M>::operator Col<-1>() const {
  return {M};
}
template <ptrdiff_t M>
constexpr auto operator==(ptrdiff_t x, Col<M> y) -> bool {
  return x == ptrdiff_t(y);
}
template <ptrdiff_t M>
constexpr auto operator==(Col<M> y, ptrdiff_t x) -> bool {
  return x == ptrdiff_t(y);
}
template <ptrdiff_t M, ptrdiff_t N>
constexpr auto operator==(Col<M> x, Col<N> y) -> bool {
  return ptrdiff_t(x) == ptrdiff_t(y);
}
template <ptrdiff_t M>
constexpr auto operator<=>(ptrdiff_t x, Col<M> y) -> std::strong_ordering {
  return x <=> ptrdiff_t(y);
}
template <ptrdiff_t M>
constexpr auto operator<=>(Col<M> y, ptrdiff_t x) -> std::strong_ordering {
  return x <=> ptrdiff_t(y);
}
template <ptrdiff_t M, ptrdiff_t N>
constexpr auto operator<=>(Col<M> x, Col<N> y) -> std::strong_ordering {
  return ptrdiff_t(x) <=> ptrdiff_t(y);
}
template <ptrdiff_t M>
inline auto operator<<(std::ostream &os, Col<M> x) -> std::ostream & {
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
template <ptrdiff_t M>
constexpr auto operator==(ptrdiff_t x, RowStride<M> y) -> bool {
  return x == ptrdiff_t(y);
}
template <ptrdiff_t M>
constexpr auto operator==(RowStride<M> y, ptrdiff_t x) -> bool {
  return x == ptrdiff_t(y);
}
template <ptrdiff_t M, ptrdiff_t N>
constexpr auto operator==(RowStride<M> x, RowStride<N> y) -> bool {
  return ptrdiff_t(x) == ptrdiff_t(y);
}
template <ptrdiff_t M>
constexpr auto operator<=>(ptrdiff_t x, RowStride<M> y)
  -> std::strong_ordering {
  return x <=> ptrdiff_t(y);
}
template <ptrdiff_t M>
constexpr auto operator<=>(RowStride<M> y, ptrdiff_t x)
  -> std::strong_ordering {
  return x <=> ptrdiff_t(y);
}
template <ptrdiff_t M, ptrdiff_t N>
constexpr auto operator<=>(RowStride<M> x, RowStride<N> y)
  -> std::strong_ordering {
  return ptrdiff_t(x) <=> ptrdiff_t(y);
}
template <ptrdiff_t M>
inline auto operator<<(std::ostream &os, RowStride<M> x) -> std::ostream & {
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

} // namespace poly::math
