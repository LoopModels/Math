#ifdef USE_MODULE
module;
#else
#pragma once
#endif
#ifndef USE_MODULE
#include <compare>
#include <concepts>
#include <cstddef>
#include <iostream>
#include <limits>
#include <type_traits>

#include "Utilities/Invariant.cxx"
#else
export module AxisTypes;

import Invariant;
import STL;
#endif

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
#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif

using utils::invariant;

template <ptrdiff_t M = -1, std::signed_integral I = ptrdiff_t> struct Length {
  static constexpr ptrdiff_t nrow = 1;
  static constexpr ptrdiff_t ncol = M;
  static constexpr ptrdiff_t nstride = M;
  static_assert(M >= 0);
  static_assert(M <= std::numeric_limits<I>::max());
  [[gnu::artificial, gnu::always_inline]] explicit inline constexpr
  operator I() const
  requires(!std::same_as<I, ptrdiff_t>)
  {
    return M;
  }
  [[gnu::artificial, gnu::always_inline]] explicit inline constexpr
  operator ptrdiff_t() const {
    return M;
  }
  [[gnu::artificial, gnu::always_inline]] explicit inline constexpr
  operator bool() const {
    return M;
  }
  [[gnu::artificial, gnu::always_inline]] static inline constexpr auto
  staticint() {
    return std::integral_constant<I, M>{};
  }

  [[gnu::artificial, gnu::always_inline]] inline constexpr
  operator Length<-1>() const;
  static constexpr auto comptime() -> ptrdiff_t { return M; }

  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  flat() const -> Length {
    return *this;
  }

private:
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(ptrdiff_t x, Length) -> bool {
    return x == M;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(Length, ptrdiff_t x) -> bool {
    return M == x;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(Length, Length) -> bool {
    return true;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(ptrdiff_t x, Length) -> std::strong_ordering {
    return x <=> M;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Length, ptrdiff_t y) -> std::strong_ordering {
    return M <=> y;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Length, Length) -> std::strong_ordering {
    return std::strong_ordering::equal;
  }
  template <ptrdiff_t N>
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator+(Length, Length<N>) -> Length<M + N>
  requires(N != -1)
  {
    return {};
  }
  template <ptrdiff_t N>
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator-(Length, Length<N>) -> Length<M - N>
  requires(N != -1)
  {
    static_assert(M >= N);
    return {};
  }
};
template <std::signed_integral I> struct Length<-1, I> {
  static constexpr ptrdiff_t nrow = 1;
  static constexpr ptrdiff_t ncol = -1;
  static constexpr ptrdiff_t nstride = -1;
  enum class len : I {};
  len value_;
  [[gnu::artificial, gnu::always_inline]] explicit inline constexpr
  operator I() const {
    auto m = static_cast<I>(value_);
    invariant(m >= I(0));
    return m;
  }
  [[gnu::artificial, gnu::always_inline]] explicit inline constexpr
  operator ptrdiff_t() const
  requires(!std::same_as<I, ptrdiff_t>)
  {
    auto m = static_cast<ptrdiff_t>(static_cast<I>(value_));
    invariant(m >= 0);
    return m;
  }
  [[gnu::artificial, gnu::always_inline]] explicit inline constexpr
  operator bool() const {
    return static_cast<ptrdiff_t>(value_);
  }

  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  operator++() -> Length & {
    value_ = static_cast<len>(static_cast<ptrdiff_t>(value_) + 1z);
    return *this;
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  operator--() -> Length & {
    value_ = static_cast<len>(static_cast<ptrdiff_t>(value_) - 1z);
    return *this;
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  operator++(int) -> Length {
    Length tmp{*this};
    value_ = static_cast<len>(static_cast<ptrdiff_t>(value_) + 1z);
    return tmp;
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  operator--(int) -> Length {
    Length tmp{*this};
    value_ = static_cast<len>(static_cast<ptrdiff_t>(value_) + 1z);
    return tmp;
  }

  [[gnu::artificial, gnu::always_inline]] inline constexpr
  operator Length<-1>() const
  requires(!std::same_as<I, ptrdiff_t>)
  {
    return Length<-1, ptrdiff_t>{
      static_cast<Length<-1, ptrdiff_t>::len>(ptrdiff_t(I(*this)))};
  }
  static constexpr auto comptime() -> ptrdiff_t { return -1; }

  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  flat() const -> Length {
    return *this;
  }

private:
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(ptrdiff_t x, Length y) -> bool {
    return x == ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(Length y, ptrdiff_t x) -> bool {
    return x == ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(Length x, Length y) -> bool {
    return ptrdiff_t(x) == ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(ptrdiff_t x, Length y) -> std::strong_ordering {
    return x <=> ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Length x, ptrdiff_t y) -> std::strong_ordering {
    return ptrdiff_t(x) <=> y;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Length x, Length y) -> std::strong_ordering {
    return ptrdiff_t(x) <=> ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator+(Length a, Length b) -> Length {
    return {static_cast<Length<-1>::len>(ptrdiff_t(a) + ptrdiff_t(b))};
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator-(Length a, Length b) -> Length {
    auto x = ptrdiff_t(a), y = ptrdiff_t(b);
    invariant(x >= y);
    return {static_cast<Length<-1>::len>(x - y)};
  }
};

// by default, we promote to `ptrdiff_t`; smaller sizes
// are primarilly in case we want smaller storage
template <ptrdiff_t M, std::signed_integral I>
[[gnu::artificial,
  gnu::always_inline]] inline constexpr Length<M, I>::operator Length<-1>()
  const {
  return {static_cast<Length<-1>::len>(M)};
}

template <ptrdiff_t M = -1, std::signed_integral I = ptrdiff_t>
struct Capacity {
  static_assert(M >= 0);
  static_assert(M <= std::numeric_limits<I>::max());
  [[gnu::artificial, gnu::always_inline]] inline explicit constexpr
  operator I() const {
    return M;
  }
  [[gnu::artificial, gnu::always_inline]] inline explicit constexpr
  operator bool() const {
    return M;
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr
  operator Capacity<-1, I>() const;
  static constexpr auto comptime() -> ptrdiff_t { return M; }

private:
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(ptrdiff_t x, Capacity) -> bool {
    return x == M;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(Capacity, ptrdiff_t x) -> bool {
    return M == x;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(Capacity, Capacity) -> bool {
    return true;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(ptrdiff_t x, Capacity) -> std::strong_ordering {
    return x <=> M;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Capacity, ptrdiff_t y) -> std::strong_ordering {
    return M <=> y;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Length<> x, Capacity) -> std::strong_ordering {
    return ptrdiff_t(x) <=> M;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Capacity, Length<> y) -> std::strong_ordering {
    return M <=> ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Capacity, Capacity) -> std::strong_ordering {
    return std::strong_ordering::equal;
  }
};
template <std::integral I> struct Capacity<-1, I> {
  enum class cap : I {};
  cap value_;
  [[gnu::artificial, gnu::always_inline]] inline explicit constexpr
  operator I() const {
    auto m = static_cast<I>(value_);
    invariant(m >= 0);
    return m;
  }
  [[gnu::artificial, gnu::always_inline]] inline explicit constexpr
  operator bool() const {
    return static_cast<I>(value_);
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  operator++() -> Capacity & {
    value_ = static_cast<cap>(static_cast<I>(value_) + 1z);
    return *this;
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  operator--() -> Capacity & {
    value_ = static_cast<cap>(static_cast<I>(value_) - 1z);
    return *this;
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  operator++(int) -> Capacity {
    Capacity tmp{*this};
    value_ = static_cast<cap>(static_cast<I>(value_) + 1z);
    return tmp;
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  operator--(int) -> Capacity {
    Capacity tmp{*this};
    value_ = static_cast<cap>(static_cast<I>(value_) + 1z);
    return tmp;
  }
  static constexpr auto comptime() -> ptrdiff_t { return -1; }

private:
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(ptrdiff_t x, Capacity y) -> bool {
    return x == ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(Capacity y, ptrdiff_t x) -> bool {
    return x == ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(Capacity x, Capacity y) -> bool {
    return ptrdiff_t(x) == ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(ptrdiff_t x, Capacity y) -> std::strong_ordering {
    return x <=> ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Capacity x, ptrdiff_t y) -> std::strong_ordering {
    return ptrdiff_t(x) <=> y;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Length<> x, Capacity y) -> std::strong_ordering {
    return ptrdiff_t(x) <=> ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Capacity x, Length<> y) -> std::strong_ordering {
    return ptrdiff_t(x) <=> ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Capacity x, Capacity y) -> std::strong_ordering {
    return ptrdiff_t(x) <=> ptrdiff_t(y);
  }
};
template <ptrdiff_t M, std::signed_integral I>
[[gnu::artificial, gnu::always_inline]] inline constexpr Capacity<
  M, I>::operator Capacity<-1, I>() const {
  static constexpr I cap = M;
  return {static_cast<Capacity<-1, I>::cap>(cap)};
}
template <ptrdiff_t M = -1> struct Row {
  static_assert(M >= 0);
  [[gnu::artificial, gnu::always_inline]] inline explicit constexpr
  operator ptrdiff_t() const {
    return M;
  }
  [[gnu::artificial, gnu::always_inline]] inline explicit constexpr
  operator bool() const {
    return M;
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr
  operator Row<-1>() const;
  static constexpr auto comptime() -> ptrdiff_t { return M; }

private:
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(ptrdiff_t x, Row) -> bool {
    return x == M;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(Row, ptrdiff_t x) -> bool {
    return M == x;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(Row, Row) -> bool {
    return true;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(ptrdiff_t x, Row) -> std::strong_ordering {
    return x <=> M;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Row, ptrdiff_t y) -> std::strong_ordering {
    return M <=> y;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Row, Row) -> std::strong_ordering {
    return std::strong_ordering::equal;
  }
  friend inline auto operator<<(std::ostream &os, Row) -> std::ostream & {
    return os << "Row<>{" << M << "}";
  }
  template <ptrdiff_t N>
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator+(Row, Row<N>) -> Row<M + N>
  requires(N != -1)
  {
    return {};
  }
  template <ptrdiff_t N>
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator-(Row, Row<N>) -> Row<M - N>
  requires(N != -1)
  {
    static_assert(M >= N);
    return {};
  }
};
template <> struct Row<-1> {
  enum class row : ptrdiff_t {};
  row value_;
  [[gnu::artificial, gnu::always_inline]] inline explicit constexpr
  operator ptrdiff_t() const {
    auto m = static_cast<ptrdiff_t>(value_);
    invariant(m >= 0);
    return m;
  }
  [[gnu::artificial, gnu::always_inline]] inline explicit constexpr
  operator bool() const {
    return static_cast<ptrdiff_t>(value_);
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  operator++() -> Row & {
    value_ = static_cast<row>(static_cast<ptrdiff_t>(value_) + 1z);
    return *this;
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  operator--() -> Row & {
    value_ = static_cast<row>(static_cast<ptrdiff_t>(value_) - 1z);
    return *this;
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  operator++(int) -> Row {
    Row tmp{*this};
    value_ = static_cast<row>(static_cast<ptrdiff_t>(value_) + 1z);
    return tmp;
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  operator--(int) -> Row {
    Row tmp{*this};
    value_ = static_cast<row>(static_cast<ptrdiff_t>(value_) + 1z);
    return tmp;
  }
  static constexpr auto comptime() -> ptrdiff_t { return -1; }

private:
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(ptrdiff_t x, Row y) -> bool {
    return x == ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(Row y, ptrdiff_t x) -> bool {
    return x == ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(Row x, Row y) -> bool {
    return ptrdiff_t(x) == ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(ptrdiff_t x, Row y) -> std::strong_ordering {
    return x <=> ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Row x, ptrdiff_t y) -> std::strong_ordering {
    return ptrdiff_t(x) <=> y;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Row x, Row y) -> std::strong_ordering {
    return ptrdiff_t(x) <=> ptrdiff_t(y);
  }
  friend inline auto operator<<(std::ostream &os, Row<> x) -> std::ostream & {
    return os << "Row<>{" << ptrdiff_t(x) << "}";
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator+(Row a, Row b) -> Row {
    return {static_cast<Row<-1>::row>(ptrdiff_t(a) + ptrdiff_t(b))};
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator-(Row a, Row b) -> Row {
    auto x = ptrdiff_t(a), y = ptrdiff_t(b);
    invariant(x >= y);
    return {static_cast<Row<-1>::row>(x - y)};
  }
};
static_assert(sizeof(Row<>) == sizeof(ptrdiff_t));
template <ptrdiff_t M>
[[gnu::artificial,
  gnu::always_inline]] inline constexpr Row<M>::operator Row<-1>() const {
  return {static_cast<Row<-1>::row>(M)};
}
template <ptrdiff_t M = -1> struct Col {
  static_assert(M >= 0);
  [[gnu::artificial, gnu::always_inline]] inline explicit constexpr
  operator ptrdiff_t() const {
    return M;
  }
  [[gnu::artificial, gnu::always_inline]] inline explicit constexpr
  operator bool() const {
    return M;
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr
  operator Col<-1>() const;
  static constexpr auto comptime() -> ptrdiff_t { return M; }

private:
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(ptrdiff_t x, Col) -> bool {
    return x == M;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(Col, ptrdiff_t x) -> bool {
    return M == x;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(Col, Col) -> bool {
    return true;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(ptrdiff_t x, Col) -> std::strong_ordering {
    return x <=> M;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Col, ptrdiff_t y) -> std::strong_ordering {
    return M <=> y;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Col, Col) -> std::strong_ordering {
    return std::strong_ordering::equal;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator*(Row<> r, Col) -> ptrdiff_t {
    return ptrdiff_t(r) * M;
  }
  friend inline auto operator<<(std::ostream &os, Col) -> std::ostream & {
    return os << "Col<>{" << M << "}";
  }
  template <ptrdiff_t N>
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator+(Col, Col<N>) -> Col<M + N>
  requires(N != -1)
  {
    return {};
  }
  template <ptrdiff_t N>
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator-(Col, Col<N>) -> Col<M - N>
  requires(N != -1)
  {
    static_assert(M >= N);
    return {};
  }
};
template <> struct Col<-1> {
  enum class col : ptrdiff_t {};
  col value_;
  [[gnu::artificial, gnu::always_inline]] inline explicit constexpr
  operator ptrdiff_t() const {
    auto m = static_cast<ptrdiff_t>(value_);
    invariant(m >= 0);
    return m;
  }
  [[gnu::artificial, gnu::always_inline]] inline explicit constexpr
  operator bool() const {
    return static_cast<ptrdiff_t>(value_);
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  operator++() -> Col & {
    value_ = static_cast<col>(static_cast<ptrdiff_t>(value_) + 1z);
    return *this;
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  operator--() -> Col & {
    value_ = static_cast<col>(static_cast<ptrdiff_t>(value_) - 1z);
    return *this;
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  operator++(int) -> Col {
    Col tmp{*this};
    value_ = static_cast<col>(static_cast<ptrdiff_t>(value_) + 1z);
    return tmp;
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  operator--(int) -> Col {
    Col tmp{*this};
    value_ = static_cast<col>(static_cast<ptrdiff_t>(value_) - 1z);
    return tmp;
  }
  static constexpr auto comptime() -> ptrdiff_t { return -1; }

private:
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(ptrdiff_t x, Col y) -> bool {
    return x == ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(Col y, ptrdiff_t x) -> bool {
    return x == ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(Col x, Col y) -> bool {
    return ptrdiff_t(x) == ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(ptrdiff_t x, Col y) -> std::strong_ordering {
    return x <=> ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Col x, ptrdiff_t y) -> std::strong_ordering {
    return ptrdiff_t(x) <=> y;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Col x, Col y) -> std::strong_ordering {
    return ptrdiff_t(x) <=> ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator*(Row<> r, Col c) -> ptrdiff_t {
    return ptrdiff_t(r) * ptrdiff_t(c);
  }
  friend inline auto operator<<(std::ostream &os, Col x) -> std::ostream & {
    return os << "Col<>{" << ptrdiff_t(x) << "}";
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator+(Col a, Col b) -> Col {
    return {static_cast<Col<-1>::col>(ptrdiff_t(a) + ptrdiff_t(b))};
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator-(Col a, Col b) -> Col {
    auto x = ptrdiff_t(a), y = ptrdiff_t(b);
    invariant(x >= y);
    return {static_cast<Col<-1>::col>(x - y)};
  }
};
static_assert(sizeof(Col<>) == sizeof(ptrdiff_t));
template <ptrdiff_t M>
[[gnu::artificial,
  gnu::always_inline]] inline constexpr Col<M>::operator Col<-1>() const {
  return {static_cast<Col<-1>::col>(M)};
}
template <ptrdiff_t M = -1> struct RowStride {
  static_assert(M >= 0);
  [[gnu::artificial, gnu::always_inline]] inline explicit constexpr
  operator ptrdiff_t() const {
    return M;
  }
  [[gnu::artificial, gnu::always_inline]] inline explicit constexpr
  operator bool() const {
    return M;
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr
  operator RowStride<-1>() const;
  static constexpr auto comptime() -> ptrdiff_t { return M; }

private:
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(ptrdiff_t x, RowStride) -> bool {
    return x == M;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(RowStride, ptrdiff_t x) -> bool {
    return M == x;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(RowStride, RowStride) -> bool {
    return true;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(ptrdiff_t x, RowStride) -> std::strong_ordering {
    return x <=> M;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(RowStride, ptrdiff_t y) -> std::strong_ordering {
    return M <=> y;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(RowStride, RowStride) -> std::strong_ordering {
    return std::strong_ordering::equal;
  }
  friend inline auto operator<<(std::ostream &os, RowStride) -> std::ostream & {
    return os << "RowStride<>{" << M << "}";
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(Col<> c, RowStride) -> bool {
    return ptrdiff_t(c) == M;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Col<> c, RowStride) -> std::strong_ordering {
    return ptrdiff_t(c) <=> M;
  }
};
template <> struct RowStride<-1> {
  enum class stride : ptrdiff_t {};
  stride value_;
  [[gnu::artificial, gnu::always_inline]] inline explicit constexpr
  operator ptrdiff_t() const {
    auto m = static_cast<ptrdiff_t>(value_);
    invariant(m >= 0);
    return m;
  }
  [[gnu::artificial, gnu::always_inline]] inline explicit constexpr
  operator bool() const {
    return static_cast<ptrdiff_t>(value_);
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  operator++() -> RowStride & {
    value_ = static_cast<stride>(static_cast<ptrdiff_t>(value_) + 1z);
    return *this;
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  operator--() -> RowStride & {
    value_ = static_cast<stride>(static_cast<ptrdiff_t>(value_) - 1z);
    return *this;
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  operator++(int) -> RowStride {
    RowStride tmp{*this};
    value_ = static_cast<stride>(static_cast<ptrdiff_t>(value_) + 1z);
    return tmp;
  }
  [[gnu::artificial, gnu::always_inline]] inline constexpr auto
  operator--(int) -> RowStride {
    RowStride tmp{*this};
    value_ = static_cast<stride>(static_cast<ptrdiff_t>(value_) - 1z);
    return tmp;
  }
  static constexpr auto comptime() -> ptrdiff_t { return -1; }

private:
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(ptrdiff_t x, RowStride y) -> bool {
    return x == ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(RowStride y, ptrdiff_t x) -> bool {
    return x == ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(RowStride x, RowStride y) -> bool {
    return ptrdiff_t(x) == ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(ptrdiff_t x, RowStride y) -> std::strong_ordering {
    return x <=> ptrdiff_t(y);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(RowStride x, ptrdiff_t y) -> std::strong_ordering {
    return ptrdiff_t(x) <=> y;
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(RowStride x, RowStride y) -> std::strong_ordering {
    return ptrdiff_t(x) <=> ptrdiff_t(y);
  }
  friend inline auto operator<<(std::ostream &os,
                                RowStride x) -> std::ostream & {
    return os << "RowStride<>{" << ptrdiff_t(x) << "}";
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator==(Col<> c, RowStride x) -> bool {
    return ptrdiff_t(c) == ptrdiff_t(x);
  }
  [[gnu::artificial, gnu::always_inline]] friend inline constexpr auto
  operator<=>(Col<> c, RowStride x) -> std::strong_ordering {
    return ptrdiff_t(c) <=> ptrdiff_t(x);
  }
};
static_assert(sizeof(RowStride<>) == sizeof(ptrdiff_t));
template <ptrdiff_t M>
[[gnu::artificial,
  gnu::always_inline]] inline constexpr RowStride<M>::operator RowStride<-1>()
  const {
  return {static_cast<RowStride<-1>::stride>(M)};
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

template <ptrdiff_t M>
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
unwrapRow(Row<M> x) {
  if constexpr (M == -1) return ptrdiff_t(x);
  else return std::integral_constant<ptrdiff_t, M>{};
}
template <ptrdiff_t M>
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
unwrapCol(Col<M> x) {
  if constexpr (M == -1) return ptrdiff_t(x);
  else return std::integral_constant<ptrdiff_t, M>{};
}
template <ptrdiff_t M>
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
unwrapStride(RowStride<M> x) {
  if constexpr (M == -1) return ptrdiff_t(x);
  else return std::integral_constant<ptrdiff_t, M>{};
}
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
unwrapRow(auto x) {
  return x;
}
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
unwrapCol(auto x) {
  return x;
}
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
unwrapStride(auto x) {
  return x;
}

[[gnu::artificial, gnu::always_inline]] inline constexpr auto
row(ptrdiff_t x) -> Row<> {
  invariant(x >= 0);
  return Row<-1>{static_cast<Row<-1>::row>(x)};
}
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
col(ptrdiff_t x) -> Col<> {
  invariant(x >= 0);
  return Col<-1>{static_cast<Col<-1>::col>(x)};
}
template <ptrdiff_t M>
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
col(Length<M> x) -> Col<M> {
  if constexpr (M != -1) return Col<M>{};
  else return Col<-1>{static_cast<Col<-1>::col>(ptrdiff_t(x))};
}
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
stride(ptrdiff_t x) -> RowStride<> {
  invariant(x >= 0);
  return RowStride<-1>{static_cast<RowStride<-1>::stride>(x)};
}
template <std::signed_integral Int>
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
length(Int x) -> Length<-1, Int>
requires(!std::same_as<Int, ptrdiff_t>)
{
  invariant(x >= 0);
  return Length<-1, Int>{static_cast<Length<-1, Int>::len>(x)};
}
// Overload resolution should prioritize/favor `ptrdiff_t`
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
length(ptrdiff_t x) -> Length<> {
  invariant(x >= 0);
  return Length<>{static_cast<Length<>::len>(x)};
}
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
capacity(ptrdiff_t x) -> Capacity<> {
  invariant(x >= 0);
  return Capacity<-1>{static_cast<Capacity<-1>::cap>(x)};
}
// [[gnu::artificial, gnu::always_inline]] inline constexpr auto
// capacity(size_t x) -> Capacity<> {
//   invariant(x <= size_t(std::numeric_limits<ptrdiff_t>::max()));
//   return capacity(ptrdiff_t(x));
// }
template <std::integral I, I x>
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
row(std::integral_constant<I, x>) -> Row<ptrdiff_t(x)> {
  static_assert(x >= 0);
  return {};
}
template <std::integral I, I x>
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
col(std::integral_constant<I, x>) -> Col<ptrdiff_t(x)> {
  static_assert(x >= 0);
  return {};
}
template <std::integral I, I x>
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
stride(std::integral_constant<I, x>) -> RowStride<ptrdiff_t(x)> {
  static_assert(x >= 0);
  return {};
}

template <ptrdiff_t M>
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
aslength(Col<M> len) -> Length<M> {
  if constexpr (M != -1) return {};
  else return {static_cast<Length<-1>::len>(ptrdiff_t(len))};
}
template <ptrdiff_t M>
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
aslength(Row<M> len) -> Length<M> {
  if constexpr (M != -1) return {};
  else return {static_cast<Length<-1>::len>(ptrdiff_t(len))};
}
template <ptrdiff_t M>
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
asrow(Length<M> len) -> Row<M> {
  if constexpr (M != -1) return {};
  else return {static_cast<Row<-1>::row>(ptrdiff_t(len))};
}
template <ptrdiff_t M>
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
asrow(Col<M> len) -> Row<M> {
  if constexpr (M != -1) return {};
  else return {static_cast<Row<-1>::row>(ptrdiff_t(len))};
}
template <ptrdiff_t M>
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
ascol(Length<M> len) -> Col<M> {
  if constexpr (M != -1) return {};
  else return {static_cast<Col<-1>::col>(ptrdiff_t(len))};
}
template <ptrdiff_t M>
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
ascol(Row<M> len) -> Col<M> {
  if constexpr (M != -1) return {};
  else return {static_cast<Col<-1>::col>(ptrdiff_t(len))};
}
template <ptrdiff_t M>
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
asrowStride(Length<M> len) -> RowStride<M> {
  if constexpr (M != -1) return {};
  else return {static_cast<RowStride<-1>::stride>(ptrdiff_t(len))};
}

} // namespace math
