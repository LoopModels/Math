#pragma once
#include "Utilities/Invariant.hpp"
#include <cstddef>
#include <cstdint>
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
/// Their lifetimes are governed by the BumpAlloc or RAII type used to back
/// them.
namespace poly::math {
using utils::invariant;
enum class AxisType {
  Row,
  Column,
  RowStride,
};
inline auto operator<<(std::ostream &os, AxisType x) -> std::ostream & {
  switch (x) {
  case AxisType::Row: os << "Row"; break;
  case AxisType::Column: os << "Column"; break;
  case AxisType::RowStride: os << "RowStride"; break;
  default: os << "invalid axis type"; __builtin_trap();
  }
  return os;
}

// strong typing
template <AxisType T> struct AxisInt {
  using V = ptrdiff_t;
  [[no_unique_address]] V value{0};
  // [[no_unique_address]] unsigned int value{0};
  constexpr AxisInt() = default;
  constexpr AxisInt(V v) : value(v) {}
  explicit constexpr operator size_t() const {
    invariant(value >= 0);
    return value;
  }
  explicit constexpr operator ptrdiff_t() const { return value; }
  explicit constexpr operator unsigned() const {
    invariant(value >= 0);
    invariant(value <= std::numeric_limits<unsigned>::max());
    return value;
  }
  explicit constexpr operator uint8_t() const {
    invariant(value >= 0);
    invariant(value < 256);
    return value;
  }
  explicit constexpr operator bool() const { return value; }

  constexpr auto operator+(V i) const -> AxisInt<T> { return value + i; }
  constexpr auto operator-(V i) const -> AxisInt<T> { return value - i; }
  constexpr auto operator*(V i) const -> AxisInt<T> { return value * i; }
  constexpr auto operator/(V i) const -> AxisInt<T> { return value / i; }
  constexpr auto operator%(V i) const -> AxisInt<T> { return value % i; }
  constexpr auto operator==(V i) const -> bool { return value == i; }
  constexpr auto operator!=(V i) const -> bool { return value != i; }
  constexpr auto operator<(V i) const -> bool { return value < i; }
  constexpr auto operator<=(V i) const -> bool { return value <= i; }
  constexpr auto operator>(V i) const -> bool { return value > i; }
  constexpr auto operator>=(V i) const -> bool { return value >= i; }
  constexpr auto operator++() -> AxisInt<T> & {
    ++value;
    return *this;
  }
  constexpr auto operator++(int) -> AxisInt<T> { return value++; }
  constexpr auto operator--() -> AxisInt<T> & {
    --value;
    return *this;
  }
  constexpr auto operator--(int) -> AxisInt<T> { return value--; }
  constexpr auto operator+=(AxisInt<T> i) -> AxisInt<T> & {
    value += V(i);
    return *this;
  }
  constexpr auto operator+=(V i) -> AxisInt<T> & {
    value += i;
    return *this;
  }
  constexpr auto operator-=(AxisInt<T> i) -> AxisInt<T> & {
    value -= V(i);
    return *this;
  }
  constexpr auto operator-=(V i) -> AxisInt<T> & {
    value -= i;
    return *this;
  }
  constexpr auto operator*=(AxisInt<T> i) -> AxisInt<T> & {
    value *= V(i);
    return *this;
  }
  constexpr auto operator*=(V i) -> AxisInt<T> & {
    value *= i;
    return *this;
  }
  constexpr auto operator/=(AxisInt<T> i) -> AxisInt<T> & {
    value /= V(i);
    return *this;
  }
  constexpr auto operator/=(V i) -> AxisInt<T> & {
    value /= i;
    return *this;
  }
  constexpr auto operator%=(AxisInt<T> i) -> AxisInt<T> & {
    value %= V(i);
    return *this;
  }
  constexpr auto operator%=(V i) -> AxisInt<T> & {
    value %= i;
    return *this;
  }
  constexpr auto operator*() const -> V { return value; }
  friend inline auto operator<<(std::ostream &os, AxisInt<T> x)
    -> std::ostream & {
    return os << T << "{" << *x << "}";
  }
};
template <typename T, AxisType W>
constexpr auto operator+(T *p, AxisInt<W> y) -> T * {
  return p + *y;
}
template <typename T, AxisType W>
constexpr auto operator-(T *p, AxisInt<W> y) -> T * {
  return p - *y;
}

template <AxisType T>
constexpr auto operator+(AxisInt<T> x, AxisInt<T> y) -> AxisInt<T> {
  return (*x) + (*y);
}
template <AxisType T>
constexpr auto operator-(AxisInt<T> x, AxisInt<T> y) -> AxisInt<T> {
  return (*x) - (*y);
}
template <AxisType T>
constexpr auto operator*(AxisInt<T> x, AxisInt<T> y) -> AxisInt<T> {
  return (*x) * (*y);
}
template <AxisType T>
constexpr auto operator/(AxisInt<T> x, AxisInt<T> y) -> AxisInt<T> {
  return (*x) / (*y);
}
template <AxisType T>
constexpr auto operator%(AxisInt<T> x, AxisInt<T> y) -> AxisInt<T> {
  return (*x) % (*y);
}
template <AxisType S, AxisType T>
constexpr auto operator==(AxisInt<S> x, AxisInt<T> y) -> bool {
  return *x == *y;
}
template <AxisType S, AxisType T>
constexpr auto operator!=(AxisInt<S> x, AxisInt<T> y) -> bool {
  return *x != *y;
}
template <AxisType T>
constexpr auto operator<(AxisInt<T> x, AxisInt<T> y) -> bool {
  return *x < *y;
}
template <AxisType T>
constexpr auto operator<=(AxisInt<T> x, AxisInt<T> y) -> bool {
  return *x <= *y;
}
template <AxisType T>
constexpr auto operator>(AxisInt<T> x, AxisInt<T> y) -> bool {
  return *x > *y;
}
template <AxisType T>
constexpr auto operator>=(AxisInt<T> x, AxisInt<T> y) -> bool {
  return *x >= *y;
}
using Col = AxisInt<AxisType::Column>;
using Row = AxisInt<AxisType::Row>;
using RowStride = AxisInt<AxisType::RowStride>;
using CarInd = std::pair<Row, Col>;

constexpr auto operator*(RowStride x, Row y) -> ptrdiff_t {
  return (*x) * (*y);
}
constexpr auto operator>=(RowStride x, Col u) -> bool { return (*x) >= (*u); }
constexpr auto operator<(RowStride x, Col u) -> bool { return (*x) < (*u); }

static_assert(std::is_trivially_copyable_v<Row>);
static_assert(std::is_trivially_copyable_v<Col>);
static_assert(std::is_trivially_copyable_v<RowStride>);
static_assert(std::is_trivially_copyable_v<const Row>);
static_assert(std::is_trivially_copyable_v<const Col>);
static_assert(std::is_trivially_copyable_v<const RowStride>);
static_assert(sizeof(Row) == sizeof(ptrdiff_t));
static_assert(sizeof(Col) == sizeof(ptrdiff_t));
static_assert(sizeof(RowStride) == sizeof(ptrdiff_t));
constexpr auto operator*(Row r, Col c) -> Row::V { return *r * *c; }

constexpr auto operator==(ptrdiff_t x, Row y) -> bool {
  return x == ptrdiff_t(y);
}
constexpr auto operator!=(ptrdiff_t x, Row y) -> bool {
  return x != ptrdiff_t(y);
}
constexpr auto operator==(ptrdiff_t x, Col y) -> bool {
  return x == ptrdiff_t(y);
}
constexpr auto operator!=(ptrdiff_t x, Col y) -> bool {
  return x != ptrdiff_t(y);
}
constexpr auto operator<(ptrdiff_t x, Row y) -> bool {
  return x < ptrdiff_t(y);
}
constexpr auto operator<(ptrdiff_t x, Col y) -> bool {
  return x < ptrdiff_t(y);
}
constexpr auto operator>(ptrdiff_t x, Row y) -> bool {
  return x > ptrdiff_t(y);
}
constexpr auto operator>(ptrdiff_t x, Col y) -> bool {
  return x > ptrdiff_t(y);
}
constexpr auto operator<=(ptrdiff_t x, Row y) -> bool {
  return x <= ptrdiff_t(y);
}
constexpr auto operator<=(ptrdiff_t x, Col y) -> bool {
  return x <= ptrdiff_t(y);
}
constexpr auto operator>=(ptrdiff_t x, Row y) -> bool {
  return x >= ptrdiff_t(y);
}
constexpr auto operator>=(ptrdiff_t x, Col y) -> bool {
  return x >= ptrdiff_t(y);
}

constexpr auto operator+(ptrdiff_t x, Col y) -> Col {
  return Col{x + ptrdiff_t(y)};
}
constexpr auto operator-(ptrdiff_t x, Col y) -> Col {
  return Col{x - ptrdiff_t(y)};
}
constexpr auto operator*(ptrdiff_t x, Col y) -> Col {
  return Col{x * ptrdiff_t(y)};
}
constexpr auto operator+(ptrdiff_t x, Row y) -> Row {
  return Row{x + ptrdiff_t(y)};
}
constexpr auto operator-(ptrdiff_t x, Row y) -> Row {
  return Row{x - ptrdiff_t(y)};
}
constexpr auto operator*(ptrdiff_t x, Row y) -> Row {
  return Row{x * ptrdiff_t(y)};
}
constexpr auto operator+(ptrdiff_t x, RowStride y) -> RowStride {
  return RowStride{x + ptrdiff_t(y)};
}
constexpr auto operator-(ptrdiff_t x, RowStride y) -> RowStride {
  return RowStride{x - ptrdiff_t(y)};
}
constexpr auto operator*(ptrdiff_t x, RowStride y) -> RowStride {
  return RowStride{x * ptrdiff_t(y)};
}

constexpr auto max(Row M, Col N) -> ptrdiff_t {
  return std::max(ptrdiff_t(M), ptrdiff_t(N));
}
constexpr auto max(Col N, RowStride X) -> RowStride {
  return RowStride{std::max(ptrdiff_t(N), ptrdiff_t(X))};
}
constexpr auto min(Col N, Col X) -> Col {
  return Col{std::max(Col::V(N), Col::V(X))};
}
constexpr auto min(Row N, Col X) -> ptrdiff_t {
  return std::min(ptrdiff_t(N), ptrdiff_t(X));
}

template <typename T>
concept RowOrCol = std::same_as<T, Row> || std::same_as<T, Col>;

constexpr auto unwrapRow(Row x) -> ptrdiff_t { return ptrdiff_t(x); }
constexpr auto unwrapCol(Col x) -> ptrdiff_t { return ptrdiff_t(x); }
constexpr auto unwrapRow(auto x) { return x; }
constexpr auto unwrapCol(auto x) { return x; }
constexpr auto standardizeRangeBound(RowOrCol auto x) { return ptrdiff_t(x); }
} // namespace poly::math
