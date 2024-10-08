#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifndef USE_MODULE
#include <concepts>
#include <cstddef>
#include <iterator>
#include <ostream>
#include <type_traits>

#include "Math/AxisTypes.cxx"
#include "Utilities/Invariant.cxx"
#else
export module Range;

import AxisTypes;
import Invariant;
import STL;
#endif

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif
template <ptrdiff_t M>
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
standardizeRangeBound(math::Row<M> x) {
  if constexpr (M == -1) return ptrdiff_t(x);
  else return std::integral_constant<ptrdiff_t, M>{};
}
template <ptrdiff_t M>
[[gnu::artificial, gnu::always_inline]] inline constexpr auto
standardizeRangeBound(math::Col<M> x) {
  if constexpr (M == -1) return ptrdiff_t(x);
  else return std::integral_constant<ptrdiff_t, M>{};
}

constexpr auto standardizeRangeBound(auto x) { return x; }
constexpr auto standardizeRangeBound(std::unsigned_integral auto x) {
  return ptrdiff_t(x);
}
constexpr auto standardizeRangeBound(std::signed_integral auto x) {
  return ptrdiff_t(x);
}
using utils::invariant;
template <typename B, typename E> struct Range {
  [[no_unique_address]] B b;
  [[no_unique_address]] E e;
  [[nodiscard]] constexpr auto begin() const -> B { return b; }
  [[nodiscard]] constexpr auto end() const -> E { return e; }

private:
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator+(Range r, ptrdiff_t x) {
    return Range{r.b + x, r.e + x};
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator-(Range r, ptrdiff_t x) {
    return Range{r.b - x, r.e - x};
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator+(ptrdiff_t x, Range r) {
    return Range{r.b + x, r.e + x};
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator-(ptrdiff_t x, Range r) {
    return Range{x - r.b, x - r.e};
  }
};
template <std::integral B, std::integral E> struct Range<B, E> {
  using value_type = std::common_type_t<B, E>;
  [[no_unique_address]] B b;
  [[no_unique_address]] E e;
  // wrapper that allows dereferencing
  struct Iterator {
    [[no_unique_address]] B i;
    constexpr auto operator==(E other) -> bool { return i == other; }
    auto operator++() -> Iterator & {
      ++i;
      return *this;
    }
    auto operator++(int) -> Iterator { return Iterator{i++}; }
    auto operator--() -> Iterator & {
      --i;
      return *this;
    }
    auto operator--(int) -> Iterator { return Iterator{i--}; }
    auto operator*() -> B { return i; }
  };
  [[nodiscard]] constexpr auto begin() const -> Iterator { return Iterator{b}; }
  [[nodiscard]] constexpr auto end() const -> E { return e; }
  [[nodiscard]] constexpr auto rbegin() const -> Iterator {
    return std::reverse_iterator{end()};
  }
  [[nodiscard]] constexpr auto rend() const -> E {
    return std::reverse_iterator{begin()};
  }
  [[nodiscard]] constexpr auto size() const { return e - b; }
  friend auto operator<<(std::ostream &os, Range<B, E> r) -> std::ostream & {
    return os << "[" << r.b << ":" << r.e << ")";
  }
  template <std::integral BB, std::integral EE>
  constexpr operator Range<BB, EE>() const {
    return Range<BB, EE>{static_cast<BB>(b), static_cast<EE>(e)};
  }
  [[nodiscard]] constexpr auto view() const -> Range { return *this; }
  [[nodiscard]] constexpr auto operator[](value_type i) const -> value_type {
    return b + i;
  }
  [[nodiscard]] constexpr auto operator[](auto i) const { return i + b; }

private:
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator+(Range r, ptrdiff_t x) {
    return Range{r.b + x, r.e + x};
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator-(Range r, ptrdiff_t x) {
    return Range{r.b - x, r.e - x};
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator+(ptrdiff_t x, Range r) {
    return Range{r.b + x, r.e + x};
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator-(ptrdiff_t x, Range r) {
    return Range{x - r.b, x - r.e};
  }
};
template <typename B, typename E>
Range(B b, E e) -> Range<decltype(standardizeRangeBound(b)),
                         decltype(standardizeRangeBound(e))>;

constexpr auto skipFirst(const auto &x) {
  auto b = x.begin();
  return Range{++b, x.end()};
}

template <typename T> struct StridedIterator {
  using value_type = std::remove_cvref_t<T>;
  T *ptr;
  RowStride<> stride;
  constexpr auto operator==(const StridedIterator &other) const -> bool {
    return ptr == other.ptr;
  }
  constexpr auto operator!=(const StridedIterator &other) const -> bool {
    return ptr != other.ptr;
  }
  constexpr auto operator<(const StridedIterator &other) const -> bool {
    return ptr < other.ptr;
  }
  constexpr auto operator>(const StridedIterator &other) const -> bool {
    return ptr > other.ptr;
  }
  constexpr auto operator<=(const StridedIterator &other) const -> bool {
    return ptr <= other.ptr;
  }
  constexpr auto operator>=(const StridedIterator &other) const -> bool {
    return ptr >= other.ptr;
  }
  constexpr auto operator*() const -> T & { return *ptr; }
  constexpr auto operator->() const -> T * { return ptr; }
  constexpr auto operator++() -> StridedIterator & {
    ptr += ptrdiff_t(stride);
    return *this;
  }
  constexpr auto operator++(int) -> StridedIterator {
    auto tmp = *this;
    ptr += ptrdiff_t(stride);
    return tmp;
  }
  constexpr auto operator--() -> StridedIterator & {
    ptr -= ptrdiff_t(stride);
    return *this;
  }
  constexpr auto operator--(int) -> StridedIterator {
    auto tmp = *this;
    ptr -= ptrdiff_t(stride);
    return tmp;
  }
  constexpr auto operator+(ptrdiff_t x) const -> StridedIterator {
    return StridedIterator{ptr + x * ptrdiff_t(stride), stride};
  }
  constexpr auto operator-(ptrdiff_t x) const -> StridedIterator {
    return StridedIterator{ptr - x * ptrdiff_t(stride), stride};
  }
  constexpr auto operator+=(ptrdiff_t x) -> StridedIterator & {
    ptr += x * ptrdiff_t(stride);
    return *this;
  }
  constexpr auto operator-=(ptrdiff_t x) -> StridedIterator & {
    ptr -= x * ptrdiff_t(stride);
    return *this;
  }
  constexpr auto operator-(const StridedIterator &other) const -> ptrdiff_t {
    invariant(stride == other.stride);
    return (ptr - other.ptr) / ptrdiff_t(stride);
  }
  constexpr auto operator+(const StridedIterator &other) const -> ptrdiff_t {
    invariant(stride == other.stride);
    return (ptr + other.ptr) / ptrdiff_t(stride);
  }
  constexpr auto operator[](ptrdiff_t x) const -> T & {
    return ptr[x * ptrdiff_t(stride)];
  }
  friend constexpr auto
  operator+(ptrdiff_t x, const StridedIterator &it) -> StridedIterator {
    return it + x;
  }
};
template <class T> StridedIterator(T *, RowStride<>) -> StridedIterator<T>;
static_assert(std::weakly_incrementable<StridedIterator<int64_t>>);
static_assert(std::input_or_output_iterator<StridedIterator<int64_t>>);
static_assert(std::indirectly_readable<StridedIterator<int64_t>>,
              "failed indirectly readable");
static_assert(std::indirectly_readable<StridedIterator<int64_t>>,
              "failed indirectly readable");
static_assert(std::output_iterator<StridedIterator<int64_t>, ptrdiff_t>,
              "failed output iterator");
static_assert(std::forward_iterator<StridedIterator<int64_t>>,
              "failed forward iterator");
static_assert(std::input_iterator<StridedIterator<int64_t>>,
              "failed input iterator");
static_assert(std::bidirectional_iterator<StridedIterator<int64_t>>,
              "failed bidirectional iterator");

static_assert(std::totally_ordered<StridedIterator<int64_t>>,
              "failed random access iterator");
static_assert(std::random_access_iterator<StridedIterator<int64_t>>,
              "failed random access iterator");
} // namespace math
