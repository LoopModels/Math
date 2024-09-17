#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#include "Owner.hxx"
#ifndef USE_MODULE
#include <cstddef>
#include <type_traits>
#endif

#ifndef USE_MODULE
#include "Utilities/Invariant.cxx"
#else
export module Valid;

import Invariant;
import STL;
#endif

#ifdef USE_MODULE
export namespace utils {
#else
namespace utils {
#endif
// TODO: communicate not-null to the compiler somehow?
template <typename T> class MATH_GSL_POINTER Valid {
  [[no_unique_address]] T *value;

public:
  Valid() = delete;
  constexpr Valid(T *v) : value(v) { invariant(value != nullptr); }
  constexpr explicit operator bool() const {
    invariant(value != nullptr);
    return true;
  }
  constexpr operator Valid<const T>() const {
    invariant(value != nullptr);
    return Valid<const T>(value);
  }
  [[gnu::returns_nonnull]] constexpr operator T *() {
    invariant(value != nullptr);
    return value;
  }
  [[gnu::returns_nonnull]] constexpr operator T *() const {
    invariant(value != nullptr);
    return value;
  }
  [[gnu::returns_nonnull]] constexpr auto operator->() -> T * {
    invariant(value != nullptr);
    return value;
  }
  constexpr auto operator*() -> T & {
    invariant(value != nullptr);
    return *value;
  }
  [[gnu::returns_nonnull]] constexpr auto operator->() const -> const T * {
    invariant(value != nullptr);
    return value;
  }
  constexpr auto operator*() const -> const T & {
    invariant(value != nullptr);
    return *value;
  }
  constexpr auto operator[](ptrdiff_t index) -> T & {
    invariant(value != nullptr);
    return value[index];
  }
  constexpr auto operator[](ptrdiff_t index) const -> const T & {
    invariant(value != nullptr);
    return value[index];
  }
  constexpr auto operator+(ptrdiff_t offset) -> Valid<T> {
    invariant(value != nullptr);
    return value + offset;
  }
  constexpr auto operator-(ptrdiff_t offset) -> Valid<T> {
    invariant(value != nullptr);
    return value - offset;
  }
  constexpr auto operator+(ptrdiff_t offset) const -> Valid<T> {
    invariant(value != nullptr);
    return value + offset;
  }
  constexpr auto operator-(ptrdiff_t offset) const -> Valid<T> {
    invariant(value != nullptr);
    return value - offset;
  }
  constexpr auto operator++() -> Valid<T> & {
    invariant(value != nullptr);
    ++value;
    return *this;
  }
  constexpr auto operator++(int) -> Valid<T> {
    invariant(value != nullptr);
    return value++;
  }
  constexpr auto operator--() -> Valid<T> & {
    invariant(value != nullptr);
    --value;
    return *this;
  }
  constexpr auto operator--(int) -> Valid<T> {
    invariant(value != nullptr);
    return value--;
  }
  constexpr auto operator+=(ptrdiff_t offset) -> Valid<T> & {
    invariant(value != nullptr);
    value += offset;
    return *this;
  }
  constexpr auto operator-=(ptrdiff_t offset) -> Valid<T> & {
    invariant(value != nullptr);
    value -= offset;
    return *this;
  }
  // constexpr auto operator==(const Valid<T> &other) const -> bool {
  //   invariant(value != nullptr);
  //   return value == other.value;
  // }
  // constexpr auto operator!=(const Valid<T> &other) const -> bool {
  //   invariant(value != nullptr);
  //   return value != other.value;
  // }
  // constexpr auto operator<(const Valid<T> &other) const -> bool {
  //   invariant(value != nullptr);
  //   return value < other.value;
  // }
  // constexpr auto operator<=(const Valid<T> &other) const -> bool {
  //   invariant(value != nullptr);
  //   return value <= other.value;
  // }
  // constexpr auto operator>(const Valid<T> &other) const -> bool {
  //   invariant(value != nullptr);
  //   return value > other.value;
  // }
  // constexpr auto operator>=(const Valid<T> &other) const -> bool {
  //   invariant(value != nullptr);
  //   return value >= other.value;
  // }
  constexpr auto operator-(const Valid<T> &other) const -> ptrdiff_t {
    invariant(value != nullptr);
    return value - other.value;
  }
  [[nodiscard]] constexpr auto isAligned(ptrdiff_t x) const -> bool {
    invariant(value != nullptr);
    return (reinterpret_cast<ptrdiff_t>(value) % x) == 0;
  }
};
template <typename T> Valid(T &) -> Valid<T>;
template <typename T> Valid(T *) -> Valid<T *>;
static_assert(std::is_trivially_destructible_v<Valid<ptrdiff_t>>);
static_assert(std::is_trivially_copy_constructible_v<Valid<ptrdiff_t>>);
} // namespace utils
