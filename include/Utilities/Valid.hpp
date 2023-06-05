#pragma once
#include "Utilities/Invariant.hpp"
#include <cstddef>
#include <type_traits>

namespace poly::utils {
// TODO: communicate not-null to the compiler somehow?
template <typename T> class NotNull {
  [[no_unique_address]] T *value;

public:
  NotNull() = delete;
  constexpr NotNull(T &v) : value(&v) {}
  constexpr NotNull(T *v) : value(v) { invariant(value != nullptr); }
  constexpr explicit operator bool() const {
    invariant(value != nullptr);
    return true;
  }
  constexpr operator NotNull<const T>() const {
    invariant(value != nullptr);
    return NotNull<const T>(*value);
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
  constexpr auto operator[](size_t index) -> T & {
    invariant(value != nullptr);
    return value[index];
  }
  constexpr auto operator[](size_t index) const -> const T & {
    invariant(value != nullptr);
    return value[index];
  }
  constexpr auto operator+(size_t offset) -> NotNull<T> {
    invariant(value != nullptr);
    return value + offset;
  }
  constexpr auto operator-(size_t offset) -> NotNull<T> {
    invariant(value != nullptr);
    return value - offset;
  }
  constexpr auto operator+(size_t offset) const -> NotNull<T> {
    invariant(value != nullptr);
    return value + offset;
  }
  constexpr auto operator-(size_t offset) const -> NotNull<T> {
    invariant(value != nullptr);
    return value - offset;
  }
  constexpr auto operator++() -> NotNull<T> & {
    invariant(value != nullptr);
    ++value;
    return *this;
  }
  constexpr auto operator++(int) -> NotNull<T> {
    invariant(value != nullptr);
    return value++;
  }
  constexpr auto operator--() -> NotNull<T> & {
    invariant(value != nullptr);
    --value;
    return *this;
  }
  constexpr auto operator--(int) -> NotNull<T> {
    invariant(value != nullptr);
    return value--;
  }
  constexpr auto operator+=(size_t offset) -> NotNull<T> & {
    invariant(value != nullptr);
    value += offset;
    return *this;
  }
  constexpr auto operator-=(size_t offset) -> NotNull<T> & {
    invariant(value != nullptr);
    value -= offset;
    return *this;
  }
  // constexpr auto operator==(const NotNull<T> &other) const -> bool {
  //   invariant(value != nullptr);
  //   return value == other.value;
  // }
  // constexpr auto operator!=(const NotNull<T> &other) const -> bool {
  //   invariant(value != nullptr);
  //   return value != other.value;
  // }
  // constexpr auto operator<(const NotNull<T> &other) const -> bool {
  //   invariant(value != nullptr);
  //   return value < other.value;
  // }
  // constexpr auto operator<=(const NotNull<T> &other) const -> bool {
  //   invariant(value != nullptr);
  //   return value <= other.value;
  // }
  // constexpr auto operator>(const NotNull<T> &other) const -> bool {
  //   invariant(value != nullptr);
  //   return value > other.value;
  // }
  // constexpr auto operator>=(const NotNull<T> &other) const -> bool {
  //   invariant(value != nullptr);
  //   return value >= other.value;
  // }
  constexpr auto operator-(const NotNull<T> &other) const -> ptrdiff_t {
    invariant(value != nullptr);
    return value - other.value;
  }
  [[nodiscard]] constexpr auto isAligned(size_t x) const -> bool {
    invariant(value != nullptr);
    return (reinterpret_cast<size_t>(value) % x) == 0;
  }
};
template <typename T> NotNull(T &) -> NotNull<T>;
template <typename T> NotNull(T *) -> NotNull<T *>;
static_assert(std::is_trivially_destructible_v<NotNull<size_t>>);
static_assert(std::is_trivially_copy_constructible_v<NotNull<size_t>>);
} // namespace poly::utils
