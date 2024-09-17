#ifdef USE_MODULE
module;
#else
#pragma once
#endif
#ifndef USE_MODULE
#include <concepts>
#include <limits>
#include <optional>
#include <utility>

#include "Utilities/Valid.cxx"
#else
export module Optional;
import Invariant;
import STL;
import Valid;
#endif

#ifdef USE_MODULE
export namespace utils {
#else
namespace utils {
#endif
/// Optional<T>
/// This type uses sentinels to indicate empty-optionals as an optimization for
/// certain types, oterhwise it falls back to `std::optional<T>`. Users may
/// provide `Optional` overloads for their types.
/// Specialized types: their sentinels indicating empty
/// T*: nullptr
/// std::signed_integral T: std::numeric_limits<T>::min();
/// std::unsigned_integral T: std::numeric_limits<T>::max();
/// T&: wraps a pointer, `nullptr` indicates empty.
template <typename T> struct Optional {
  std::optional<T> opt;
  [[nodiscard]] constexpr auto hasValue() const -> bool {
    return opt.has_value();
  }
  [[nodiscard]] constexpr auto getValue() -> T & {
    invariant(hasValue());
    return *opt;
  }
  constexpr explicit operator bool() const { return hasValue(); }
  constexpr auto operator->() -> T * { return &getValue(); }
  constexpr auto operator->() const -> const T * { return &getValue(); }
  constexpr Optional() = default;
  constexpr Optional(T value) : opt(std::move(value)) {}
  constexpr Optional(std::nullopt_t) {}
  constexpr auto operator*() -> T & { return getValue(); }
};

template <std::signed_integral T> struct Optional<T> {
  static constexpr T null = std::numeric_limits<T>::min();
  [[no_unique_address]] T value{null};
  [[nodiscard]] constexpr auto hasValue() const -> bool {
    return value != null;
  }
  [[nodiscard]] constexpr auto getValue() const -> T {
    invariant(hasValue());
    return value;
  }
  [[nodiscard]] constexpr auto operator*() const -> T { return getValue(); }
  constexpr auto operator->() -> T * { return &value; }
  constexpr auto operator->() const -> const T * { return &value; }
  constexpr explicit operator bool() const { return hasValue(); }
  constexpr Optional() = default;
  constexpr Optional(T v) : value(v) {}
  constexpr Optional(std::nullopt_t) {}
};
template <std::unsigned_integral T> struct Optional<T> {
  static constexpr T null = std::numeric_limits<T>::max();
  [[no_unique_address]] T value{null};
  [[nodiscard]] constexpr auto hasValue() const -> bool {
    return value != null;
  }
  [[nodiscard]] constexpr auto getValue() const -> T {
    invariant(hasValue());
    return value;
  }
  [[nodiscard]] constexpr auto operator*() const -> T { return getValue(); }
  constexpr auto operator->() -> T * { return &value; }
  constexpr auto operator->() const -> const T * { return &value; }
  constexpr explicit operator bool() const { return hasValue(); }
  constexpr Optional() = default;
  constexpr Optional(T v) : value(v) {}
  constexpr Optional(std::nullopt_t) {}
};

template <typename T> struct Optional<T &> {
  T *value{nullptr};
  [[nodiscard]] constexpr auto hasValue() const -> bool {
    return value != nullptr;
  }
  [[nodiscard]] constexpr auto getValue() -> T & {
    invariant(hasValue());
    return *value;
  }
  [[nodiscard]] constexpr auto operator*() -> T & { return getValue(); }
  constexpr explicit operator bool() const { return hasValue(); }
  constexpr auto operator->() -> T * { return value; }
  constexpr auto operator->() const -> const T * { return value; }
  constexpr Optional() = default;
  constexpr Optional(T &v) : value(&v) {}
  constexpr Optional(std::nullopt_t) {}
};

// template deduction guides
template <typename T> Optional(T) -> Optional<T>;
template <typename T> Optional(T *) -> Optional<T *>;
template <typename T> Optional(T &) -> Optional<T &>;

template <typename T> struct Optional<T *> {
  T *value{nullptr};
  [[nodiscard]] constexpr auto hasValue() const -> bool {
    return value != nullptr;
  }
  [[nodiscard]] constexpr auto getValue() -> T * {
    invariant(value != nullptr);
    return value;
  }
  [[nodiscard]] constexpr auto getValue() const -> const T * {
    invariant(value != nullptr);
    return value;
  }
  [[nodiscard]] constexpr auto operator*() -> T * { return value; }
  [[nodiscard]] constexpr auto operator*() const -> const T * { return value; }
  [[nodiscard]] constexpr explicit operator Valid<T>() {
    invariant(value != nullptr);
    return value;
  }
  constexpr explicit operator bool() const { return value != nullptr; }
  constexpr auto operator->() -> T * {
    invariant(value != nullptr);
    return value;
  }
  constexpr auto operator->() const -> const T * {
    invariant(value != nullptr);
    return value;
  }
  constexpr Optional() = default;
  constexpr Optional(T *v) : value{v} {}
  constexpr Optional(Valid<T> v) : value{v} {} // why would anyone want this?
  constexpr Optional(std::nullopt_t) {};
};
} // namespace utils
