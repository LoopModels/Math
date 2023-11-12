#pragma once
#include "Utilities/TypeCompression.hpp"

namespace poly::utils {
template <typename T> struct Reference {
  using U = utils::uncompressed_t<T>;
  static_assert(!std::same_as<U, T>);
  T *t;
  constexpr operator U() const { return T::decompress(t); }
  // constexpr operator T &() const { return *t; }
  constexpr auto operator=(const U &u) -> Reference & {
    T::compress(t, u);
    return *this;
  }
  constexpr auto operator=(const std::convertible_to<U> auto &u)
    -> Reference & {
    T::compress(t, u);
    return *this;
  }
  constexpr auto operator=(const T &u) -> Reference & {
    *t = u;
    return *this;
  }
  constexpr auto operator==(const U &u) const -> bool {
    return T::decompress(t) == u;
  }
  constexpr auto operator+=(const auto &x) -> Reference & {
    T::compress(t, T::decompress(t) + x);
    return *this;
  }
  constexpr auto operator-=(const auto &x) -> Reference & {
    T::compress(t, T::decompress(t) - x);
    return *this;
  }
  constexpr auto operator*=(const auto &x) -> Reference & {
    T::compress(t, T::decompress(t) * x);
    return *this;
  }
  constexpr auto operator/=(const auto &x) -> Reference & {
    T::compress(t, T::decompress(t) / x);
    return *this;
  }
  constexpr auto operator%=(const auto &x) -> Reference & {
    T::compress(t, T::decompress(t) % x);
    return *this;
  }
  constexpr auto operator<<=(const auto &x) -> Reference & {
    T::compress(t, T::decompress(t) << x);
    return *this;
  }
  constexpr auto operator>>=(const auto &x) -> Reference & {
    T::compress(t, T::decompress(t) >> x);
    return *this;
  }
  constexpr auto operator&=(const auto &x) -> Reference & {
    T::compress(t, T::decompress(t) & x);
    return *this;
  }
  constexpr auto operator^=(const auto &x) -> Reference & {
    T::compress(t, T::decompress(t) ^ x);
    return *this;
  }
  constexpr auto operator|=(const auto &x) -> Reference & {
    T::compress(t, T::decompress(t) | x);
    return *this;
  }
  constexpr auto operator[](auto i) -> decltype(auto) { return (*t)[i]; }
  constexpr auto operator[](auto i) const -> decltype(auto) { return (*t)[i]; }
  constexpr auto operator()(auto x) -> decltype(auto) { return (*t)(x); }
  constexpr auto operator()(auto x) const -> decltype(auto) { return (*t)(x); }
  constexpr auto operator()(auto x, auto y) -> decltype(auto) {
    return (*t)(x, y);
  }
  constexpr auto operator()(auto x, auto y) const -> decltype(auto) {
    return (*t)(x, y);
  }
  friend constexpr void swap(Reference x, Reference y) {
    U oldx = x;
    U oldy = y;
    x = oldy;
    y = oldx;
  }
  constexpr auto operator+(const auto &x) { return T::decompress(t) + x; }
  constexpr auto operator-(const auto &x) { return T::decompress(t) - x; }
  constexpr auto operator*(const auto &x) { return T::decompress(t) * x; }
  constexpr auto operator/(const auto &x) { return T::decompress(t) / x; }
  constexpr auto operator%(const auto &x) { return T::decompress(t) % x; }
  constexpr auto operator>>(const auto &x) { return T::decompress(t) >> x; }
  constexpr auto operator<<(const auto &x) { return T::decompress(t) << x; }
  constexpr auto operator&(const auto &x) { return T::decompress(t) & x; }
  constexpr auto operator^(const auto &x) { return T::decompress(t) ^ x; }
  constexpr auto operator|(const auto &x) { return T::decompress(t) | x; }

  friend constexpr auto operator+(const auto &x, Reference y) {
    return x + T::decompress(y.t);
  }
  friend constexpr auto operator-(const auto &x, Reference y) {
    return x - T::decompress(y.t);
  }
  friend constexpr auto operator*(const auto &x, Reference y) {
    return x * T::decompress(y.t);
  }
  friend constexpr auto operator/(const auto &x, Reference y) {
    return x / T::decompress(y.t);
  }
  friend constexpr auto operator%(const auto &x, Reference y) {
    return x % T::decompress(y.t);
  }
  friend constexpr auto operator>>(const auto &x, Reference y) {
    return x >> T::decompress(y.t);
  }
  friend constexpr auto operator<<(const auto &x, Reference y) {
    return x << T::decompress(y.t);
  }
  friend constexpr auto operator&(const auto &x, Reference y) {
    return x & T::decompress(y.t);
  }
  friend constexpr auto operator^(const auto &x, Reference y) {
    return x ^ T::decompress(y.t);
  }
  friend constexpr auto operator|(const auto &x, Reference y) {
    return x | T::decompress(y.t);
  }
};

template <Compressible T>
[[gnu::always_inline]] constexpr auto ref(T *p, ptrdiff_t i) -> Reference<T> {
  return Reference<T>{p + i};
}
template <typename T>
[[gnu::always_inline]] constexpr auto ref(T *p, ptrdiff_t i) -> T & {
  return p[i];
}
template <Compressible T>
[[gnu::always_inline]] constexpr auto ref(const T *p, ptrdiff_t i)
  -> uncompressed_t<T> {
  return T::decompress(p + i);
}
template <typename T>
[[gnu::always_inline]] constexpr auto ref(const T *p, ptrdiff_t i)
  -> const T & {
  return p[i];
}

} // namespace poly::utils
