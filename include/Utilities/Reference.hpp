#pragma once
#include "Utilities/TypeCompression.hpp"

namespace poly::utils {
template <typename T> struct Reference {
  using U = utils::uncompressed_t<T>;
  T *t;
  constexpr operator U() const { return T::decompress(t); }
  // constexpr operator T &() const { return *t; }
  constexpr auto operator=(const U &u) -> Reference & {
    T::compress(t, u);
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
} // namespace poly::utils
