#pragma once
#include "Utilities/TypeCompression.hpp"

namespace poly::utils {
template <typename T> struct Reference {
  using C = utils::compressed_t<T>;
  static_assert(!std::same_as<C, T>);
  C *c;
  constexpr operator T() const { return T::decompress(c); }
  // constexpr operator T &() const { return *t; }
  constexpr auto operator=(const T &t) -> Reference & {
    t.compress(c);
    return *this;
  }
  constexpr auto operator=(const C &x) -> Reference & {
    // Probably shouldn't be needed?
    // TODO: try removing this method
    *c = x;
    return *this;
  }
  constexpr auto operator==(const T &t) const -> bool {
    return T::decompress(c) == t;
  }
  constexpr auto operator+=(const auto &x) -> Reference & {
    T y{*this};
    y += x;
    y.compress(c);
    return *this;
  }
  constexpr auto operator-=(const auto &x) -> Reference & {
    T y{*this};
    y -= x;
    y.compress(c);
    return *this;
  }
  constexpr auto operator*=(const auto &x) -> Reference & {
    T y{*this};
    y *= x;
    y.compress(c);
    return *this;
  }
  constexpr auto operator/=(const auto &x) -> Reference & {
    T y{*this};
    y /= x;
    y.compress(c);
    return *this;
  }
  constexpr auto operator%=(const auto &x) -> Reference & {
    T y{*this};
    y %= x;
    y.compress(c);
    return *this;
  }
  constexpr auto operator<<=(const auto &x) -> Reference & {
    T y{*this};
    y <<= x;
    y.compress(c);
    return *this;
  }
  constexpr auto operator>>=(const auto &x) -> Reference & {
    T y{*this};
    y >>= x;
    y.compress(c);
    return *this;
  }
  constexpr auto operator&=(const auto &x) -> Reference & {
    T y{*this};
    y &= x;
    y.compress(c);
    return *this;
  }
  constexpr auto operator^=(const auto &x) -> Reference & {
    T y{*this};
    y ^= x;
    y.compress(c);
    return *this;
  }
  constexpr auto operator|=(const auto &x) -> Reference & {
    T y{*this};
    y |= x;
    y.compress(c);
    return *this;
  }

  constexpr auto operator[](auto i) -> decltype(auto) {
    return c->operator[](i);
  }
  constexpr auto operator[](auto i) const -> decltype(auto) {
    return c->operator[](i);
  }
  constexpr auto operator[](auto i, auto j) -> decltype(auto) {
    return c->operator[](i, j);
  }
  constexpr auto operator[](auto i, auto j) const -> decltype(auto) {
    return c->operator[](i, j);
  }

  friend constexpr void swap(Reference x, Reference y) {
    std::swap(*x.c, *y.c);
  }
  // TODO:: are these really needed / can we rely on implicit conversion?
  constexpr auto operator+(const auto &x) { return T::decompress(c) + x; }
  constexpr auto operator-(const auto &x) { return T::decompress(c) - x; }
  constexpr auto operator*(const auto &x) { return T::decompress(c) * x; }
  constexpr auto operator/(const auto &x) { return T::decompress(c) / x; }
  constexpr auto operator%(const auto &x) { return T::decompress(c) % x; }
  constexpr auto operator>>(const auto &x) { return T::decompress(c) >> x; }
  constexpr auto operator<<(const auto &x) { return T::decompress(c) << x; }
  constexpr auto operator&(const auto &x) { return T::decompress(c) & x; }
  constexpr auto operator^(const auto &x) { return T::decompress(c) ^ x; }
  constexpr auto operator|(const auto &x) { return T::decompress(c) | x; }

  friend constexpr auto operator+(const auto &x, Reference y) {
    return x + T::decompress(y.c);
  }
  friend constexpr auto operator-(const auto &x, Reference y) {
    return x - T::decompress(y.c);
  }
  friend constexpr auto operator*(const auto &x, Reference y) {
    return x * T::decompress(y.c);
  }
  friend constexpr auto operator/(const auto &x, Reference y) {
    return x / T::decompress(y.c);
  }
  friend constexpr auto operator%(const auto &x, Reference y) {
    return x % T::decompress(y.c);
  }
  friend constexpr auto operator>>(const auto &x, Reference y) {
    return x >> T::decompress(y.c);
  }
  friend constexpr auto operator<<(const auto &x, Reference y) {
    return x << T::decompress(y.c);
  }
  friend constexpr auto operator&(const auto &x, Reference y) {
    return x & T::decompress(y.c);
  }
  friend constexpr auto operator^(const auto &x, Reference y) {
    return x ^ T::decompress(y.c);
  }
  friend constexpr auto operator|(const auto &x, Reference y) {
    return x | T::decompress(y.c);
  }
};

template <typename T>
[[gnu::always_inline]] constexpr auto ref(T *p, ptrdiff_t i) -> T & {
  return p[i];
}
template <typename T>
[[gnu::always_inline]] constexpr auto ref(const T *p, ptrdiff_t i)
  -> const T & {
  return p[i];
}
template <Compressible T>
[[gnu::always_inline]] constexpr auto ref(utils::compressed_t<T> *p,
                                          ptrdiff_t i) -> Reference<T> {
  return Reference<T>{p + i};
}

template <Compressible T>
[[gnu::always_inline]] constexpr auto ref(const utils::compressed_t<T> *p,
                                          ptrdiff_t i) -> T {
  return T::decompress(p + i);
}

} // namespace poly::utils
