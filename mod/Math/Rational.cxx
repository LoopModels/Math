#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifndef USE_MODULE
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <ostream>
#include <type_traits>

#include "Math/GreatestCommonDivisor.cxx"
#include "Utilities/TypeCompression.cxx"
#include "Utilities/Widen.cxx"
#else
export module Rational;

import GCD;
import Widen;
import STL;
import TypeCompression;
#endif

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif

struct Rational {
  [[no_unique_address]] int64_t numerator{0};
  [[no_unique_address]] int64_t denominator{1};
  // should be invariant that denominator >= 0
  constexpr Rational() = default;
  constexpr Rational(int64_t coef) : numerator(coef) {};
  constexpr Rational(int coef) : numerator(coef) {};
  constexpr Rational(int64_t n, int64_t d)
    : numerator(d > 0 ? n : -n), denominator(n ? (d > 0 ? d : -d) : 1) {}
  static constexpr auto create(int64_t n, int64_t d) -> Rational {
    if (n) {
      int64_t sign = 2 * (d > 0) - 1;
      int64_t g = gcd(n, d);
      n *= sign;
      d *= sign;
      if (g != 1) {
        n /= g;
        d /= g;
      }
      return Rational{n, d};
    }
    return Rational{0, 1};
  }
  static constexpr auto createPositiveDenominator(int64_t n,
                                                  int64_t d) -> Rational {
    if (n) {
      int64_t g = gcd(n, d);
      if (g != 1) {
        n /= g;
        d /= g;
      }
      return Rational{n, d};
    }
    return Rational{0, 1};
  }

  [[nodiscard]] constexpr auto
  safeAdd(Rational y) const -> std::optional<Rational> {
    auto [xd, yd] = divgcd(denominator, y.denominator);
    int64_t a, b, n, d;
    bool o1 = __builtin_smull_overflow(numerator, yd, &a);
    bool o2 = __builtin_smull_overflow(y.numerator, xd, &b);
    bool o3 = __builtin_smull_overflow(denominator, yd, &d);
    bool o4 = __builtin_saddl_overflow(a, b, &n);
    if ((o1 | o2) | (o3 | o4)) return {};
    if (!n) return Rational{0, 1};
    auto [nn, nd] = divgcd(n, d);
    return Rational{nn, nd};
  }
  constexpr auto operator+(Rational y) const -> Rational {
    return *safeAdd(y); // NOLINT(bugprone-unchecked-optional-access)
  }
  constexpr auto operator+=(Rational y) -> Rational & {
    std::optional<Rational> a = *this + y;
    invariant(a.has_value());
    *this = *a;
    return *this;
  }
  [[nodiscard]] constexpr auto
  safeSub(Rational y) const -> std::optional<Rational> {
    auto [xd, yd] = divgcd(denominator, y.denominator);
    int64_t a, b, n, d;
    bool o1 = __builtin_smull_overflow(numerator, yd, &a);
    bool o2 = __builtin_smull_overflow(y.numerator, xd, &b);
    bool o3 = __builtin_smull_overflow(denominator, yd, &d);
    bool o4 = __builtin_ssubl_overflow(a, b, &n);
    if ((o1 | o2) | (o3 | o4)) return {};
    if (!n) return Rational{0, 1};
    auto [nn, nd] = divgcd(n, d);
    return Rational{nn, nd};
  }
  constexpr auto operator-(Rational y) const -> Rational {
    return *safeSub(y); // NOLINT(bugprone-unchecked-optional-access)
  }
  constexpr auto operator-=(Rational y) -> Rational & {
    std::optional<Rational> a = *this - y;
    invariant(a.has_value());
    *this = *a;
    return *this;
  }
  [[nodiscard]] constexpr auto
  safeMul(int64_t y) const -> std::optional<Rational> {
    auto [xd, yn] = divgcd(denominator, y);
    int64_t n;
    if (__builtin_mul_overflow(numerator, yn, &n)) return {};
    return Rational{n, xd};
  }
  [[nodiscard]] constexpr auto
  safeMul(Rational y) const -> std::optional<Rational> {
    if ((numerator == 0) | (y.numerator == 0)) return Rational{0, 1};
    auto [xn, yd] = divgcd(numerator, y.denominator);
    auto [xd, yn] = divgcd(denominator, y.numerator);
    int64_t n, d;
    bool o1 = __builtin_smull_overflow(xn, yn, &n);
    bool o2 = __builtin_smull_overflow(xd, yd, &d);
    if (o1 | o2) return {};
    return Rational{n, d};
  }
  constexpr auto operator*(Rational y) const -> Rational {
    return *safeMul(y); // NOLINT(bugprone-unchecked-optional-access)
  }
  constexpr auto operator*=(Rational y) -> Rational & {
    if ((numerator != 0) & (y.numerator != 0)) {
      auto [xn, yd] = divgcd(numerator, y.denominator);
      auto [xd, yn] = divgcd(denominator, y.numerator);
      numerator = xn * yn;
      denominator = xd * yd;
    } else {
      numerator = 0;
      denominator = 1;
    }
    return *this;
  }
  [[nodiscard]] constexpr auto inv() const -> Rational {
    if (numerator > 0) return Rational{denominator, numerator};
    invariant(denominator != std::numeric_limits<int64_t>::min());
    invariant(numerator != 0);
    return Rational{-denominator, -numerator};
  }
  [[nodiscard]] constexpr auto
  safeDiv(Rational y) const -> std::optional<Rational> {
    return (*this) * y.inv();
  }
  constexpr auto operator/(Rational y) const -> Rational {
    return *safeDiv(y); // NOLINT(bugprone-unchecked-optional-access)
  }
  // *this -= a*b
  constexpr auto fnmadd(Rational a, Rational b) -> bool {
    if (std::optional<Rational> ab = a.safeMul(b)) {
      if (std::optional<Rational> c = safeSub(*ab)) {
        *this = *c;
        return false;
      }
    }
    return true;
  }
  constexpr auto div(Rational a) -> bool {
    if (std::optional<Rational> d = safeDiv(a)) {
      *this = *d;
      return false;
    }
    return true;
  }
  // Rational operator/=(Rational y) { return (*this) *= y.inv(); }
  constexpr operator double() const {
    return double(numerator) / double(denominator);
  }

  constexpr auto operator==(Rational y) const -> bool {
    return (numerator == y.numerator) & (denominator == y.denominator);
  }
  constexpr auto operator!=(Rational y) const -> bool {
    return (numerator != y.numerator) | (denominator != y.denominator);
  }
  [[nodiscard]] constexpr auto isEqual(int64_t y) const -> bool {
    if (denominator == 1) return (numerator == y);
    if (denominator == -1) return (numerator == -y);
    return false;
  }
  constexpr auto operator==(int y) const -> bool { return isEqual(y); }
  constexpr auto operator==(int64_t y) const -> bool { return isEqual(y); }
  constexpr auto operator!=(int y) const -> bool { return !isEqual(y); }
  constexpr auto operator!=(int64_t y) const -> bool { return !isEqual(y); }
  constexpr auto operator<(Rational y) const -> bool {
    return (utils::widen(numerator) * utils::widen(y.denominator)) <
           (utils::widen(y.numerator) * utils::widen(denominator));
  }
  constexpr auto operator<=(Rational y) const -> bool {
    return (utils::widen(numerator) * utils::widen(y.denominator)) <=
           (utils::widen(y.numerator) * utils::widen(denominator));
  }
  constexpr auto operator>(Rational y) const -> bool {
    return (utils::widen(numerator) * utils::widen(y.denominator)) >
           (utils::widen(y.numerator) * utils::widen(denominator));
  }
  constexpr auto operator>=(Rational y) const -> bool {
    return (utils::widen(numerator) * utils::widen(y.denominator)) >=
           (utils::widen(y.numerator) * utils::widen(denominator));
  }
  constexpr auto operator>=(int y) const -> bool {
    return *this >= Rational(y);
  }
  [[nodiscard]] constexpr auto isInteger() const -> bool {
    return denominator == 1;
  }
  constexpr void negate() { numerator = -numerator; }
  constexpr explicit operator bool() const { return numerator != 0; }

#ifndef NDEBUG
  [[gnu::used]] void dump() const { std::cout << *this << "\n"; }
#endif

private:
  friend constexpr auto operator+(Rational x, int64_t y) -> Rational {
    return Rational{x.numerator + y * x.denominator, x.denominator};
  }
  friend constexpr auto operator+(int64_t y, Rational x) -> Rational {
    return x + y;
  }
  friend constexpr auto operator-(Rational x, int64_t y) -> Rational {
    return Rational{x.numerator - y * x.denominator, x.denominator};
  }
  friend constexpr auto operator-(int64_t y, Rational x) -> Rational {
    return Rational{y * x.denominator - x.numerator, x.denominator};
  }
  friend constexpr auto operator*(Rational x, int64_t y) -> Rational {
    return *x.safeMul(y); // NOLINT(bugprone-unchecked-optional-access)
  }
  friend constexpr auto operator*(int64_t y, Rational x) -> Rational {
    return x * y;
  }
  friend constexpr auto operator==(int64_t x, Rational y) -> bool {
    return y == x;
  }
  friend auto operator<<(std::ostream &os,
                         const Rational &x) -> std::ostream & {
    os << x.numerator;
    if (x.denominator != 1) os << " // " << x.denominator;
    return os;
  }
};
constexpr auto gcd(Rational x, Rational y) -> std::optional<Rational> {
  return Rational{gcd(x.numerator, y.numerator),
                  lcm(x.denominator, y.denominator)};
}
} // namespace math
#ifdef USE_MODULE
export {
#endif
  template <> struct std::common_type<math::Rational, int> {
    using type = math::Rational;
  };
  template <> struct std::common_type<int, math::Rational> {
    using type = math::Rational;
  };
  template <> struct std::common_type<math::Rational, int64_t> {
    using type = math::Rational;
  };
  template <> struct std::common_type<int64_t, math::Rational> {
    using type = math::Rational;
  };
#ifdef USE_MODULE
} // namespace std
#endif
static_assert(
  std::same_as<std::common_type_t<math::Rational, int>, math::Rational>);
static_assert(
  std::same_as<std::common_type_t<int, math::Rational>, math::Rational>);
static_assert(
  std::same_as<utils::decompressed_t<math::Rational>, math::Rational>);
