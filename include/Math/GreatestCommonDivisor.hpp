#pragma once
#include "Utilities/Invariant.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
namespace poly::math {
using utils::invariant;
constexpr inline auto constexpr_abs(std::signed_integral auto x) noexcept {
  return x < 0 ? -x : x;
}

constexpr auto gcd(int64_t x, int64_t y) -> int64_t {
  if (x == 0) return constexpr_abs(y);
  if (y == 0) return constexpr_abs(x);
  invariant(x != std::numeric_limits<int64_t>::min());
  invariant(y != std::numeric_limits<int64_t>::min());
  int64_t a = constexpr_abs(x);
  int64_t b = constexpr_abs(y);
  if ((a == 1) | (b == 1)) return 1;
  int64_t az = std::countr_zero(uint64_t(x));
  int64_t bz = std::countr_zero(uint64_t(y));
  b >>= bz;
  int64_t k = std::min(az, bz);
  while (a) {
    a >>= az;
    int64_t d = a - b;
    az = std::countr_zero(uint64_t(d));
    b = std::min(a, b);
    a = constexpr_abs(d);
  }
  return b << k;
}
constexpr auto lcm(int64_t x, int64_t y) -> int64_t {
  int64_t ax = constexpr_abs(x);
  int64_t ay = constexpr_abs(y);
  if (ax == 1) return ay;
  if (ay == 1) return ax;
  if (ax == ay) return ax;
  return ax * (ay / gcd(ax, ay));
}

// inline auto copySign(double x, double s) -> double {
//   // TODO: c++23 makes std::copysign constexpr
//   return std::copysign(x, s);
// }
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
template <std::integral I> constexpr auto copySign(I x, I s) -> I {
  if (s >= 0) return constexpr_abs(x);
  return -constexpr_abs(x);
}

// https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
template <std::integral T>
constexpr auto dgcdx(T a, T b)
  -> std::array<T, 5> { // NOLINT(bugprone-easily-swappable-parameters)
  T rOld = a, r = b;
  T sOld = 1, s = 0;
  T tOld = 0, t = 1;
  while (r) {
    T quotient = rOld / r;
    rOld -= quotient * r;
    sOld -= quotient * s;
    tOld -= quotient * t;
    std::swap(r, rOld);
    std::swap(s, sOld);
    std::swap(t, tOld);
  }
  // Solving for `t` at the end has 1 extra division, but lets us remove
  // the `t` updates in the loop:
  // T t = (b == 0) ? 0 : ((old_r - old_s * a) / b);
  // For now, I'll favor forgoing the division.
  return {rOld, sOld, tOld, copySign(t, a), copySign(s, b)};
}
template <
  std::integral T> // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
constexpr auto gcdx(T a, T b) -> std::array<T, 3> {
  auto [g, x, y, t, s] = dgcdx(a, b);
  return {g, x, y};
}

/// divgcd(x, y) = (x / gcd(x, y), y / gcd(x, y))
constexpr auto divgcd(int64_t a, int64_t b) -> std::array<int64_t, 2> {
  auto [g, x, y, t, s] = dgcdx(a, b);
  return {t, s};
}
} // namespace poly::math
