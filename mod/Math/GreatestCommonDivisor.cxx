#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifndef USE_MODULE
#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "SIMD/SIMD.cxx"
#include "Utilities/Invariant.cxx"
#else
export module GCD;

import Invariant;
import SIMD;
import STL;
#endif

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif
using utils::invariant;
constexpr auto constexpr_abs(std::signed_integral auto x) noexcept {
  return x < 0 ? -x : x;
}
template <ptrdiff_t W, std::integral T>
constexpr auto constexpr_abs(simd::Vec<W, T> x) noexcept {
  if constexpr (std::signed_integral<T>) return x < 0 ? -x : x;
  else return x;
}

constexpr auto gcd(int64_t x, int64_t y) -> int64_t {
  invariant(x != std::numeric_limits<int64_t>::min());
  invariant(y != std::numeric_limits<int64_t>::min());
  if (x == 0) return constexpr_abs(y);
  if (y == 0) return constexpr_abs(x);
  int64_t a = constexpr_abs(x), b = constexpr_abs(y);
  if ((a == 1) | (b == 1)) return 1;
  int64_t az = std::countr_zero(uint64_t(x)),
          bz = std::countr_zero(uint64_t(y));
  b >>= bz;
  int64_t k = std::min(az, bz);
  for (;;) {
    a >>= az;
    int64_t d = a - b;
    az = std::countr_zero(uint64_t(d));
    b = std::min(a, b);
    if (!d) break;
    a = constexpr_abs(d);
  }
  return b << k;
}
// TODO: add `Unroll` method, use in vectorized GCD?
template <ptrdiff_t W>
constexpr auto gcd(simd::Vec<W, int64_t> x,
                   simd::Vec<W, int64_t> y) -> simd::Vec<W, int64_t> {
  constexpr simd::Vec<W, int64_t> zero = {};
  constexpr simd::Vec<W, int64_t> one = zero + 1;
  constexpr simd::Vec<W, int64_t> invalid =
    zero + std::numeric_limits<int64_t>::min();

  invariant(!bool(simd::cmp::eq<W, int64_t>(x, invalid)));
  invariant(!bool(simd::cmp::eq<W, int64_t>(y, invalid)));
  simd::Vec<W, int64_t> b = constexpr_abs<W, int64_t>(y),
                        a = constexpr_abs<W, int64_t>(x),
                        bz = simd::crz<W, int64_t>(y),
                        az = simd::crz<W, int64_t>(x);
  auto on = ((a > 1) & (b > 1));
  simd::Vec<W, int64_t> offb = (x == 0 ? b : (y == 0 ? a : one));
  if (!bool(simd::cmp::ne<W, int64_t>(on, zero))) return offb;
  b = on ? (b >> bz) : offb;
  simd::Vec<W, int64_t> k = on ? (az > bz ? bz : az) : zero;
  for (;;) {
    a >>= az;
    simd::Vec<W, int64_t> d = a - b;
    az = simd::crz<W, int64_t>(d);
    b = (on & (a <= b)) ? a : b;
    on = on ? (d != zero) : on;
    if (!bool(simd::cmp::ne<W, int64_t>(on, zero))) break;
    a = constexpr_abs<W, int64_t>(d);
  }
  return b << k;
}
template <> constexpr auto gcd<1>(int64_t x, int64_t y) -> int64_t {
  return gcd(x, y);
}

template <ptrdiff_t W>
constexpr auto gcdreduce(simd::Vec<W, int64_t> v) -> int64_t {
  if constexpr (W != 2) {
    simd::Vec<W / 2, int64_t> a, b;
    for (ptrdiff_t w = 0; w < W / 2; ++w) {
      a[w] = v[w];
      b[w] = v[w + (W / 2)];
    }
    return gcdreduce<W / 2>(gcd<W / 2>(a, b));
  } else return gcd(v[0], v[1]);
}
template <> constexpr auto gcdreduce<1>(int64_t v) -> int64_t { return v; }

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
  T r_old = a, r = b;
  T s_old = 1, s = 0;
  T t_old = 0, t = 1;
  while (r) {
    T quotient = r_old / r, r_next = r_old - quotient * r,
      s_next = s_old - quotient * s, t_next = t_old - quotient * t;
    r_old = r, s_old = s, t_old = t;
    r = r_next, s = s_next, t = t_next;
  }
  // Solving for `t` at the end has 1 extra division, but lets us remove
  // the `t` updates in the loop:
  // T t = (b == 0) ? 0 : ((old_r - old_s * a) / b);
  // For now, I'll favor forgoing the division.
  return {r_old, s_old, t_old, copySign(t, a), copySign(s, b)};
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

} // namespace math
