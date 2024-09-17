#ifdef USE_MODULE
module;
#else
#pragma once
#endif
#ifndef USE_MODULE
#include <array>
#include <cstdint>
#include <optional>

#include "Math/GreatestCommonDivisor.cxx"
#else
export module LinearDiophantine;

import GCD;
import STL;
#endif

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline auto linearDiophantine(int64_t c, int64_t a, int64_t b)
  -> std::optional<std::array<int64_t, 2>> {
  if (c == 0) return {{int64_t(0), int64_t(0)}};
  if ((a | b) == 0) return {};
  auto [g, x, y] = gcdx(a, b);
  int64_t cDivG = g == 1 ? c : c / g;
  // g = a*x + b*y;
  if (cDivG * g == c) return {{x * cDivG, y * cDivG}};
  return {};
}

// d = a*x; x = d/a
inline auto linearDiophantine(int64_t d, std::array<int64_t, 1> a)
  -> std::optional<std::array<int64_t, 1>> {
  int64_t a0 = a[0];
  if (d == 0) return {{int64_t(0)}};
  if (a0) {
    int64_t x = d / a0;
    if (a0 * x == d) return {{x}};
  }
  return {};
}
// d = a[0]*x + a[1]*y;
inline auto linearDiophantine(int64_t d, std::array<int64_t, 2> a)
  -> std::optional<std::array<int64_t, 2>> {
  return linearDiophantine(d, a[0], a[1]);
}

template <size_t N>
inline auto linearDiophantine(int64_t d, std::array<int64_t, N> a)
  -> std::optional<std::array<int64_t, N>> {
  std::array<int64_t, N - 2> aRem;
  std::copy(a.begin() + 2, a.end(), aRem.begin());
  if ((a[0] | a[1]) == 0) {
    auto opt = linearDiophantine(d, aRem);
    if (!opt) return {};
    a[0] = 0;
    a[1] = 0;
    std::copy(opt->begin(), opt->end(), a.begin() + 2);
    return a;
  }
  int64_t q = gcd(a[0], a[1]);
  std::array<int64_t, N - 1> aRem1;
  aRem1[0] = q;
  std::copy(aRem.begin(), aRem.end(), aRem1.begin() + 1);
  // d == q*((a/q)*x + (b/q)*y) + ... == q*w + ...
  // solve the rest
  auto dioDqc = linearDiophantine(d, aRem1);
  if (!dioDqc) return {};
  auto t = *dioDqc;
  int64_t w = t[0];
  // w == ((a0/q)*x + (a1/q)*y)
  auto o = linearDiophantine(w, a[0] / q, a[1] / q);
  if (!o) return {};
  std::copy(o->begin(), o->end(), a.begin());
  std::copy(t.begin() + 1, t.end(), a.begin() + 2);
  return a;
}
} // namespace math
