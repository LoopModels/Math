#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifndef USE_MODULE
#include <array>
#include <cmath>

#else
export module Factor;
import STL;
#endif

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif

/// Tested from [1..32]
/// returns integer valued `double`s such that
/// auto [x,y] = lower_bound_factor(N, a)
/// x*y == N           // true
/// x <= a             // true
/// std::round(x) == x // true
/// std::round(y) == y // true
constexpr auto lower_bound_factor(double N, double x) -> std::array<double, 2> {
  if (x <= 1.0) return {1.0, N};
  if (x >= N) return {N, 1.0};
  double y = std::ceil(N / x);
  while (x * y != N) {
    x = std::floor(N / y);
    y = std::ceil(N / x);
  }
  return {x, y};
}

} // namespace math
