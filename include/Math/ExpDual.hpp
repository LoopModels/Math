#pragma once
#include "Math/Dual.hpp"
#include "Math/Exp.hpp"
namespace poly::math {

template <int l = 8> constexpr auto smax(auto x, auto y, auto z) {
  double m = std::max(std::max(value(x), value(y)), value(z));
  constexpr double f = l, i = 1 / f;
  return m + log(exp(f * (x - m)) + exp(f * (y - m)) + exp(f * (z - m))) * i;
}

} // namespace poly::math
