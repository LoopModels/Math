#pragma once
#include "Math/Array.hpp"
#include "Math/BoxOpt.hpp"
#include "Math/Constructors.hpp"
#include <cstdint>

namespace poly::math {

// recursive branch and bound
constexpr auto branchandbound(utils::Arena<> *alloc, MutPtrVector<double> x,
                              BoxTransformView trf, double lower, double upper,
                              const auto &f) -> double {}
// set floor and ceil
// Assumes that integer floor of values is optimal
constexpr auto minimizeIntSol(utils::Arena<> *alloc, MutPtrVector<int32_t> r,
                              int32_t lb, int32_t ub, const auto &f) {
  // goal is to shrink all bounds such that lb==ub, i.e. we have all
  // integer solutions.
  unsigned N = r.size();
  auto s = alloc->scope();
  MutPtrVector<double> x{vector<double>(alloc, N)};
  // MutDensePtrMatrix<double> X{matrix<double>(alloc, Row{2}, Col{N})};
  poly::math::BoxTransform box{N, lb, ub};
  double lower = minimize(alloc, x, box, f);
  double upper = f(Elementwise{[](auto &x) { return std::floor(x); },
                               BoxTransformVector{x, box}});
  double opt = branchandbound(alloc, x, box, lower, upper, f);
  r << box.getLowerBounds();
  return opt;
}

} // namespace poly::math
