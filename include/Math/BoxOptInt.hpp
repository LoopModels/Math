#pragma once
#include "Math/Array.hpp"
#include "Math/BoxOpt.hpp"
#include "Math/Constructors.hpp"
#include <cstdint>

namespace poly::math {

// recursive branch and bound
constexpr auto branchandbound(utils::Arena<> *alloc, BoxTransform &trf,
                              double lower, double upper, const auto &f)
  -> double {
  double newlower = minimize(alloc, trf, f);
  if (newlower >= upper) return upper; // bad solution
  auto [j, fl] = trf.maxFractionalComponent<>();
  if (j < 0) return lower; // was an integer solution
  // if j >= 0, then `x[trf.getInds()[j]]` is the largest fractional component
  // we'll thus split `trf` on it, creating two new boxes `trf1` and `trf2`
  // and two new vectors `x1` and `x2`
  // e.g., 3.4 -> two trfs with new bounds <=3 and >=4
  BoxTransform other{trf.branch(j, fl)};
  double tupper = trf.getRaw().empty()
                    ? f(trf.getLowerBounds())
                    : branchandbound(alloc, trf, lower, upper, f);
  double tlower = other.getRaw().empty()
                    ? f(other.getLowerBounds())
                    : branchandbound(alloc, other, lower, upper, f);
  if (tlower < tupper) {
  } else {
  }
  if (true) {
    // we have an integer solution

  } else {
    // we need to descend
  }
}
// set floor and ceil
// Assumes that integer floor of values is optimal
constexpr auto minimizeIntSol(utils::Arena<> *alloc, MutPtrVector<int32_t> r,
                              int32_t lb, int32_t ub, const auto &f) {
  // goal is to shrink all bounds such that lb==ub, i.e. we have all
  // integer solutions.
  unsigned N = r.size();
  poly::math::BoxTransform box{N, lb, ub};
  box.getRaw() << -3.0;
  double lower = minimize(alloc, box, f);
  double upper =
    f(Elementwise{[](auto &x) { return std::floor(x); }, box.transformed()});
  double opt = branchandbound(alloc, box, lower, upper, f);
  r << box.getLowerBounds();
  return opt;
}

} // namespace poly::math
