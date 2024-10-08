#ifdef USE_MODULE
module;
#else
#pragma once
#endif
#ifndef USE_MODULE
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "Alloc/Arena.cxx"
#include "Math/Array.cxx"
#include "Math/BoxOpt.cxx"
#else
export module BoxOptInt;

import Arena;
import Array;
import BoxOpt;
import STL;
#endif

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif

// recursive branch and bound
constexpr auto branch(alloc::Arena<> *alloc, BoxTransform &box, double upper,
                      double lower, const auto &f) -> double {
  auto [j, fl] = box.maxFractionalComponent<>();
  if (j < 0) return lower; // was an integer solution
  // if j >= 0, then `x[trf.getInds()[j]]` is the largest fractional component
  // we'll thus split `trf` on it, creating two new boxes `trf1` and `trf2`
  // and two new vectors `x1` and `x2`
  // e.g., 3.4 -> two trfs with new bounds <=3 and >=4
  BoxTransform bin{box.fork(j, fl)};
  upper = box.getRaw().empty() ? f(box.getLowerBounds())
                               : bound(alloc, box, upper, f);
  double oupper = bin.getRaw().empty() ? f(bin.getLowerBounds())
                                       : bound(alloc, bin, upper, f);
  if (oupper < upper) std::swap(box, bin);
  return oupper;
}
// recursive branch and bound
constexpr auto bound(alloc::Arena<> *alloc, BoxTransform &box, double upper,
                     const auto &f) -> double {
  double lower = minimize(alloc, box, f);
  if (lower >= upper) return upper; // bad solution
  return branch(alloc, box, upper, lower, f);
}
// set floor and ceil
// Assumes that integer floor of values is optimal
constexpr auto minimizeIntSol(
  alloc::Arena<> *alloc, MutPtrVector<int32_t> r, BoxTransform &box,
  const auto &f,
  double globalupper = std::numeric_limits<double>::infinity()) -> double {
  // goal is to shrink all bounds such that lb==ub, i.e. we have all
  // integer solutions.
  if (minimize(alloc, box, f) >= globalupper) return globalupper;
  // cache upper bound result
  {
    auto btf = box.transformed();
    for (ptrdiff_t i = 0, L = r.size(); i < L; ++i) r[i] = std::floor(btf[i]);
  }
  // TODO: simd floor support?
  // r << Elementwise{[](auto x) { return std::floor(x); }, box.transformed()};
  double upper = f(r);
  double opt = branch(alloc, box, upper, upper, f);
  if (opt >= upper) return upper;
  if (box.getRaw().empty()) {
    r << box.getLowerBounds();
  } else {
    // TODO: fix these
    r << box.transformed();
    // auto btf = box.transformed();
    // for (ptrdiff_t i = 0, L = r.size(); i < L; ++i) r[i] =
    // std::floor(btf[i]);
  }
  return opt;
}
constexpr auto minimizeIntSol(
  alloc::Arena<> *alloc, MutPtrVector<int32_t> r, int32_t lb, int32_t ub,
  const auto &f,
  double globalupper = std::numeric_limits<double>::infinity()) -> double {
  // goal is to shrink all bounds such that lb==ub, i.e. we have all
  // integer solutions.
  unsigned N = r.size();
  BoxTransform box{N, lb, ub};
  box.getRaw() << -3.0;
  return minimizeIntSol(alloc, r, box, f, globalupper);
}

} // namespace math
