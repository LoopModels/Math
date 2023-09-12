#pragma once
#include "Math/Array.hpp"
#include "Math/Dual.hpp"
#include "Math/Exp.hpp"
#include "Math/LinearAlgebra.hpp"
#include "Math/Math.hpp"
#include "Utilities/Allocators.hpp"
#include <cstddef>
#include <cstdint>

namespace poly::math {

constexpr double EXTREME = 8.0;

class BoxTransformView {
public:
  constexpr BoxTransformView(double *f, int32_t *i, unsigned ntotal)
    : f64{f}, i32{i}, Ntotal{ntotal} {}
  [[nodiscard]] constexpr auto size() const -> unsigned { return Ntotal; }
  constexpr auto operator()(const AbstractVector auto &x, ptrdiff_t i) const
    -> utils::eltype_t<decltype(x)> {
    invariant(i < Ntotal);
    int j = getInds()[i];
    double off = offs()[i];
    if (j < 0) return off;
    return scales()[i] * sigmoid(x[j]) + off;
  }
  [[nodiscard]] constexpr auto view() const -> BoxTransformView {
    return *this;
  }
  [[nodiscard]] constexpr auto getLowerBounds() -> MutPtrVector<int32_t> {
    return {i32 + Ntotal, Ntotal};
  }
  [[nodiscard]] constexpr auto getUpperBounds() -> MutPtrVector<int32_t> {
    return {i32 + ptrdiff_t(2) * Ntotal, Ntotal};
  }
  // gives max fractional ind on the transformed scale
  template <bool Relative = true>
  constexpr auto maxFractionalComponent(MutPtrVector<double> x) -> ptrdiff_t {
    double max = 0.0;
    ptrdiff_t k = -1;
    for (ptrdiff_t i = 0, j = 0; i < Ntotal; ++i) {
      double s = scales()[i];
      if (s == 0.0) continue;
      double a = s * sigmoid(x[j++]) + offs()[i];
      double d = a - std::floor(a);
      d = (d > 0.5) ? 1.0 - d : d;
      if constexpr (Relative) d /= a;
      if (d <= max) continue;
      max = d;
      k = i;
    }
    return k;
  }

protected:
  /// Ntotal offsets
  /// Ntotal scales
  double *f64;
  /// Ntotal indices
  /// Ntotal lower bounds
  /// Ntotal upper bounds
  int32_t *i32;
  unsigned Ntotal;

  [[nodiscard]] constexpr auto getInds() const -> PtrVector<int32_t> {
    return {i32, Ntotal};
  }
  [[nodiscard]] constexpr auto getLowerBounds() const -> PtrVector<int32_t> {
    return {i32 + Ntotal, Ntotal};
  }
  [[nodiscard]] constexpr auto getUpperBounds() const -> PtrVector<int32_t> {
    return {i32 + ptrdiff_t(2) * Ntotal, Ntotal};
  }
  [[nodiscard]] constexpr auto offs() const -> PtrVector<double> {
    return {f64, Ntotal};
  }
  [[nodiscard]] constexpr auto scales() const -> PtrVector<double> {
    return {f64 + Ntotal, Ntotal};
  }
  [[nodiscard]] constexpr auto getInds() -> MutPtrVector<int32_t> {
    return {i32, Ntotal};
  }
  [[nodiscard]] constexpr auto offs() -> MutPtrVector<double> {
    return {f64, Ntotal};
  }
  [[nodiscard]] constexpr auto scales() -> MutPtrVector<double> {
    return {f64 + Ntotal, Ntotal};
  }
  static constexpr auto scaleOff(int32_t lb, int32_t ub)
    -> std::pair<double, double> {
#if __cplusplus >= 202202L
    // constexpr std::fma requires c++23
    constexpr double slb = sigmoid(-EXTREME);
    constexpr double sub = sigmoid(EXTREME);
    constexpr double scale = 1.0 / (sub - slb);
#else
    double slb = sigmoid(-EXTREME);
    double sub = sigmoid(EXTREME);
    double scale = 1.0 / (sub - slb);
#endif
    double s = scale * (ub - lb), o = lb - slb * s;
    return {s, o};
  }
};
class BoxTransform : public BoxTransformView {

public:
  constexpr BoxTransform(unsigned ntotal, int32_t lb, int32_t ub)
    : BoxTransformView{new double[long(2) * ntotal],
                       new int32_t[long(3) * ntotal], ntotal} {
    invariant(lb < ub);
    auto [s, o] = scaleOff(lb, ub);
    for (int32_t i = 0; i < int32_t(ntotal); ++i) {
      getInds()[i] = i;
      getLowerBounds()[i] = lb;
      getUpperBounds()[i] = ub;
      scales()[i] = s;
      offs()[i] = o;
    }
  }
  constexpr void increaseLowerBound(MutPtrVector<double> &untrf, ptrdiff_t idx,
                                    int32_t lb) {
    invariant(idx < Ntotal);
    invariant(lb > getLowerBounds()[idx]);
    int32_t ub = getUpperBounds()[idx];
    invariant(lb <= ub);
    getLowerBounds()[idx] = lb;
    double newScale, newOff;
    if (lb == ub) {
      untrf.erase(getInds()[idx]);
      getInds()[idx] = -1;
      // we remove a fixed, so we must now decrement all following inds
      for (ptrdiff_t i = idx; ++i < Ntotal;) --getInds()[i];
      newScale = 0.0;
      newOff = lb;
    } else {
      untrf[getInds()[idx]] = -EXTREME;
      std::tie(newScale, newOff) = scaleOff(lb, ub);
    }
    scales()[idx] = newScale;
    offs()[idx] = newOff;
  }
  constexpr void decreaseUpperBound(MutPtrVector<double> &untrf, ptrdiff_t idx,
                                    int32_t ub) {
    invariant(idx < Ntotal);
    invariant(ub < getUpperBounds()[idx]);
    int32_t lb = getLowerBounds()[idx];
    invariant(lb <= ub);
    getUpperBounds()[idx] = ub;
    double newScale, newOff;
    if (lb == ub) {
      untrf.erase(getInds()[idx]);
      getInds()[idx] = -1;
      // we remove a fixed, so we must now decrement all following inds
      for (ptrdiff_t i = idx; ++i < Ntotal;) --getInds()[i];
      newScale = 0.0;
      newOff = lb;
    } else {
      untrf[getInds()[idx]] = EXTREME;
      std::tie(newScale, newOff) = scaleOff(lb, ub);
    }
    scales()[idx] = newScale;
    offs()[idx] = newOff;
  }
  constexpr ~BoxTransform() {
    if (f64 == nullptr) {
      invariant(i32 == nullptr);
      return;
    }
    delete[] f64;
    delete[] i32;
  }
  constexpr BoxTransform(BoxTransform &&other) noexcept
    : BoxTransformView{other.f64, other.i32, other.Ntotal} {
    other.f64 = nullptr;
    other.i32 = nullptr;
  }
  constexpr BoxTransform(const BoxTransform &other)
    : BoxTransformView{new double[long(2) * other.Ntotal],
                       new int32_t[long(3) * other.Ntotal], other.Ntotal} {
    std::copy_n(other.f64, 2 * Ntotal, f64);
    std::copy_n(other.i32, 3 * Ntotal, i32);
  }
};

template <AbstractVector V> struct BoxTransformVector {
  using value_type = utils::eltype_t<V>;
  static_assert(Trivial<V>);
  V v;
  BoxTransformView btv;

  [[nodiscard]] constexpr auto size() const -> ptrdiff_t { return btv.size(); }
  constexpr auto operator[](ptrdiff_t i) const -> value_type {
    return btv(v, i);
  }
  [[nodiscard]] constexpr auto view() const -> BoxTransformVector {
    return *this;
  }
};
template <AbstractVector V>
BoxTransformVector(V, BoxTransformView) -> BoxTransformVector<V>;

static_assert(AbstractVector<BoxTransformVector<PtrVector<double>>>);

template <typename F> struct BoxCall {
  [[no_unique_address]] F f;
  BoxTransformView transform;
  constexpr auto operator()(const AbstractVector auto &x) const
    -> utils::eltype_t<decltype(x)> {
    return f(BoxTransformVector{x.view(), transform});
  }
};
template <typename T> struct IsBoxCall {
  static constexpr bool value = false;
};
template <typename F> struct IsBoxCall<BoxCall<F>> {
  static constexpr bool value = true;
};

constexpr auto minimize(utils::Arena<> *alloc, MutPtrVector<double> x,
                        const auto &f) -> double {
  constexpr bool constrained =
    IsBoxCall<std::remove_cvref_t<decltype(f)>>::value;
  constexpr double tol = 1e-8;
  constexpr double tol2 = tol * tol;
  constexpr double c = 0.5;
  constexpr double tau = 0.5;
  double alpha = 0.25, fx;
  ptrdiff_t L = x.size();
  auto scope = alloc->scope();
  auto *data = alloc->allocate<double>(2 * L);
  MutPtrVector<double> xnew{data, L}, dir{data + L, L}, xcur{x};
  HessianResultCore hr{alloc, unsigned(L)};
  for (ptrdiff_t n = 0; n < 1000; ++n) {
    fx = hessian(hr, xcur, f);
    if (hr.gradient().norm2() < tol2) break;
    LDL::ldiv<true>(hr.hessian(), dir, hr.gradient());
    if constexpr (!constrained)
      if (dir.norm2() < tol2) break;
    double t = 0.0;
    for (ptrdiff_t i = 0; i < L; ++i) {
      // TODO: 0 clamped dirs
      t += hr.gradient()[i] * dir[i];
      double xi = xcur[i] - alpha * dir[i];
      if constexpr (constrained) {
        double xinew = std::clamp(xi, -EXTREME, EXTREME);
        if (xinew != xi) dir[i] = 0.0;
        xnew[i] = xinew;
      } else xnew[i] = xi;
    }
    if constexpr (constrained)
      if (dir.norm2() < tol2) break;
    t *= c;
    double fxnew = f(xnew);
    bool cond = (fx - fxnew) >= alpha * t;
    bool dobreak = false;
    if (cond) {
      // success; we try to grow alpha, if we're not already optimal
      if (((fx - fxnew) <= tol) || (norm2(xcur - xnew) <= tol2)) {
        std::swap(xcur, xnew);
        break;
      }
      double alphaorig = alpha;
      for (;;) {
        double alphanew = alpha * (1 / tau);
        // xnew == xcur - alphaorig * dir;
        // xcur == xnew + alphaorig * dir;
        // xtmp == xcur - alphanew * dir;
        // xtmp == xnew + (alphaorig - alphanew) * dir;
        // thus, we can use `xcur` as an `xtmp` buffer
        double a = alphaorig - alphanew;
        for (ptrdiff_t i = 0; i < L; ++i) {
          double xi = xnew[i] + a * dir[i];
          if constexpr (constrained) xi = std::clamp(xi, -EXTREME, EXTREME);
          xcur[i] = xi;
        }
        double fxtmp = f(xcur);
        if ((fx - fxtmp) < alphanew * t) break;
        std::swap(xcur, xnew); // we keep xcur as new best
        alpha = alphanew;
        fxnew = fxtmp;
        dobreak = (fx - fxtmp) <= tol;
        if (dobreak) break;
      }
    } else {
      // failure, we shrink alpha
      for (;;) {
        alpha *= tau;
        for (ptrdiff_t i = 0; i < L; ++i) {
          double xi = xcur[i] - alpha * dir[i];
          if constexpr (constrained) xi = std::clamp(xi, -EXTREME, EXTREME);
          xnew[i] = xi;
        }
        fxnew = f(xnew);
        dobreak = norm2(xcur - xnew) <= tol2;
        if (dobreak) break;
        if ((fx - fxnew) < alpha * t) continue;
        dobreak = (fx - fxnew) <= tol;
        break;
      }
    }
    fx = fxnew;
    std::swap(xcur, xnew);
    if (dobreak) break;
  }
  if (x.data() != xcur.data()) x << xcur;
  return fx;
}

constexpr auto minimize(utils::Arena<> *alloc, PtrVector<double> x,
                        BoxTransformView trf, const auto &f) -> double {
  return minimize(alloc, x, BoxCall<decltype(f)>{f, trf});
}

} // namespace poly::math
