#ifdef USE_MODULE
module;
#else
#pragma once
#endif
#include "LoopMacros.hxx"
#ifndef USE_MODULE
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

#include "Alloc/Arena.cxx"
#include "Alloc/Mallocator.cxx"
#include "Containers/Pair.cxx"
#include "Containers/Tuple.cxx"
#include "Math/Array.cxx"
#include "Math/ArrayConcepts.cxx"
#include "Math/AxisTypes.cxx"
#include "Math/Dual.cxx"
#include "Math/ElementarySIMD.cxx"
#include "Math/LinearAlgebra.cxx"
#include "Math/MatrixDimensions.cxx"
#include "Math/Reductions.cxx"
#include "SIMD/SIMD.cxx"
#include "Utilities/Parameters.cxx"
#else
export module BoxOpt;

import Allocator;
import Arena;
import Array;
import ArrayConcepts;
import AxisTypes;
import Dual;
import Elementary;
import LinearAlgebra;
import MatDim;
import Pair;
import Param;
import Reductions;
import SIMD;
import STL;
import Tuple;
#endif

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif

constexpr double EXTREME = 8.0;

class BoxTransformView {
public:
  constexpr BoxTransformView(char *d, unsigned ntotal)
    : data{d}, Ntotal{ntotal}, Nraw{ntotal} {}
  constexpr BoxTransformView(char *d, unsigned ntotal, unsigned nraw)
    : data{d}, Ntotal{ntotal}, Nraw{nraw} {}
  [[nodiscard]] constexpr auto size() const -> unsigned { return Ntotal; }
  constexpr auto operator()(const AbstractVector auto &x,
                            ptrdiff_t i) const -> utils::eltype_t<decltype(x)> {
    invariant(i < Ntotal);
    int j = getInds()[i];
    double off = offs()[i];
    if (j < 0) return off;
    return scales()[i] * sigmoid(x[j]) + off;
  }
  template <ptrdiff_t U, ptrdiff_t W, typename M>
  constexpr auto operator()(const AbstractVector auto &x,
                            simd::index::Unroll<U, W, M> i) const
    -> simd::Unroll<1, U, W, utils::eltype_t<decltype(x)>> {
    using T = utils::eltype_t<decltype(x)>;
    using V = simd::Vec<W, T>;
    simd::Unroll<1, U, W, int> j = getInds()[i];
    simd::Unroll<1, U, W, double> off = offs()[i], scale = scales()[i];
    simd::Unroll<1, U, W, T> y;
    if constexpr (U == 1) {
#ifdef __AVX512F__
      auto m = simd::cmp::ge<W, int>(j.vec_, simd::Vec<W, int>{});
#else
      auto m = simd::cmp::ge<W, int64_t>(simd::zextelts<W>(j.vec_),
                                         simd::Vec<W, int64_t>{});
#endif
      V xload = simd::gather(x.data(), i.mask_ & m, j.vec_);
      y.vec_ = simd::select<double>(
        m, scale.vec_ * sigmoid<W>(xload) + off.vec_, off.vec_);
    } else {
      POLYMATHFULLUNROLL
      for (ptrdiff_t u = 0; u < U; ++u) {
        auto m = simd::cmp::ge<W, int>(j.data_[u], simd::Vec<W, int>{});
        V xload = simd::gather(x.data(), i.mask_ & m, j.data_[u]);
        y.data_[u] = simd::select<double>(
          m, scale.data_[u] * sigmoid<W>(xload) + off.data_[u], off.data_[u]);
      }
    }
    return y;
  }
  constexpr void set(AbstractVector auto &x, auto y, ptrdiff_t i) {
    invariant(i < Ntotal);
    int j = getInds()[i];
    if (j < 0) return;
    double off = offs()[i];
    x[j] = logit((y - off) / scales()[i]);
  }
  [[nodiscard]] constexpr auto view() const -> BoxTransformView {
    return *this;
  }
  [[nodiscard]] constexpr auto getLowerBounds() -> MutPtrVector<int32_t> {
    return {i32() + Ntotal, length(Ntotal)};
  }
  [[nodiscard]] constexpr auto getUpperBounds() -> MutPtrVector<int32_t> {
    return {i32() + ptrdiff_t(2) * Ntotal, length(Ntotal)};
  }
  // gives max fractional ind on the transformed scale
  template <bool Relative = true>
  constexpr auto
  maxFractionalComponent() -> containers::Pair<ptrdiff_t, int32_t> {
    double max = 0.0, lb = 0.0;
    ptrdiff_t k = -1;
    MutPtrVector<double> x{getRaw()};
    for (ptrdiff_t i = 0, j = 0; i < Ntotal; ++i) {
      double s = scales()[i];
      if (s == 0.0) continue;
      double a = s * sigmoid(x[j++]) + offs()[i], fa = std::floor(a),
             d = a - fa;
      d = (d > 0.5) ? 1.0 - d : d;
      if constexpr (Relative) d /= a;
      if (d <= max) continue;
      max = d;
      k = i;
      lb = fa;
    }
    return {k, static_cast<int32_t>(lb)};
  }
  [[nodiscard]] constexpr auto getRaw() -> MutPtrVector<double> {
    return {f64() + ptrdiff_t(2) * Ntotal, length(Nraw)};
  }
  [[nodiscard]] constexpr auto getRaw() const -> PtrVector<double> {
    return {f64() + ptrdiff_t(2) * Ntotal, length(Nraw)};
  }

protected:
  static constexpr ptrdiff_t NDV = 3;
  static constexpr ptrdiff_t NIV = 3;

  char *data;
  /// f64 data:
  /// Ntotal offsets
  /// Ntotal scales
  /// i32 data:
  /// Ntotal indices
  /// Ntotal lower bounds
  /// Ntotal upper bounds
  unsigned Ntotal;
  unsigned Nraw;

  [[nodiscard]] auto f64() -> double * {
    return reinterpret_cast<double *>(data);
  }
  [[nodiscard]] auto f64() const -> const double * {
    return reinterpret_cast<const double *>(data);
  }
  [[nodiscard]] static constexpr auto f64Bytes(size_t Ntotal) -> size_t {
    return (NDV * sizeof(double)) * Ntotal;
  }
  [[nodiscard]] static constexpr auto i32Bytes(size_t Ntotal) -> size_t {
    return (NIV * sizeof(int32_t)) * Ntotal;
  }
  [[nodiscard]] static constexpr auto dataBytes(size_t Ntotal) -> size_t {
    // dataBytes must be a multiple of the alignment to make aligned alloc happy
    // so, if Ntotal is odd, we padd with an extra 4 bytes
    return f64Bytes(Ntotal) + i32Bytes(Ntotal) + (Ntotal & 1) * sizeof(int32_t);
  }
  [[nodiscard]] constexpr auto f64Bytes() const -> size_t {
    return f64Bytes(Ntotal);
  }
  [[nodiscard]] constexpr auto i32Bytes() const -> size_t {
    return i32Bytes(Ntotal);
  }
  [[nodiscard]] constexpr auto dataBytes() const -> size_t {
    return dataBytes(Ntotal);
  }
  [[nodiscard]] auto i32() -> int32_t * {
    return reinterpret_cast<int32_t *>(data + f64Bytes());
  }
  [[nodiscard]] auto i32() const -> const int32_t * {
    return reinterpret_cast<const int32_t *>(data + f64Bytes());
  }
  [[nodiscard]] constexpr auto getInds() const -> PtrVector<int32_t> {
    return {i32(), length(Ntotal)};
  }
  [[nodiscard]] constexpr auto getLowerBounds() const -> PtrVector<int32_t> {
    return {i32() + Ntotal, length(Ntotal)};
  }
  [[nodiscard]] constexpr auto getUpperBounds() const -> PtrVector<int32_t> {
    return {i32() + ptrdiff_t(2) * Ntotal, length(Ntotal)};
  }
  [[nodiscard]] constexpr auto offs() const -> PtrVector<double> {
    return {f64(), length(Ntotal)};
  }
  [[nodiscard]] constexpr auto scales() const -> PtrVector<double> {
    return {f64() + Ntotal, length(Ntotal)};
  }
  [[nodiscard]] constexpr auto getInds() -> MutPtrVector<int32_t> {
    return {i32(), length(Ntotal)};
  }
  [[nodiscard]] constexpr auto offs() -> MutPtrVector<double> {
    return {f64(), length(Ntotal)};
  }
  [[nodiscard]] constexpr auto scales() -> MutPtrVector<double> {
    return {f64() + Ntotal, length(Ntotal)};
  }
  static constexpr auto
  scaleOff(int32_t lb, int32_t ub) -> containers::Pair<double, double> {
#ifdef __cpp_if_consteval
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
template <typename V>
concept IsMutable = requires(V v, utils::eltype_t<V> x, ptrdiff_t i) {
  { v.data()[i] = x };
};
static_assert(IsMutable<math::MutArray<double, Length<>>>);
static_assert(!IsMutable<math::Array<double, Length<>>>);
static_assert(!IsMutable<DualVector<2, math::Array<double, Length<>>>>);
template <AbstractVector V> struct BoxTransformVector {
  using value_type = utils::eltype_t<V>;
  static_assert(utils::TriviallyCopyable<V>);
  V v;
  BoxTransformView btv;

  [[nodiscard]] constexpr auto size() const -> ptrdiff_t { return btv.size(); }
  constexpr auto operator[](ptrdiff_t i) const -> value_type {
    return btv(v, i);
  }
  template <ptrdiff_t U, ptrdiff_t W, typename M>
  constexpr auto operator[](simd::index::Unroll<U, W, M> i) const
    -> simd::Unroll<1, U, W, double> {
    return btv(v, i);
  }
  struct Reference {
    BoxTransformVector &x;
    ptrdiff_t i;
    constexpr operator value_type() const { return x.btv(x.v, i); }
    constexpr auto operator=(value_type y) -> Reference & {
      x.btv.set(x.v, y, i);
      return *this;
    }
  };
  constexpr auto operator[](ptrdiff_t i) -> Reference
  requires(IsMutable<V>)
  {
    return {*this, i};
  }
  [[nodiscard]] constexpr auto view() const -> BoxTransformVector {
    return *this;
  }
};
template <AbstractVector V>
BoxTransformVector(V, BoxTransformView) -> BoxTransformVector<V>;

class BoxTransform : public BoxTransformView {
  [[nodiscard]] static auto allocate(size_t ntotal) -> char * {
    return alloc::Mallocator<char>::allocate(dataBytes(ntotal),
                                             std::align_val_t{alignof(double)});
  }
  static void deallocate(char *data, size_t ntotal) {
    if (data)
      alloc::Mallocator<char>::deallocate(data, dataBytes(ntotal),
                                          std::align_val_t{alignof(double)});
  }

public:
  template <size_t N>
  constexpr BoxTransform(std::array<int32_t, N> lb, std::array<int32_t, N> ub)
    : BoxTransformView{allocate(N), N} {
    for (int32_t i = 0; i < int32_t(N); ++i) {
      int32_t l = lb[i], u = ub[i];
      invariant(l < u);
      auto [s, o] = scaleOff(l, u);
      getInds()[i] = i;
      getLowerBounds()[i] = l;
      getUpperBounds()[i] = u;
      scales()[i] = s;
      offs()[i] = o;
    }
  }

  constexpr BoxTransform(unsigned ntotal, int32_t lb, int32_t ub)
    : BoxTransformView{allocate(ntotal), ntotal} {
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
  constexpr void increaseLowerBound(ptrdiff_t idx, int32_t lb) {
    invariant(idx < Ntotal);
    invariant(lb > getLowerBounds()[idx]);
    int32_t ub = getUpperBounds()[idx];
    invariant(lb <= ub);
    getLowerBounds()[idx] = lb;
    double newScale, newOff;
    MutPtrVector<double> untrf{getRaw()};
    if (lb == ub) {
      --Nraw;
      untrf.erase(getInds()[idx]);
      getInds()[idx] = -1;
      // we remove a fixed, so we must now decrement all following inds
      for (ptrdiff_t i = idx; ++i < Ntotal;) --getInds()[i];
      newScale = 0.0;
      newOff = lb;
    } else {
      untrf[getInds()[idx]] = -EXTREME;
      containers::tie(newScale, newOff) = scaleOff(lb, ub);
    }
    scales()[idx] = newScale;
    offs()[idx] = newOff;
  }
  constexpr void decreaseUpperBound(ptrdiff_t idx, int32_t ub) {
    invariant(idx < Ntotal);
    invariant(ub < getUpperBounds()[idx]);
    int32_t lb = getLowerBounds()[idx];
    invariant(lb <= ub);
    getUpperBounds()[idx] = ub;
    double newScale, newOff;
    MutPtrVector<double> untrf{getRaw()};
    if (lb == ub) {
      --Nraw;
      untrf.erase(getInds()[idx]);
      getInds()[idx] = -1;
      // we remove a fixed, so we must now decrement all following inds
      for (ptrdiff_t i = idx; ++i < Ntotal;) --getInds()[i];
      newScale = 0.0;
      newOff = lb;
    } else {
      untrf[getInds()[idx]] = EXTREME;
      containers::tie(newScale, newOff) = scaleOff(lb, ub);
    }
    scales()[idx] = newScale;
    offs()[idx] = newOff;
  }
  constexpr ~BoxTransform() { deallocate(data, Ntotal); }
  constexpr BoxTransform(BoxTransform &&other) noexcept
    : BoxTransformView{other.data, other.Ntotal, other.Nraw} {
    other.data = nullptr;
  }
  constexpr BoxTransform(const BoxTransform &other)
    : BoxTransformView{allocate(other.Ntotal), other.Ntotal, other.Nraw} {
    std::memcpy(data, other.data, dataBytes());
  }
  constexpr auto operator=(BoxTransform &&other) noexcept -> BoxTransform & {
    if (this == &other) return *this;
    deallocate(data, Ntotal);
    data = std::exchange(other.data, nullptr);
    Ntotal = other.Ntotal;
    Nraw = other.Nraw;
    return *this;
  }
  // returns a copy of `this` with the `i`th upper bound decreased to `nlb`
  // this's `i`th lower bound gets increased to `nlb+1`
  constexpr auto fork(ptrdiff_t i, int32_t nlb) -> BoxTransform {
    BoxTransform ret{*this};
    ret.getRaw() << getRaw();
    decreaseUpperBound(i, nlb);
    ret.increaseLowerBound(i, ++nlb);
    return ret;
  }

  [[nodiscard]] constexpr auto
  transformed() const -> BoxTransformVector<PtrVector<double>> {
    return {getRaw(), view()};
  }
  [[nodiscard]] constexpr auto
  transformed() -> BoxTransformVector<MutPtrVector<double>> {
    return {getRaw(), view()};
  }
};

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

constexpr auto minimize(alloc::Arena<> *alloc, MutPtrVector<double> x,
                        const auto &f) -> double {
  constexpr bool constrained =
    IsBoxCall<std::remove_cvref_t<decltype(f)>>::value;
  constexpr double tol = 1e-8;
  constexpr double tol2 = tol * tol;
  constexpr double tol4 = tol2 * tol2;
  constexpr double c = 0.5;
  constexpr double tau = 0.5;
  double alpha = 0.25, fx;
  ptrdiff_t L = x.size();
  auto scope = alloc->scope();
  auto *data = alloc->allocate<double>(2 * L);
  MutPtrVector<double> xnew{data, length(L)}, dir{data + L, length(L)}, xcur{x};
  HessianResultCore hr{alloc, L};
  for (ptrdiff_t n = 0; n < 1000; ++n) {
    fx = hessian(hr, xcur, f);
    invariant(fx == fx);
    if (hr.gradient().norm2() < tol2) break;
    LDL::ldiv<true>(hr.hessian(), dir, hr.gradient());
    if constexpr (!constrained)
      if (dir.norm2() < tol4) break;
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
      if (dir.norm2() < tol4) break;
    t *= c;
    double fxnew = f(xnew);
    invariant(fxnew == fxnew);
    bool dobreak = false;
    if ((fx - fxnew) >= alpha * t) {
      // success; we try to grow alpha, if we're not already optimal
      if (((fx - fxnew) <= tol2) || (norm2(xcur - xnew) <= tol4)) {
        std::swap(xcur, xnew);
        break;
      }
      for (;;) {
        // here, we know `xnew` is better than `xcur`
        // so we write into `xcur` to try and find something even better
        double alphanew = alpha * (1 / tau);
        // xnew == xcur - alpha * dir;
        // xcur == xnew + alpha * dir;
        // xtmp == xcur - alphanew * dir;
        // xtmp == xnew + (alpha - alphanew) * dir;
        // thus, we can use `xcur` as an `xtmp` buffer
        double a = alpha - alphanew;
        for (ptrdiff_t i = 0; i < L; ++i) {
          double xi = xnew[i] + a * dir[i];
          if constexpr (constrained) xi = std::clamp(xi, -EXTREME, EXTREME);
          xcur[i] = xi;
        }
        double fxtmp = f(xcur);
        if (fxtmp != fxtmp) break;
        if ((fx - fxtmp) < alphanew * t) break; // xnew is better
        std::swap(xcur, xnew);                  // we keep xcur as new best
        alpha = alphanew;
        fxnew = fxtmp;
        dobreak = (fx - fxtmp) <= tol2;
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
        if (fxnew != fxnew) continue;
        dobreak = norm2(xcur - xnew) <= tol4;
        if (dobreak) break;
        if ((fx - fxnew) < alpha * t) continue;
        dobreak = (fx - fxnew) <= tol2;
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

constexpr auto minimize(alloc::Arena<> *alloc, BoxTransformView trf,
                        const auto &f) -> double {
  return minimize(alloc, trf.getRaw(), BoxCall<decltype(f)>{f, trf});
}

} // namespace math
