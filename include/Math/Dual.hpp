#pragma once
#include "Math/Vector.hpp"
#include <Math/Array.hpp>
#include <Math/Constructors.hpp>
#include <Math/Math.hpp>
#include <Math/Matrix.hpp>
#include <Math/StaticArrays.hpp>
#include <Utilities/Invariant.hpp>
#include <cstddef>
#include <functional>
#include <utility>

namespace poly::math {

template <class T, ptrdiff_t N> class Dual {
  T val{};
  SVector<T, N> partials{T{}};

public:
  constexpr Dual() = default;
  constexpr Dual(T v) : val(v) {}
  constexpr Dual(T v, ptrdiff_t n) : val(v) { partials[n] = T{1}; }
  constexpr Dual(T v, SVector<T, N> g) : val(v) { partials << g; }
  constexpr Dual(std::integral auto v) : val(v) {}
  constexpr Dual(std::floating_point auto v) : val(v) {}
  constexpr auto value() -> T & { return val; }
  constexpr auto gradient() -> SVector<T, N> & { return partials; }
  [[nodiscard]] constexpr auto value() const -> const T & { return val; }
  [[nodiscard]] constexpr auto gradient() const -> const SVector<T, N> & {
    return partials;
  }
  constexpr auto operator-() const & -> Dual { return {-val, -partials}; }
  constexpr auto operator+(const Dual &other) const & -> Dual {
    return {val + other.val, partials + other.partials};
  }
  constexpr auto operator-(const Dual &other) const -> Dual {
    return {val - other.val, partials - other.partials};
  }
  constexpr auto operator*(const Dual &other) const -> Dual {
    return {val * other.val, val * other.partials + other.val * partials};
  }
  constexpr auto operator/(const Dual &other) const -> Dual {
    return {val / other.val, (other.val * partials - val * other.partials) /
                               (other.val * other.val)};
  }
  constexpr auto operator+=(const Dual &other) -> Dual & {
    val += other.val;
    partials += other.partials;
    return *this;
  }
  constexpr auto operator-=(const Dual &other) -> Dual & {
    val -= other.val;
    partials -= other.partials;
    return *this;
  }
  constexpr auto operator*=(const Dual &other) -> Dual & {
    val *= other.val;
    partials << val * other.partials + other.val * partials;
    return *this;
  }
  constexpr auto operator/=(const Dual &other) -> Dual & {
    val /= other.val;
    partials << (other.val * partials - val * other.partials) /
                  (other.val * other.val);
    return *this;
  }
  constexpr auto operator+(double other) const & -> Dual {
    return {val + other, partials};
  }
  constexpr auto operator-(double other) const -> Dual {
    return {val - other, partials};
  }
  constexpr auto operator*(double other) const -> Dual {
    return {val * other, other * partials};
  }
  constexpr auto operator/(double other) const -> Dual {
    return {val / other, partials / other};
  }
  constexpr auto operator+=(double other) -> Dual & {
    val += other;
    return *this;
  }
  constexpr auto operator-=(double other) -> Dual & {
    val -= other;
    return *this;
  }
  constexpr auto operator*=(double other) -> Dual & {
    val *= other;
    partials *= other;
    return *this;
  }
  constexpr auto operator/=(double other) -> Dual & {
    val /= other;
    partials /= other;
    return *this;
  }
  constexpr auto operator==(const Dual &other) const -> bool {
    return val == other.val; // && grad == other.grad;
  }
  constexpr auto operator!=(const Dual &other) const -> bool {
    return val != other.val; // || grad != other.grad;
  }
  constexpr auto operator==(double other) const -> bool { return val == other; }
  constexpr auto operator!=(double other) const -> bool { return val != other; }
  constexpr auto operator<(double other) const -> bool { return val < other; }
  constexpr auto operator>(double other) const -> bool { return val > other; }
  constexpr auto operator<=(double other) const -> bool { return val <= other; }
  constexpr auto operator>=(double other) const -> bool { return val >= other; }
  constexpr auto operator<(const Dual &other) const -> bool {
    return val < other.val;
  }
  constexpr auto operator>(const Dual &other) const -> bool {
    return val > other.val;
  }
  constexpr auto operator<=(const Dual &other) const -> bool {
    return val <= other.val;
  }
  constexpr auto operator>=(const Dual &other) const -> bool {
    return val >= other.val;
  }
};
template <class T, ptrdiff_t N> Dual(T, SVector<T, N>) -> Dual<T, N>;

template <class T, ptrdiff_t N>
constexpr auto operator+(double other, Dual<T, N> x) -> Dual<T, N> {
  return {x.value() + other, x.gradient()};
}
template <class T, ptrdiff_t N>
constexpr auto operator-(double other, Dual<T, N> x) -> Dual<T, N> {
  return {x.value() - other, -x.gradient()};
}
template <class T, ptrdiff_t N>
constexpr auto operator*(double other, Dual<T, N> x) -> Dual<T, N> {
  return {x.value() * other, other * x.gradient()};
}
template <class T, ptrdiff_t N>
constexpr auto operator/(double other, Dual<T, N> x) -> Dual<T, N> {
  return {other / x.value(), -other * x.gradient() / (x.value() * x.value())};
}
template <class T, ptrdiff_t N> constexpr auto exp(Dual<T, N> x) -> Dual<T, N> {
  T expx = exp(x.value());
  return {expx, expx * x.gradient()};
}

template <typename T>
constexpr auto gradient(utils::Arena<> *arena, PtrVector<T> x, const auto &f) {
  constexpr ptrdiff_t U = 7;
  using D = Dual<T, U>;
  ptrdiff_t N = x.size();
  MutPtrVector<T> grad = vector<T>(arena, N);
  auto p = arena->scope();
  MutPtrVector<D> dx = vector<D>(arena, N);
  for (ptrdiff_t i = 0; i < N; ++i) dx[i] = x[i];
  for (ptrdiff_t i = 0;; i += U) {
    for (ptrdiff_t j = 0; ((j < U) && (i + j < N)); ++j)
      dx[i + j] = D(x[i + j], j);
    D fx = utils::call(*arena, f, dx);
    for (ptrdiff_t j = 0; ((j < U) && (i + j < N)); ++j)
      grad[i + j] = fx.gradient()[j];
    if (i + U >= N) return std::make_pair(fx.value(), grad);
    for (ptrdiff_t j = 0; ((j < U) && (i + j < N)); ++j)
      dx[i + j] = x[i + j]; // reset
  }
}
// only computes the upper triangle blocks
template <typename T>
constexpr auto hessian(utils::Arena<> *arena, PtrVector<T> x, const auto &f) {
  constexpr ptrdiff_t Ui = 7;
  constexpr ptrdiff_t Uj = 2;
  using D = Dual<T, Ui>;
  using DD = Dual<D, Uj>;
  ptrdiff_t N = x.size();
  MutPtrVector<T> grad = vector<T>(arena, N);
  MutSquarePtrMatrix<T> hess = matrix<T>(arena, N);
  auto p = arena->scope();
  MutPtrVector<DD> dx = vector<DD>(arena, N);
  for (ptrdiff_t i = 0; i < N; ++i) dx[i] = x[i];
  for (ptrdiff_t j = 0;; j += Uj) {
    for (ptrdiff_t i = j;; i += Ui) {
      // df^2/dx_i dx_j
      // we want to copy into both regions _(j, j+Uj) and _(i, i+Ui)
      // these regions overlap for the first `i` iteration only
      if (i == j)
        for (ptrdiff_t k = 0; ((k < Uj) && (j + k < N)); ++k)
          dx[j + k] = DD(D(x[j + k], k), k);
      for (ptrdiff_t k = (i == j) ? Uj : 0; ((k < Ui) && (i + k < N)); ++k)
        dx[i + k] = D(x[i + k], k);

      DD fx = utils::call(*arena, f, dx);
      for (ptrdiff_t k = 0; ((k < Uj) && (j + k < N)); ++k)
        for (ptrdiff_t l = 0; ((l < Ui) && (i + l < N)); ++l)
          hess(j + k, i + l) = fx.gradient()[k].gradient()[l];
      if (i == j)
        for (ptrdiff_t k = 0; ((k < Ui) && (i + k < N)); ++k)
          grad[i + k] = fx.value().gradient()[k];

      if (i + Ui >= N) {
        if (j + Uj >= N) return std::make_tuple(fx.value().value(), grad, hess);
        // we have another `j` iteration, so we reset
        // if `i != j`, the `i` and `j` blocks aren't contiguous, we reset both
        // if `i == j`, we have one block; we only bother resetting
        // the lower `j` subsection, because we're about to overwrite
        // the upper `i` subsection anyway
        for (ptrdiff_t k = 0; ((k < Uj) && (j + k < N)); ++k)
          dx[j + k] = x[j + k];
        if (i != j)
          for (ptrdiff_t k = 0; ((k < Ui) && (i + k < N)); ++k)
            dx[i + k] = x[i + k];
        break;
      }
      // if we're here, we have another `i` iteration
      // if we're in the first `i` iteration, we set the first Uj iter
      if (i == j)
        for (ptrdiff_t k = 0; ((k < Uj) && (j + k < N)); ++k)
          dx[j + k] = DD(x[j + k], k);
      for (ptrdiff_t k = (i == j ? Uj : 0); ((k < Ui) && (i + k < N)); ++k)
        dx[i + k] = x[i + k]; // reset `i` block
    }
  }
}

constexpr auto extractDualValRecurse(std::floating_point auto x) { return x; }
template <class T, ptrdiff_t N>
constexpr auto extractDualValRecurse(const Dual<T, N> &x) {
  return extractDualValRecurse(x.value());
}

} // namespace poly::math
