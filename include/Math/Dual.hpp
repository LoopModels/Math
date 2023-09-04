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
  static constexpr ptrdiff_t L = N + 1;
  static constexpr ptrdiff_t value_idx = 0; // N;
  static constexpr ptrdiff_t partial_offset = value_idx != N;
  SVector<T, L> data{T{}};

public:
  static constexpr bool is_scalar = true;
  using val_type = T;
  static constexpr size_t num_partials = N;
  // constexpr Dual() = default;
  constexpr Dual() = default;
  constexpr Dual(T v) { data[value_idx] = v; }
  constexpr Dual(T v, ptrdiff_t n) {
    data[value_idx] = v;
    data[partial_offset + n] = T{1};
  }
  constexpr Dual(T v, ptrdiff_t n, T p) {
    data[value_idx] = v;
    data[partial_offset + n] = p;
  }
  constexpr Dual(T v, AbstractVector auto g) {
    value() = v;
    gradient() << g;
  }
  constexpr Dual(SVector<T, L> d) : data{d} {}
  constexpr Dual(std::integral auto v) { value() = v; }
  constexpr Dual(std::floating_point auto v) { value() = v; }
  constexpr auto value() -> T & { return data[value_idx]; }
  constexpr auto gradient() -> MutPtrVector<T> {
    return data[_(0, N) + partial_offset];
  }
  [[nodiscard]] constexpr auto value() const -> const T & {
    return data[value_idx];
  }
  [[nodiscard]] constexpr auto gradient() const -> PtrVector<T> {
    return data[_(0, N) + partial_offset];
  }
  constexpr auto operator-() const & -> Dual { return {-data}; }
  constexpr auto operator+(const Dual &other) const & -> Dual {
    return {data + other.data};
  }
  constexpr auto operator-(const Dual &other) const -> Dual {
    return {data - other.data};
  }
  constexpr auto operator*(const Dual &other) const -> Dual {
    return {conditional(std::plus<>{},
                        elementwise_not_equal(_(0, N + 1), value_idx),
                        value() * other.data, data * other.value())};
  }
  constexpr auto operator/(const Dual &other) const -> Dual {
    // val = value() / other.value()
    // partials = (other.value() * gradient() - value() * other.gradient()) /
    // (other.value() * other.value())
    // partials = (gradient()) / (other.value())
    //  - (value() * other.gradient()) / (other.value() * other.value())
    return {
      conditional(std::minus<>{}, elementwise_not_equal(_(0, N + 1), value_idx),
                  data / other.value(),
                  value() * other.data / (other.value() * other.value()))};
  }
  constexpr auto operator+=(const Dual &other) -> Dual & {
    data += other.data;
    return *this;
  }
  constexpr auto operator-=(const Dual &other) -> Dual & {
    data -= other.data;
    return *this;
  }
  constexpr auto operator*=(const Dual &other) -> Dual & {
    data << conditional(std::plus<>{},
                        elementwise_not_equal(_(0, N + 1), value_idx),
                        value() * other.data, data * other.value());
    return *this;
  }
  constexpr auto operator/=(const Dual &other) -> Dual & {
    data << conditional(std::minus<>{},
                        elementwise_not_equal(_(0, N + 1), value_idx),
                        data / other.value(),
                        value() * other.data / (other.value() * other.value()));
    return *this;
  }
  constexpr auto operator+(double other) const & -> Dual {
    Dual ret = *this;
    ret.value() += other;
    return ret;
  }
  constexpr auto operator-(double other) const -> Dual {
    Dual ret = *this;
    ret.value() -= other;
    return ret;
  }
  constexpr auto operator*(double other) const -> Dual {
    return {data * other};
  }
  constexpr auto operator/(double other) const -> Dual {
    return {data / other};
  }
  constexpr auto operator+=(double other) -> Dual & {
    value() += other;
    return *this;
  }
  constexpr auto operator-=(double other) -> Dual & {
    value() -= other;
    return *this;
  }
  constexpr auto operator*=(double other) -> Dual & {
    data *= other;
    return *this;
  }
  constexpr auto operator/=(double other) -> Dual & {
    data /= other;
    return *this;
  }
  constexpr auto operator==(const Dual &other) const -> bool {
    return value() == other.value(); // && grad == other.grad;
  }
  constexpr auto operator!=(const Dual &other) const -> bool {
    return value() != other.value(); // || grad != other.grad;
  }
  constexpr auto operator==(double other) const -> bool {
    return value() == other;
  }
  constexpr auto operator!=(double other) const -> bool {
    return value() != other;
  }
  constexpr auto operator<(double other) const -> bool {
    return value() < other;
  }
  constexpr auto operator>(double other) const -> bool {
    return value() > other;
  }
  constexpr auto operator<=(double other) const -> bool {
    return value() <= other;
  }
  constexpr auto operator>=(double other) const -> bool {
    return value() >= other;
  }
  constexpr auto operator<(const Dual &other) const -> bool {
    return value() < other.value();
  }
  constexpr auto operator>(const Dual &other) const -> bool {
    return value() > other.value();
  }
  constexpr auto operator<=(const Dual &other) const -> bool {
    return value() <= other.value();
  }
  constexpr auto operator>=(const Dual &other) const -> bool {
    return value() >= other.value();
  }

  friend constexpr auto exp(Dual x) -> Dual {
    return {conditional(std::multiplies<>{},
                        elementwise_not_equal(_(0, N + 1), value_idx),
                        exp(x.value()), x.data)};
  }

  friend constexpr auto operator+(double other, Dual x) -> Dual {
    return {other + x.data};
  }
  friend constexpr auto operator-(double other, Dual x) -> Dual {
    return {other - x.data};
  }
  friend constexpr auto operator*(double other, Dual x) -> Dual {
    return {other * x.data};
  }
  friend constexpr auto operator/(double other, Dual x) -> Dual {
    T v = other / x.value();
    return {conditional(std::multiplies<>{},
                        elementwise_not_equal(_(0, N + 1), value_idx), v,
                        -x.data / x.value())};
    // return {v, -v * x.gradient() / (x.value())};
  }
};
//   template <class T, size_t N>
// friend constexpr auto operator+(double other, Dual<T, N> x) -> Dual<T, N> {
//   return {other + data};
// }
// template <class T, size_t N>
// friend constexpr auto operator-(double other, Dual<T, N> x) -> Dual<T, N> {
//   return {other - data};
// }
// template <class T, size_t N>
// friend constexpr auto operator*(double other, Dual<T, N> x) -> Dual<T, N> {
//   return {other * data};
// }

// template <class T, class... U> Dual(T, U...) -> Dual<T, 0 + sizeof...(U)>;

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
