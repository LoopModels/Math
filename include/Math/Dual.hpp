#pragma once
#include "Math/Exp.hpp"
#include "Math/MatrixDimensions.hpp"
#include <Math/Array.hpp>
#include <Math/Constructors.hpp>
#include <Math/Math.hpp>
#include <Math/Matrix.hpp>
#include <Math/StaticArrays.hpp>
#include <Utilities/Invariant.hpp>
#include <cstddef>
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
  constexpr auto gradient(ptrdiff_t i) -> T & { return partials[i]; }
  [[nodiscard]] constexpr auto value() const -> const T & { return val; }
  [[nodiscard]] constexpr auto gradient() const -> const SVector<T, N> & {
    return partials;
  }
  [[nodiscard]] constexpr auto gradient(ptrdiff_t i) const -> const T & {
    return partials[i];
  }
  [[gnu::always_inline]] constexpr auto operator-() const & -> Dual {
    return {-val, -partials};
  }
  [[gnu::always_inline]] constexpr auto
  operator+(const Dual &other) const & -> Dual {
    return {val + other.val, partials + other.partials};
  }
  [[gnu::always_inline]] constexpr auto operator-(const Dual &other) const
    -> Dual {
    return {val - other.val, partials - other.partials};
  }
  [[gnu::always_inline]] constexpr auto operator*(const Dual &other) const
    -> Dual {
    return {val * other.val, val * other.partials + other.val * partials};
  }
  [[gnu::always_inline]] constexpr auto operator/(const Dual &other) const
    -> Dual {
    return {val / other.val, (other.val * partials - val * other.partials) /
                               (other.val * other.val)};
  }
  [[gnu::always_inline]] constexpr auto operator+=(const Dual &other)
    -> Dual & {
    val += other.val;
    partials += other.partials;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator-=(const Dual &other)
    -> Dual & {
    val -= other.val;
    partials -= other.partials;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator*=(const Dual &other)
    -> Dual & {
    val *= other.val;
    partials << val * other.partials + other.val * partials;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator/=(const Dual &other)
    -> Dual & {
    val /= other.val;
    partials << (other.val * partials - val * other.partials) /
                  (other.val * other.val);
    return *this;
  }
  [[gnu::always_inline]] constexpr auto
  operator+(double other) const & -> Dual {
    return {val + other, partials};
  }
  [[gnu::always_inline]] constexpr auto operator-(double other) const -> Dual {
    return {val - other, partials};
  }
  [[gnu::always_inline]] constexpr auto operator*(double other) const -> Dual {
    return {val * other, other * partials};
  }
  [[gnu::always_inline]] constexpr auto operator/(double other) const -> Dual {
    return {val / other, partials / other};
  }
  [[gnu::always_inline]] constexpr auto operator+=(double other) -> Dual & {
    val += other;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator-=(double other) -> Dual & {
    val -= other;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator*=(double other) -> Dual & {
    val *= other;
    partials *= other;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator/=(double other) -> Dual & {
    val /= other;
    partials /= other;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator==(const Dual &other) const
    -> bool {
    return val == other.val; // && grad == other.grad;
  }
  [[gnu::always_inline]] constexpr auto operator!=(const Dual &other) const
    -> bool {
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
[[gnu::always_inline]] constexpr auto operator+(double other, Dual<T, N> x)
  -> Dual<T, N> {
  return {x.value() + other, x.gradient()};
}
template <class T, ptrdiff_t N>
[[gnu::always_inline]] constexpr auto operator-(double other, Dual<T, N> x)
  -> Dual<T, N> {
  return {x.value() - other, -x.gradient()};
}
template <class T, ptrdiff_t N>
[[gnu::always_inline]] constexpr auto operator*(double other, Dual<T, N> x)
  -> Dual<T, N> {
  return {x.value() * other, other * x.gradient()};
}
template <class T, ptrdiff_t N>
[[gnu::always_inline]] constexpr auto operator/(double other, Dual<T, N> x)
  -> Dual<T, N> {
  return {other / x.value(), -other * x.gradient() / (x.value() * x.value())};
}
template <class T, ptrdiff_t N> constexpr auto exp(Dual<T, N> x) -> Dual<T, N> {
  T expx = exp(x.value());
  return {expx, expx * x.gradient()};
}
template <class T, ptrdiff_t N>
constexpr auto sigmoid(Dual<T, N> x) -> Dual<T, N> {
  T s = sigmoid(x.value());
  return {s, (s - s * s) * x.gradient()};
}
template <class T, ptrdiff_t N>
constexpr auto softplus(Dual<T, N> x) -> Dual<T, N> {
  return {softplus(x.value()), sigmoid(x.value()) * x.gradient()};
}
template <class T, ptrdiff_t N> constexpr auto log(Dual<T, N> x) -> Dual<T, N> {
  return {log2(x.value()), x.gradient() / x.value()};
}
template <class T, ptrdiff_t N>
constexpr auto log2(Dual<T, N> x) -> Dual<T, N> {
  constexpr double log2 = 0.6931471805599453; // log(2);
  return {log2(x.value()), x.gradient() / (log2 * x.value())};
}

constexpr auto dval(double &x) -> double & { return x; }
template <typename T, ptrdiff_t N>
constexpr auto dval(Dual<T, N> &x) -> double & {
  return dval(x.value());
}

class GradientResult {
  double x;
  MutPtrVector<double> grad;

public:
  [[nodiscard]] constexpr auto value() const -> double { return x; }
  [[nodiscard]] constexpr auto gradient() const -> MutPtrVector<double> {
    return grad;
  }
};
class HessianResultCore {
  double *ptr;
  ptrdiff_t dim;

public:
  [[nodiscard]] constexpr auto gradient() const -> MutPtrVector<double> {
    return {ptr, dim};
  }
  [[nodiscard]] constexpr auto hessian() const -> MutSquarePtrMatrix<double> {
    return {ptr + dim, SquareDims<>{{dim}}};
  }
  constexpr HessianResultCore(alloc::Arena<> *alloc, ptrdiff_t d)
    : ptr{alloc->allocate<double>(size_t(d) * (d + 1))}, dim{d} {}
};
class HessianResult : public HessianResultCore {
  double x{};

public:
  [[nodiscard]] constexpr auto value() -> double & { return x; }
  [[nodiscard]] constexpr auto value() const -> double { return x; }

  constexpr HessianResult(alloc::Arena<> *alloc, unsigned d)
    : HessianResultCore{alloc, d} {}

  template <size_t I> constexpr auto get() const {
    if constexpr (I == 0) return x;
    else if constexpr (I == 1) return gradient();
    else return hessian();
  }
};

template <ptrdiff_t N, AbstractVector T> struct DualVector {
  using value_type = Dual<utils::eltype_t<T>, N>;
  static_assert(Trivial<T>);
  T x;
  ptrdiff_t offset;
  [[nodiscard]] constexpr auto operator[](ptrdiff_t i) const -> value_type {
    value_type v{x[i]};
    if ((i >= offset) && (i < offset + N)) dval(v.gradient()[i - offset]) = 1.0;
    return v;
  }
  [[nodiscard]] constexpr auto size() const -> ptrdiff_t { return x.size(); }
  [[nodiscard]] constexpr auto numRow() const -> Row<1> { return {}; }
  [[nodiscard]] constexpr auto numCol() const -> Col<> { return {x.size()}; }
  [[nodiscard]] constexpr auto view() const -> DualVector { return *this; }
};
static_assert(AbstractVector<DualVector<8, PtrVector<double>>>);
static_assert(AbstractVector<DualVector<2, DualVector<8, PtrVector<double>>>>);

template <ptrdiff_t N>
constexpr auto dual(const AbstractVector auto &x, ptrdiff_t offset) {
  return DualVector<N, decltype(x.view())>{x.view(), offset};
}

struct Assign {
  constexpr void operator()(double &x, double y) const { x = y; }
};
struct Increment {
  constexpr void operator()(double &x, double y) const { x += y; }
};
struct ScaledIncrement {
  double scale;
  constexpr void operator()(double &x, double y) const { x += scale * y; }
};

constexpr auto gradient(alloc::Arena<> *arena, PtrVector<double> x,
                        const auto &f) {
  constexpr ptrdiff_t U = 8;
  using D = Dual<double, U>;
  ptrdiff_t N = x.size();
  MutPtrVector<double> grad = vector<double>(arena, N);
  auto p = arena->scope();
  for (ptrdiff_t i = 0;; i += U) {
    D fx = alloc::call(*arena, f, dual<U>(x, i));
    for (ptrdiff_t j = 0; ((j < U) && (i + j < N)); ++j)
      grad[i + j] = fx.gradient()[j];
    if (i + U >= N) return std::make_pair(fx.value(), grad);
  }
}
// only computes the upper triangle blocks
template <class T, ptrdiff_t N> constexpr auto value(const Dual<T, N> &x) {
  return value(x.value());
}

/// fills the lower triangle of the hessian
constexpr auto hessian(HessianResultCore hr, PtrVector<double> x, const auto &f,
                       auto update) -> double {
  constexpr ptrdiff_t Ui = 8;
  constexpr ptrdiff_t Uj = 2;
  using D = Dual<double, Ui>;
  using DD = Dual<D, Uj>;
  ptrdiff_t N = x.size();
  MutPtrVector<double> grad = hr.gradient();
  MutSquarePtrMatrix<double> hess = hr.hessian();
  invariant(N == grad.size());
  invariant(N == hess.numCol());
  for (ptrdiff_t j = 0;; j += Uj) {
    bool jbr = j + Uj >= N;
    for (ptrdiff_t i = 0;; i += Ui) {
      // df^2/dx_i dx_j
      bool ibr = i + Ui - Uj >= j;
      // we want to copy into both regions _(j, j+Uj) and _(i, i+Ui)
      // these regions overlap for the last `i` iteration only
      DD fx = f(dual<Uj>(dual<Ui>(x, i), j));
      // DD fx = alloc::call(arena, f, x);
      for (ptrdiff_t k = 0; ((k < Uj) && (j + k < N)); ++k)
        for (ptrdiff_t l = 0; ((l < Ui) && (i + l < N)); ++l)
          update(hess[j + k, i + l], fx.gradient()[k].gradient()[l]);
      if (jbr)
        for (ptrdiff_t k = 0; ((k < Ui) && (i + k < N)); ++k)
          grad[i + k] = fx.value().gradient()[k];
      if (!ibr) continue;
      if (jbr) return fx.value().value();
      break;
    }
  }
}
constexpr auto hessian(HessianResultCore hr, PtrVector<double> x, const auto &f)
  -> double {
  Assign assign{};
  return hessian(hr, x, f, assign);
}

constexpr auto hessian(alloc::Arena<> *arena, PtrVector<double> x,
                       const auto &f) -> HessianResult {
  unsigned N = x.size();
  HessianResult hr{arena, N};
  hr.value() = hessian(hr, x, f);
  return hr;
}
static_assert(MatrixDimension<SquareDims<>>);

} // namespace poly::math
namespace std {
template <> struct tuple_size<poly::math::HessianResult> {
  static constexpr size_t value = 3;
};
template <> struct tuple_element<size_t(0), poly::math::HessianResult> {
  using type = double;
};
template <> struct tuple_element<size_t(1), poly::math::HessianResult> {
  using type = poly::math::MutPtrVector<double>;
};
template <> struct tuple_element<size_t(2), poly::math::HessianResult> {
  using type = poly::math::MutSquarePtrMatrix<double>;
};

} // namespace std
