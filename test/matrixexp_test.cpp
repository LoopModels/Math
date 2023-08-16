#include <Containers/TinyVector.hpp>
#include <Math/Array.hpp>
#include <Math/LinearAlgebra.hpp>
#include <Math/Matrix.hpp>
#include <Math/StaticArrays.hpp>
#include <Utilities/Invariant.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <utility>

namespace poly::math {

template <class T, ptrdiff_t N, ptrdiff_t P = N> class Dual {
  T val{};
  SVector<T, N, P> partials{T{}};

  using PaddedPartials = decltype(SVector<T, N, P>::decompress(nullptr));
  static constexpr ptrdiff_t L =
    std::remove_cvref_t<PaddedPartials>::paddedlength();

public:
  static constexpr bool is_scalar = true;
  using val_type = T;
  static constexpr ptrdiff_t num_partials = N;
  // constexpr Dual() = default;
  constexpr Dual() = default;
  constexpr Dual(T v) : val(v) {}
  constexpr Dual(T v, ptrdiff_t n) : val(v) { partials[n] = T{1}; }
  constexpr Dual(T v, SVector<T, N, P> g) : val(v) { partials << g; }
  constexpr Dual(std::integral auto v) : val(v) {}
  constexpr Dual(std::floating_point auto v) : val(v) {}
  constexpr auto value() -> T & { return val; }
  constexpr auto gradient() -> SVector<T, N, P> & { return partials; }
  [[nodiscard]] constexpr auto value() const -> const T & { return val; }
  [[nodiscard]] constexpr auto gradient() const -> const SVector<T, N, P> & {
    return partials;
  }

  static constexpr auto decompress(const Dual *x) -> Dual<T, N, L> {
    const T *v = &(x->val);
    const SVector<T, N, P> *p = &(x->partials);
    return {utils::decompress(v), SVector<T, N, P>::decompress(p)};
  }
  static constexpr void compress(Dual *x, const Dual<T, N, L> &rhs) {
    T *v = &(x->val);
    SVector<T, N, P> *p = &(x->partials);
    utils::compress(v, rhs.value());
    SVector<T, N, P>::compress(p, rhs.gradient());
  }

  constexpr auto operator-() const & -> Dual { return {-val, -partials}; }
  constexpr auto operator+(const Dual &other) const & -> Dual {
    static_assert(std::popcount(size_t(P)) == 1);
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
template <class T, ptrdiff_t N, ptrdiff_t P>
Dual(T, SVector<T, N, P>) -> Dual<T, N, P>;

template <class T, ptrdiff_t N, ptrdiff_t P>
constexpr auto operator+(double other, Dual<T, N, P> x) -> Dual<T, N, P> {
  return {x.value() + other, x.gradient()};
}
template <class T, ptrdiff_t N, ptrdiff_t P>
constexpr auto operator-(double other, Dual<T, N, P> x) -> Dual<T, N, P> {
  return {x.value() - other, -x.gradient()};
}
template <class T, ptrdiff_t N, ptrdiff_t P>
constexpr auto operator*(double other, Dual<T, N, P> x) -> Dual<T, N, P> {
  return {x.value() * other, other * x.gradient()};
}
template <class T, ptrdiff_t N, ptrdiff_t P>
constexpr auto operator/(double other, Dual<T, N, P> x) -> Dual<T, N, P> {
  return {other / x.value(), -other * x.gradient() / (x.value() * x.value())};
}
constexpr size_t L = 16;
static_assert(
  utils::ElementOf<double, SquareMatrix<Dual<Dual<double, 4, 4>, 2, 2>, L>>);
// auto x = Dual<Dual<double, 4>, 2>{1.0};
// auto y = x * 3.4;
// static_assert(Scalar<Dual<double, 4>>);
static_assert(std::convertible_to<int, Dual<double, 4, 4>>);
static_assert(std::convertible_to<int, Dual<Dual<double, 4, 4>, 2, 2>>);
static_assert(std::same_as<Dual<double, 1, 1>, Dual<double, 1>>);

template <class D> struct URand {
  using T = typename D::val_type;
  static constexpr ptrdiff_t N = D::num_partials;
  auto operator()(std::mt19937_64 &mt) -> D {
    D x{URand<T>{}(mt)};
    for (ptrdiff_t i = 0; i < N; ++i) x.gradient()[i] = URand<T>{}(mt);
    return x;
  }
};
template <> struct URand<double> {
  auto operator()(std::mt19937_64 &mt) -> double {
    return std::uniform_real_distribution<double>(-2, 2)(mt);
  }
};

constexpr auto extractDualValRecurse(std::floating_point auto x) { return x; }
template <class T, ptrdiff_t N, ptrdiff_t P>
constexpr auto extractDualValRecurse(const Dual<T, N, P> &x) {
  return extractDualValRecurse(x.value());
}
template <typename T>
constexpr void evalpoly(MutSquarePtrMatrix<T> B, SquarePtrMatrix<T> C,
                        const auto &p) {
  using S = SquareMatrix<T, L>;
  ptrdiff_t N = p.size();
  invariant(N > 0);
  invariant(ptrdiff_t(C.numRow()), ptrdiff_t(C.numCol()));
  invariant(ptrdiff_t(B.numRow()), ptrdiff_t(B.numCol()));
  invariant(ptrdiff_t(B.numRow()), ptrdiff_t(C.numRow()));
  S A_{SquareDims{N == 2 ? Row{0} : B.numRow()}};
  MutSquarePtrMatrix<T> A{A_};
  if (N & 1) std::swap(A, B);
  B << p[0] * C + p[1] * I;
  for (ptrdiff_t i = 2; i < N; ++i) {
    std::swap(A, B);
    B << A * C + p[i] * I;
  }
}

template <AbstractMatrix T> constexpr auto opnorm1(const T &A) {
  using S = decltype(extractDualValRecurse(std::declval<utils::eltype_t<T>>()));
  ptrdiff_t n = ptrdiff_t(A.numRow());
  invariant(n > 0);
  Vector<S> v;
  v.resizeForOverwrite(n);
  invariant(A.numRow() > 0);
  for (ptrdiff_t j = 0; j < n; ++j)
    v[j] = std::abs(extractDualValRecurse(A(0, j)));
  for (ptrdiff_t i = 1; i < n; ++i)
    for (ptrdiff_t j = 0; j < n; ++j)
      v[j] += std::abs(extractDualValRecurse(A(i, j)));
  return *std::max_element(v.begin(), v.end());
}

constexpr auto exp2(int64_t x) -> double {
  if (x > 1023) return std::numeric_limits<double>::infinity();
  if (x <= -1023) return std::bit_cast<double>(uint64_t(1) << ((x + 1074)));
  return std::bit_cast<double>((x + 1023) << 52);
}
/// computes ceil(log2(x)) for x >= 1
constexpr auto log2ceil(double x) -> unsigned {
  invariant(x >= 1);
  uint64_t u = std::bit_cast<uint64_t>(x) - 1;
  return (u >> 52) - 1022;
}

template <typename T>
constexpr void expm(MutSquarePtrMatrix<T> V, SquarePtrMatrix<T> A) {
  invariant(ptrdiff_t(V.numRow()), ptrdiff_t(A.numRow()));
  unsigned n = unsigned(A.numRow());
  auto nA = opnorm1(A);
  SquareMatrix<T, L> A2_{A * A}, U_{SquareDims{n}};
  MutSquarePtrMatrix<T> A2{A2_}, U{U_};
  unsigned int s = 0;
  if (nA <= 2.1) {
    containers::TinyVector<double, 5> p0, p1;
    if (nA > 0.95) {
      p0 = {1.0, 3960.0, 2162160.0, 302702400.0, 8821612800.0};
      p1 = {90.0, 110880.0, 3.027024e7, 2.0756736e9, 1.76432256e10};
    } else if (nA > 0.25) {
      p0 = {1.0, 1512.0, 277200.0, 8.64864e6};
      p1 = {56.0, 25200.0, 1.99584e6, 1.729728e7};
    } else if (nA > 0.015) {
      p0 = {1.0, 420.0, 15120.0};
      p1 = {30.0, 3360.0, 30240.0};
    } else {
      p0 = {1.0, 60.0};
      p1 = {12.0, 120.0};
    }
    evalpoly(V, A2, p0);
    U << A * V;
    evalpoly(V, A2, p1);
  } else {
    // s = std::max(unsigned(std::ceil(std::log2(nA / 5.4))), unsigned(0));
    s = nA > 5.4 ? log2ceil(nA / 5.4) : unsigned(0);
    double t = 1.0;
    if (s > 0) {
      // t = 1.0 / std::exp2(s);
      t = 1.0 / exp2(s);
      A2 *= (t * t);
      if (s & 1) std::swap(U, V);
    }
    SquareMatrix<T, L> A4{A2 * A2}, A6{A2 * A4};

    V << A6 * (A6 + 16380 * A4 + 40840800 * A2) +
           (33522128640 * A6 + 10559470521600 * A4 + 1187353796428800 * A2) +
           32382376266240000 * I;
    U << A * V;
    if (s > 0) U *= t;
    V << A6 * (182 * A6 + 960960 * A4 + 1323241920 * A2) +
           (670442572800 * A6 + 129060195264000 * A4 + 7771770303897600 * A2) +
           64764752532480000 * I;
  }
  for (auto v = V.begin(), u = U.begin(), e = V.end(); v != e; ++v, ++u) {
    auto &&d = *v - *u;
    *v += *u;
    *u = d;
  }
  LU::ldiv(U, MutPtrMatrix<T>(V));
  for (; s--;) {
    U << V * V;
    std::swap(U, V);
  }
}
template <typename T> constexpr auto expm(SquarePtrMatrix<T> A) {
  SquareMatrix<T, L> V{SquareDims{A.numRow()}};
  expm(V, A);
  return V;
}
template <typename T> void expm(T *A, T *B, ptrdiff_t N) {
  expm(MutSquarePtrMatrix<T>(A, N), SquarePtrMatrix<T>(B, N));
}
template <typename T>
constexpr auto colwise(DensePtrMatrix<T> X, const auto &f, const auto &g,
                       const auto &h) -> DenseMatrix<T> {
  invariant(X.numRow() == 3);
  DenseMatrix<T> Y{math::DenseDims{X.numRow(), X.numCol()}};
  for (ptrdiff_t j = 0; j < X.numCol(); ++j) {
    Y(0, j) = f(X(0, j));
    Y(1, j) = g(X(1, j));
    Y(2, j) = h(X(2, j));
  }
  return Y;
}

template <typename T>
constexpr auto foo(DensePtrMatrix<T> X, ptrdiff_t n) -> T {
  auto f = [](auto x) { return 2 * x; };
  auto g = [](auto x) { return 3 * x; };
  auto h = [](auto x) { return 4 * x; };
  T y = 0;
  for (ptrdiff_t i = 0; i < n; ++i) y += colwise(X, f, g, h).sum();
  return y;
}

} // namespace poly::math

using poly::math::expm, poly::math::Dual;
template <ptrdiff_t N, ptrdiff_t M>
using DDual = Dual<Dual<double, N, N>, M, M>;

static_assert(
  std::same_as<
    double, decltype(poly::math::opnorm1(
              std::declval<poly::math::MutSquarePtrMatrix<DDual<7, 2>>>()))>);
// void expmf64d7d2(DDual<7, 2> *A, DDual<7, 2> *B, ptrdiff_t N) { expm(A, B,
// N); }

auto expwork(const auto &A) {
  auto B = expm(A);
  decltype(B) C{A.dim()};
  for (size_t i = 0; i < 8; ++i) {
    C << A * exp2(-double(i));
    B += expm(C);
  }
  return B;
}

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(MatrixExpTest, BasicAssertions) {
  constexpr int dim = 3;

  std::mt19937_64 mt(0);
  // using D = Dual<Dual<double, 4>, 2>;
  using D = Dual<double, 3>;
  poly::math::SquareMatrix<D> A{poly::math::SquareDims{dim}};
  for (auto &a : A) a = poly::math::URand<D>{}(mt);

  poly::math::SquareMatrix<D> B{poly::math::SquareDims{dim}};
  B.fill(D{0});
  B += expwork(A);
  EXPECT_TRUE(B(0, 0) != 0);
}
