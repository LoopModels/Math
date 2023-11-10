#include <Containers/TinyVector.hpp>
#include <Math/Array.hpp>
#include <Math/Dual.hpp>
#include <Math/LinearAlgebra.hpp>
#include <Math/Matrix.hpp>
#include <Math/StaticArrays.hpp>
#include <Utilities/Invariant.hpp>
#include <algorithm>
#include <array>
#include <benchmark/benchmark.h>
#include <concepts>
#include <cstdint>
#include <random>
#include <ranges>

// auto x = Dual<Dual<double, 4>, 2>{1.0};
// auto y = x * 3.4;
namespace poly::math {
static_assert(std::convertible_to<int, Dual<double, 4>>);
static_assert(std::convertible_to<int, Dual<Dual<double, 4>, 2>>);

template <class T> struct URand {};

template <class T, ptrdiff_t N> struct URand<Dual<T, N>> {
  auto operator()(std::mt19937_64 &rng) -> Dual<T, N> {
    Dual<T, N> x{URand<T>{}(rng)};
    for (size_t i = 0; i < N; ++i) x.gradient()[i] = URand<T>{}(rng);
    return x;
  }
};
template <> struct URand<double> {
  auto operator()(std::mt19937_64 &rng) -> double {
    return std::uniform_real_distribution<double>(-2, 2)(rng);
  }
};

template <typename T>
constexpr void evalpoly(MutSquarePtrMatrix<T> B, MutSquarePtrMatrix<T> A,
                        SquarePtrMatrix<T> C, const auto &p) {
  ptrdiff_t N = p.size();
  invariant(N > 0);
  invariant(ptrdiff_t(B.numRow()), ptrdiff_t(C.numRow()));
  if (N & 1) std::swap(A, B);
  B << p[0] * C + p[1] * I;
  for (ptrdiff_t i = 2; i < N; ++i) {
    std::swap(A, B);
    B << A * C + p[i] * I;
  }
}

template <AbstractMatrix T> constexpr auto opnorm1(const T &A) {
  using S = decltype(extractDualValRecurse(std::declval<utils::eltype_t<T>>()));
  auto [M, N] = shape(A);
  invariant(M > 0);
  invariant(N > 0);
  S a{};
  for (ptrdiff_t n = 0; n < N; ++n) {
    S s{};
    for (ptrdiff_t m = 0; m < M; ++m)
      s += std::abs(extractDualValRecurse(A[m, n]));
    a = std::max(a, s);
  }
  return a;
  // Vector<S> v{N};
  // for (ptrdiff_t n = 0; n < N; ++n)
  //   v[n] = std::abs(extractDualValRecurse(A[0, n]));
  // for (ptrdiff_t m = 1; m < M; ++m)
  //   for (ptrdiff_t n = 0; n < N; ++n)
  //     v[n] += std::abs(extractDualValRecurse(A[m, n]));
  // return *std::max_element(v.begin(), v.end());
}

/// computes ceil(log2(x)) for x >= 1
constexpr auto log2ceil(double x) -> ptrdiff_t {
  invariant(x >= 1);
  uint64_t u = std::bit_cast<uint64_t>(x) - 1;
  auto y = ptrdiff_t((u >> 52) - 1022);
  invariant(y >= 0);
  return y;
}

template <typename T> constexpr void expm(MutSquarePtrMatrix<T> A) {
  ptrdiff_t n = ptrdiff_t(A.numRow()), s = 0;
  SquareMatrix<T> A2{A * A}, U_{SquareDims<>{{n}}};
  MutSquarePtrMatrix<T> U{U_};
  if (double nA = opnorm1(A); nA <= 0.015) {
    U << A * (A2 + 60.0 * I);
    A << 12.0 * A2 + 120.0 * I;
  } else {
    SquareMatrix<T> B{SquareDims<>{{n}}};
    if (nA <= 2.1) {
      containers::TinyVector<double, 5> p0, p1;
      if (nA > 0.95) {
        p0 = {1.0, 3960.0, 2162160.0, 302702400.0, 8821612800.0};
        p1 = {90.0, 110880.0, 3.027024e7, 2.0756736e9, 1.76432256e10};
      } else if (nA > 0.25) {
        p0 = {1.0, 1512.0, 277200.0, 8.64864e6};
        p1 = {56.0, 25200.0, 1.99584e6, 1.729728e7};
      } else {
        p0 = {1.0, 420.0, 15120.0};
        p1 = {30.0, 3360.0, 30240.0};
      }
      evalpoly(B, U, A2, p0);
      U << A * B;
      evalpoly(A, B, A2, p1);
    } else {
      // s = std::max(unsigned(std::ceil(std::log2(nA / 5.4))), 0);
      s = nA > 5.4 ? log2ceil(nA / 5.4) : 0;
      double t = (s > 0) ? exp2(-s) : 0.0;
      if (s > 0) A2 *= (t * t);
      // here we take an estrin (instead of horner) approach to cut down flops
      SquareMatrix<T> A4{A2 * A2}, A6{A2 * A4};
      B << A6 * (A6 + 16380 * A4 + 40840800 * A2) +
             (33522128640 * A6 + 10559470521600 * A4 + 1187353796428800 * A2) +
             32382376266240000 * I;
      U << A * B;
      if (s & 1) {  // we have an odd number of swaps at the end
        A << U * t; // copy data to `A`, so we can swap and make it even
        std::swap(A, U);
      } else if (s > 0) U *= t;
      A << A6 * (182 * A6 + 960960 * A4 + 1323241920 * A2) +
             (670442572800 * A6 + 129060195264000 * A4 +
              7771770303897600 * A2) +
             64764752532480000 * I;
    }
  }
  for (auto &&[a, u] : std::ranges::zip_view(A, U))
    std::tie(a, u) = std::make_pair(a + u, a - u);
  LU::ldiv(U, MutPtrMatrix<T>(A));
  for (; s--; std::swap(A, U)) U << A * A;
}

} // namespace poly::math

using poly::math::Dual, poly::math::SquareDims, poly::math::SquareMatrix,
  poly::math::URand;

auto expwork(const auto &A) {
  SquareMatrix<poly::math::eltype_t<decltype(A)>> C{SquareDims{A.numRow()}},
    B{A};
  expm(B);
  for (size_t i = 0; i < 8; ++i) {
    expm(C << A * exp2(-double(i)));
    B += C;
  }
  return B;
}
void expbench(const auto &A) {
  auto B{expwork(A)};
  for (auto &b : B) benchmark::DoNotOptimize(b);
}

static void BM_expm(benchmark::State &state) {
  std::mt19937_64 rng0;
  SquareMatrix<double> A{SquareDims{{state.range(0)}}};
  for (auto &a : A) a = URand<double>{}(rng0);
  for (auto b : state) expbench(A);
}
BENCHMARK(BM_expm)->DenseRange(2, 10, 1);
static void BM_expm_dual4(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<double, 8>;
  SquareMatrix<D> A{SquareDims{{state.range(0)}}};
  for (auto &a : A) a = URand<D>{}(rng0);
  for (auto b : state) expbench(A);
}
BENCHMARK(BM_expm_dual4)->DenseRange(2, 10, 1);

static void BM_expm_dual8x2(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<Dual<double, 8>, 2>;
  SquareMatrix<D> A{SquareDims{{state.range(0)}}};
  for (auto &a : A) a = URand<D>{}(rng0);
  for (auto b : state) expbench(A);
}
BENCHMARK(BM_expm_dual8x2)->DenseRange(2, 10, 1);
/*
using D8D2 = Dual<Dual<double, 8>, 2>;
using SMDD = SquareMatrix<D8D2>;
#ifdef __INTEL_LLVM_COMPILER
using SMDD0 = poly::math::ManagedArray<D8D2, SquareDims<>, 0>;
#else
using SMDD0 = poly::math::ManagedArray<D8D2, SquareDims<>>;
#endif
#pragma omp declare reduction(+ : SMDD0 : omp_out += omp_in)                   \
  initializer(omp_priv = SMDD0{omp_orig.dim(), D8D2{}})

static void BM_expm_dual8x2_threads(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = D8D2;
  ptrdiff_t dim = state.range(0);
  SquareMatrix<D> A{SquareDims{{dim}}};
  for (auto &a : A) a = URand<D>{}(rng0);
  for (auto bch : state) {
    SMDD0 B{SquareDims{{dim}}};
    B.fill(D{0});
#pragma omp parallel for reduction(+ : B)
    for (int i = 0; i < 1000; ++i) B += expwork(A);
    for (auto &b : B) benchmark::DoNotOptimize(b);
  }
}
BENCHMARK(BM_expm_dual8x2_threads)->DenseRange(2, 10, 1);
*/
