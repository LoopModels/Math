
#include "Math/Array.cxx"
#include "Math/AxisTypes.cxx"
#include "Math/ManagedArray.cxx"
#include "Math/Ranges.cxx"
#include "Math/ScalarizeViaCastArrayOps.cxx"
#include "include/randdual.hpp"
#include <benchmark/benchmark.h>
#include <concepts>
#include <cstddef>
#include <random>
#include <type_traits>

using math::Dual, math::SquareDims, math::SquareMatrix, math::MutArray,
  math::Array, math::URand, containers::tie, containers::Tuple;

namespace {
#ifndef NDEBUG
template <typename A, typename... As, typename B, typename... Bs>
constexpr void tuplecheck(Tuple<A, As...> &, const Tuple<B, Bs...> &) {
  using C = math::scalarize_via_cast_t<
    std::remove_cvref_t<decltype(std::declval<A>().view())>>;
  static_assert(
    !std::same_as<C, void> &&
    detail::ScalarizeViaCastTo<C, As..., decltype(std::declval<B>().view()),
                               Bs...>());
}
#endif

template <typename T, typename S>
[[gnu::noinline]] void eltmul(MutArray<T, S> A, double t) {
  A *= t;
}

template <typename T, typename S>
[[gnu::noinline]] void eltadd(MutArray<T, S> C, Array<T, S> A, Array<T, S> B) {
  static_assert(
    std::same_as<double, math::scalarize_via_cast_t<MutArray<T, S>>>);
  static_assert(
    std::same_as<double, math::scalarize_via_cast_t<decltype(A + B)>>);
  C << A + B;
}
template <typename T, typename S>
[[gnu::noinline]] void eltaddsub(MutArray<T, S> C, MutArray<T, S> D,
                                 Array<T, S> A, Array<T, S> B) {
#ifndef NDEBUG
  {
    auto lval{tie(C, D)};
    tuplecheck(lval, Tuple(A + B, A - B));
  }
#endif
  tie(C, D) << Tuple(A + B, A - B);
}

template <ptrdiff_t N> void BM_dualNdoublemul(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<double, N>;
  SquareMatrix<D> A{SquareDims{math::row(state.range(0))}};
  for (auto &&a : A) a = URand<D>{}(rng0);
  double t = 1.00000000001;
  for (auto b : state) eltmul(A, t);
}

template <ptrdiff_t M, ptrdiff_t N>
void BM_dualMxNdoublemul(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<Dual<double, M>, N>;
  SquareMatrix<D> A{SquareDims{math::row(state.range(0))}};
  for (auto &&a : A) a = URand<D>{}(rng0);
  double t = 1.00000000001;
  for (auto b : state) eltmul(A, t);
}

template <ptrdiff_t N> void BM_dualNadd(benchmark::State &state) {
  std::mt19937_64 rng0;
  using T = Dual<double, N>;
  SquareDims<> dim{math::row(state.range(0))};
  SquareMatrix<T> A{dim}, B{dim}, C{dim};
  for (auto &&a : A) a = URand<T>{}(rng0);
  for (auto &&b : B) b = URand<T>{}(rng0);
  for (auto b : state) eltadd(C, A, B);
}
template <ptrdiff_t M, ptrdiff_t N>
void BM_dualMxNadd(benchmark::State &state) {
  std::mt19937_64 rng0;
  using T = Dual<Dual<double, M>, N>;
  SquareDims<> dim{math::row(state.range(0))};
  SquareMatrix<T> A{dim}, B{dim}, C{dim};
  for (auto &&a : A) a = URand<T>{}(rng0);
  for (auto &&b : B) b = URand<T>{}(rng0);
  for (auto b : state) eltadd(C, A, B);
}
template <ptrdiff_t N> void BM_dualNaddsub(benchmark::State &state) {
  std::mt19937_64 rng0;
  using T = Dual<double, N>;
  SquareDims<> dim{math::row(state.range(0))};
  SquareMatrix<T> A{dim}, B{dim}, C{dim}, D{dim};
  for (auto &&a : A) a = URand<T>{}(rng0);
  for (auto &&b : B) b = URand<T>{}(rng0);
  for (auto b : state) eltaddsub(C, D, A, B);
}
template <ptrdiff_t M, ptrdiff_t N>
void BM_dualMxNaddsub(benchmark::State &state) {
  std::mt19937_64 rng0;
  using T = Dual<Dual<double, M>, N>;
  SquareDims<> dim{math::row(state.range(0))};
  SquareMatrix<T> A{dim}, B{dim}, C{dim}, D{dim};
  for (auto &&a : A) a = URand<T>{}(rng0);
  for (auto &&b : B) b = URand<T>{}(rng0);
  for (auto b : state) eltaddsub(C, D, A, B);
}
} // namespace

BENCHMARK(BM_dualNadd<1>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualNadd<2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualNadd<3>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualNadd<4>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualNadd<5>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualNadd<6>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualNadd<7>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualNadd<8>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNadd<1, 2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNadd<2, 2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNadd<3, 2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNadd<4, 2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNadd<5, 2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNadd<6, 2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNadd<7, 2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNadd<8, 2>)->DenseRange(2, 10, 1);

BENCHMARK(BM_dualNaddsub<1>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualNaddsub<2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualNaddsub<3>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualNaddsub<4>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualNaddsub<5>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualNaddsub<6>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualNaddsub<7>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualNaddsub<8>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNaddsub<1, 2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNaddsub<2, 2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNaddsub<3, 2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNaddsub<4, 2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNaddsub<5, 2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNaddsub<6, 2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNaddsub<7, 2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNaddsub<8, 2>)->DenseRange(2, 10, 1);

BENCHMARK(BM_dualNdoublemul<1>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualNdoublemul<2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualNdoublemul<3>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualNdoublemul<4>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualNdoublemul<5>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualNdoublemul<6>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualNdoublemul<7>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualNdoublemul<8>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNdoublemul<1, 2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNdoublemul<2, 2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNdoublemul<3, 2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNdoublemul<4, 2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNdoublemul<5, 2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNdoublemul<6, 2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNdoublemul<7, 2>)->DenseRange(2, 10, 1);
BENCHMARK(BM_dualMxNdoublemul<8, 2>)->DenseRange(2, 10, 1);
