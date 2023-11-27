
#include "include/randdual.hpp"
#include <benchmark/benchmark.h>

using poly::math::Dual, poly::math::SquareDims, poly::math::SquareMatrix,
  poly::math::MutArray, poly::math::Array, poly::math::URand,
  poly::containers::tie, poly::containers::Tuple;

template <typename A, typename... As, typename B, typename... Bs>
constexpr void tuplecheck(Tuple<A, As...> &, const Tuple<B, Bs...> &) {
  using C = poly::math::scalarize_via_cast_t<
    std::remove_cvref_t<decltype(std::declval<A>().view())>>;
  static_assert(
    !std::same_as<C, void> &&
    poly::math::ScalarizeViaCastTo<C, As..., decltype(std::declval<B>().view()),
                                   Bs...>());
}

template <typename T, typename S>
[[gnu::noinline]] void eltmul(MutArray<T, S> A, double t) {
  A *= t;
}

template <typename T, typename S>
[[gnu::noinline]] void eltadd(MutArray<T, S> C, Array<T, S> A, Array<T, S> B) {
  static_assert(
    std::same_as<double, poly::math::scalarize_via_cast_t<MutArray<T, S>>>);
  static_assert(
    std::same_as<double, poly::math::scalarize_via_cast_t<decltype(A + B)>>);
  C << A + B;
}
template <typename T, typename S>
[[gnu::noinline]] void eltaddsub(MutArray<T, S> C, MutArray<T, S> D,
                                 Array<T, S> A, Array<T, S> B) {
  {
    auto lval{tie(C, D)};
    tuplecheck(lval, Tuple(A + B, A - B));
  }
  tie(C, D) << Tuple(A + B, A - B);
}

template <ptrdiff_t N> static void BM_dualNdoublemul(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<double, N>;
  SquareMatrix<D> A{SquareDims{{state.range(0)}}};
  for (auto &&a : A) a = URand<D>{}(rng0);
  double t = 1.00000000001;
  for (auto b : state) eltmul(A, t);
}

template <ptrdiff_t M, ptrdiff_t N>
static void BM_dualMxNdoublemul(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<Dual<double, M>, N>;
  SquareMatrix<D> A{SquareDims{{state.range(0)}}};
  for (auto &&a : A) a = URand<D>{}(rng0);
  double t = 1.00000000001;
  for (auto b : state) eltmul(A, t);
}

template <ptrdiff_t N> static void BM_dualNadd(benchmark::State &state) {
  std::mt19937_64 rng0;
  using T = Dual<double, N>;
  SquareDims<> dim{state.range(0)};
  SquareMatrix<T> A{dim}, B{dim}, C{dim};
  for (auto &&a : A) a = URand<T>{}(rng0);
  for (auto &&b : B) b = URand<T>{}(rng0);
  for (auto b : state) eltadd(C, A, B);
}
template <ptrdiff_t M, ptrdiff_t N>
static void BM_dualMxNadd(benchmark::State &state) {
  std::mt19937_64 rng0;
  using T = Dual<Dual<double, M>, N>;
  SquareDims<> dim{state.range(0)};
  SquareMatrix<T> A{dim}, B{dim}, C{dim};
  for (auto &&a : A) a = URand<T>{}(rng0);
  for (auto &&b : B) b = URand<T>{}(rng0);
  for (auto b : state) eltadd(C, A, B);
}
template <ptrdiff_t N> static void BM_dualNaddsub(benchmark::State &state) {
  std::mt19937_64 rng0;
  using T = Dual<double, N>;
  SquareDims<> dim{state.range(0)};
  SquareMatrix<T> A{dim}, B{dim}, C{dim}, D{dim};
  for (auto &&a : A) a = URand<T>{}(rng0);
  for (auto &&b : B) b = URand<T>{}(rng0);
  for (auto b : state) eltaddsub(C, D, A, B);
}
template <ptrdiff_t M, ptrdiff_t N>
static void BM_dualMxNaddsub(benchmark::State &state) {
  std::mt19937_64 rng0;
  using T = Dual<Dual<double, M>, N>;
  SquareDims<> dim{state.range(0)};
  SquareMatrix<T> A{dim}, B{dim}, C{dim}, D{dim};
  for (auto &&a : A) a = URand<T>{}(rng0);
  for (auto &&b : B) b = URand<T>{}(rng0);
  for (auto b : state) eltaddsub(C, D, A, B);
}

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
