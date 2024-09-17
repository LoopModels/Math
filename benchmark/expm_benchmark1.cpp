#include "include/expm.hpp"

static void BM_expm_dual1(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<double, 1>;
  SquareMatrix<D> A{SquareDims{math::row(state.range(0))}};
  for (auto &&a : A) a = URand<D>{}(rng0);
  for (auto b : state) expbench(A);
}

BENCHMARK(BM_expm_dual1)->DenseRange(2, 10, 1);
