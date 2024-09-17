#include "include/expm.hpp"

static void BM_expm(benchmark::State &state) {
  std::mt19937_64 rng0;
  SquareMatrix<double> A{SquareDims{math::row(state.range(0))}};
  for (auto &&a : A) a = URand<double>{}(rng0);
  for (auto b : state) expbench(A);
}
BENCHMARK(BM_expm)->DenseRange(2, 10, 1);
