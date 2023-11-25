#include "include/expm.hpp"

static void BM_expm_dual2x2(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<Dual<double, 2>, 2>;
  SquareMatrix<D> A{SquareDims{{state.range(0)}}};
  for (auto &&a : A) a = URand<D>{}(rng0);
  for (auto b : state) expbench(A);
}

BENCHMARK(BM_expm_dual2x2)->DenseRange(2, 10, 1);
