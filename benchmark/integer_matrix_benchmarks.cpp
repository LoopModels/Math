#include "Math/LinearAlgebra.hpp"
#include "Math/NormalForm.hpp"
#include <Utilities/MatrixStringParse.hpp>
#include <benchmark/benchmark.h>
#include <random>

using namespace poly;

static void BM_normal_form(benchmark::State &state) {
  using math::_;
  std::mt19937_64 rng0;
  std::uniform_int_distribution<> distrib(-10, 10);
  ptrdiff_t d = state.range(0), numIter = 100;
  math::DenseDims dim{math::DenseDims<>{{numIter * d}, {d}}};
  math::DenseMatrix<int64_t> A{dim}, B{dim}, C{dim}, D{dim};
  for (auto &&x : C) x = distrib(rng0);
  for (auto &&x : D) x = distrib(rng0);
  for (auto b : state) {
    A << C;
    B << D;
    for (ptrdiff_t n = 0; n < numIter; ++n)
      math::NormalForm::solveSystem(A[_(n * d, (n + 1) * d), _],
                                    B[_(n * d, (n + 1) * d), _]);
  }
}
BENCHMARK(BM_normal_form)->DenseRange(2, 10, 1);

