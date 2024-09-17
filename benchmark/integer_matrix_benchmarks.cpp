#include <benchmark/benchmark.h>

#ifndef USE_MODULE
#include "Math/AxisTypes.cxx"
#include "Math/Indexing.cxx"
#include "Math/ManagedArray.cxx"
#include "Math/MatrixDimensions.cxx"
#include "Math/NormalForm.cxx"
#include <cstddef>
#include <cstdint>
#include <random>
#else
import Array;
import ArrayParse;
import NormalForm;
import STL;
#endif

static void BM_normal_form(benchmark::State &state) {
  using math::_;
  std::mt19937_64 rng0;
  std::uniform_int_distribution<> distrib(-10, 10);
  ptrdiff_t d = state.range(0), num_iter = 100;
  math::DenseDims dim{math::DenseDims<>{math::row(num_iter * d), math::col(d)}};
  math::DenseMatrix<int64_t> A{dim}, B{dim}, C{dim}, D{dim};
  for (auto &&x : C) x = distrib(rng0);
  for (auto &&x : D) x = distrib(rng0);
  for (auto b : state) {
    A << C;
    B << D;
    for (ptrdiff_t n = 0; n < num_iter; ++n)
      math::NormalForm::solveSystem(A[_(n * d, (n + 1) * d), _],
                                    B[_(n * d, (n + 1) * d), _]);
  }
}
BENCHMARK(BM_normal_form)->DenseRange(2, 10, 1);
