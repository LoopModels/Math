#include <gtest/gtest.h>
#ifndef USE_MODULE
#include "Math/ManagedArray.cxx"
#include "Math/MatrixDimensions.cxx"
#include "Math/NormalForm.cxx"
#include <random>
#else

import ManagedArray;
import MatDim;
import NormalForm;
import STL;
#endif

using math::DenseMatrix, math::DenseDims, math::row, math::col;

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(OrthogonalizeMatricesTest, BasicAssertions) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(-3, 3);

  const size_t M = 7;
  const size_t N = 7;
  DenseMatrix<int64_t> A(DenseDims<>{row(M), col(N)});
  DenseMatrix<int64_t> B(DenseDims<>{row(N), col(N)});
  const size_t iters = 1000;
  for (size_t i = 0; i < iters; ++i) {
    for (auto &&a : A) a = distrib(gen);
    // std::cout << "Random A =\n" << A << "\n";
    A = math::orthogonalize(std::move(A));
    // std::cout << "Orthogonal A =\n" << A << "\n";
    // note, A'A is not diagonal
    // but AA' is
    B = A * A.t();
    // std::cout << "A'A =\n" << B << "\n";
#if !defined(__clang__) && defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-else"
#endif
    for (size_t m = 0; m < M; ++m)
      for (size_t n = 0; n < N; ++n)
        if (m != n) EXPECT_EQ((B[m, n]), 0);
#if !defined(__clang__) && defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
  }
}
