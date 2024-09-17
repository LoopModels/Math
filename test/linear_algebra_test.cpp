#include <gtest/gtest.h>

#ifndef USE_MODULE
#include "Math/LinearAlgebra.cxx"
#include "Math/ManagedArray.cxx"
#include "Math/MatrixDimensions.cxx"
#include "Math/Rational.cxx"
#include "Math/Reductions.cxx"
#include "Utilities/Invariant.cxx"
#include <random>
#else

import Invariant;
import LinearAlgebra;
import ManagedArray;
import MatDim;
import Rational;
import Reductions;
import STL;
#endif

using namespace math;

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(LinearAlgebraTest, BasicAssertions) {
  const SquareMatrix<Rational> identity = SquareMatrix<Rational>::identity(4);
  SquareMatrix<int64_t> A(SquareDims{row(4)});
  A[0, 0] = 2;
  A[0, 1] = -10;
  A[0, 2] = 6;
  A[0, 3] = -9;
  A[1, 0] = -10;
  A[1, 1] = 6;
  A[1, 2] = 5;
  A[1, 3] = -7;
  A[2, 0] = -1;
  A[2, 1] = -7;
  A[2, 2] = 0;
  A[2, 3] = 1;
  A[3, 0] = -8;
  A[3, 1] = 9;
  A[3, 2] = -2;
  A[3, 3] = 4;

  auto opt_luf = LU::fact(A);
  ASSERT_TRUE(opt_luf.has_value());
  auto &LUF = *opt_luf;
  Matrix<Rational> B0 = A;
  std::cout << "A = \n" << A << "\nB = \n" << B0 << "\n";
  std::cout << LUF;

  auto B1 = B0;
  EXPECT_FALSE(LUF.ldivrat(B1));
  std::cout << "LUF.ldiv(B) = \n" << B1 << "\n";
  EXPECT_EQ(B1, identity);
  std::cout << "I = " << identity << "\n";

  EXPECT_FALSE(LUF.rdivrat(B0));
  std::cout << "LUF.rdiv(B) = \n" << B0 << "\n";
  EXPECT_EQ(B0, identity);
}

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(DoubleFactorization, BasicAssertions) {
  SquareMatrix<double> A(SquareDims{row(7)}), B(SquareDims{row(7)}),
    C(SquareDims{row(7)}), D(SquareDims{row(7)});
  std::mt19937 gen(0);
  std::uniform_real_distribution<double> dist(-1, 1);
  for (ptrdiff_t i = 0; i < 10; ++i) {
    for (auto &a : A) a = dist(gen);
    for (auto &b : B) b = dist(gen);
    C << B;
    // LU
    // B = A \ B
    // C == A*B == A * (A \ B)
    LU::fact(A).ldiv(MutPtrMatrix<double>(B));
    EXPECT_LT(norm2(A * B - C), 1e-10);
    B << C;
    D << A;
    LU::ldiv(A, MutPtrMatrix<double>(B));
    EXPECT_LT(norm2(D * B - C), 1e-10);

    // LDL; make `A` symmetric
    D << A + A.t();
    A << D;
    B << C;
    // B = A \ B
    // C == A*B == A * (A \ B)
    LDL::factorize<>(D).ldiv(MutPtrMatrix<double>(B));
    EXPECT_LT(norm2(A * B - C), 1e-10);
    B << C;
    D << A;
    LDL::ldiv<>(A, MutPtrMatrix<double>(B));
    EXPECT_LT(norm2(D * B - C), 1e-10);
  }
}
