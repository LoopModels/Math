#include <gtest/gtest.h>
#ifndef USE_MODULE
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
#include <utility>

#include "Alloc/Arena.cxx"
#include "Containers/Pair.cxx"
#include "Math/Array.cxx"
#include "Math/ArrayConcepts.cxx"
#include "Math/Comparisons.cxx"
#include "Math/Constructors.cxx"
#include "Math/LinearAlgebra.cxx"
#include "Math/ManagedArray.cxx"
#include "Math/MatrixDimensions.cxx"
#include "Math/NormalForm.cxx"
#include "Math/UniformScaling.cxx"
#include "Utilities/MatrixStringParse.cxx"
#else

import Arena;
import Array;
import ArrayConcepts;
import ArrayConstructors;
import ArrayParse;
import Comparisons;
import LinearAlgebra;
import ManagedArray;
import MatDim;
import NormalForm;
import Pair;
import STL;
import UniformScaling;
#endif

using namespace math;
using utils::operator""_mat;

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(OrthogonalizationTest, BasicAssertions) {
  SquareMatrix<int64_t> A(SquareDims<>{math::row(4)});
  std::cout << "\n\n\n========\n========\n========\n\n";
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(-10, 10);
  ptrdiff_t orth_any_count = 0;
  ptrdiff_t orth_max_count = 0;
  ptrdiff_t orth_count = 0;
  ptrdiff_t lu_failed_count = 0;
  ptrdiff_t inv_failed_count = 0;
  ptrdiff_t num_iters = 1000;
  IntMatrix<> B(DenseDims<>{row(4), col(8)});
  SquareMatrix<int64_t> I4 = SquareMatrix<int64_t>::identity(4);
  for (ptrdiff_t i = 0; i < num_iters; ++i) {
    for (ptrdiff_t n = 0; n < 4; ++n)
      for (ptrdiff_t m = 0; m < 8; ++m) B[n, m] = distrib(gen);
    // std::cout << "\nB = " << B << "\n";
    auto [K, included] = NormalForm::orthogonalize(B);
    orth_count += included.size();
    orth_any_count += (!included.empty());
    orth_max_count += (included.size() == 4);
    // std::cout << "included.size() = " << included.size() << "\n";
    if (included.size() == 4) {
      for (ptrdiff_t n = 0; n < 4; ++n) {
        ptrdiff_t m = 0;
        for (auto mb : included) A[n, m++] = B[n, mb];
      }
      std::cout << "K=\n" << K << "\n";
      std::cout << "A=\n" << A << "\n";
      EXPECT_EQ(K * A, I4);
      EXPECT_EQ(K * A, I);
    } else {
      // std::cout << "K= " << K << "\nB= " << B << "\n";
      // printVector(std::cout << "included = ", included)
      // << "\n";
      if (auto optlu = LU::fact(K)) {
        if (auto opt_a2 = (*optlu).inv()) {
          auto &A2 = *opt_a2;
          for (ptrdiff_t n = 0; n < 4; ++n) {
            for (ptrdiff_t j = 0; j < included.size(); ++j) {
              // std::cout << "A2(" << n << ", " << j << ") = " << A2(n, j)
              //              << "; B(" << n << ", " << included[j]
              //              << ") = " << B(n, included[j]) << "\n";
              EXPECT_EQ((A2[n, j]), (B[n, included[j]]));
            }
          }
        } else {
          ++inv_failed_count;
        }
      } else {
        ++lu_failed_count;
        std::cout << "B = " << B << "\nK = " << K << "\n";
        continue;
      }
    }
  }
  std::cout << "Mean orthogonalized: " << double(orth_count) / double(num_iters)
            << "\nOrthogonalization succeeded on at least one: "
            << orth_any_count << " / " << num_iters
            << "\nOrthogonalization succeeded on 4: " << orth_max_count << " / "
            << num_iters << "\nLU fact failed count: " << lu_failed_count
            << " / " << num_iters
            << "\nInv fact failed count: " << inv_failed_count << " / "
            << num_iters << "\n";

  B[0, 0] = 1;
  B[1, 0] = 0;
  B[2, 0] = 1;
  B[3, 0] = 0;
  B[0, 1] = 0;
  B[1, 1] = 1;
  B[2, 1] = 0;
  B[3, 1] = 1;
  B[0, 2] = 1;
  B[1, 2] = 0;
  B[2, 2] = 0;
  B[3, 2] = 0;
  B[0, 3] = 0;
  B[1, 3] = 1;
  B[2, 3] = 0;
  B[3, 3] = 0;
  B[0, 4] = 0;
  B[1, 4] = 0;
  B[2, 4] = 1;
  B[3, 4] = 0;
  B[0, 5] = 0;
  B[1, 5] = 0;
  B[2, 5] = 0;
  B[3, 5] = 1;
  std::cout << "B_orth_motivating_example = " << B << "\n";
  auto [K, included] = NormalForm::orthogonalize(B);
  // printVector(std::cout << "K = " << K << "\nincluded = ",
  //                            included)
  //   << "\n";
  EXPECT_EQ(included.size(), 4);
  for (ptrdiff_t i = 0; i < 4; ++i) EXPECT_EQ(included[i], i);
  for (ptrdiff_t n = 0; n < 4; ++n) {
    ptrdiff_t m = 0;
    for (auto mb : included) {
      A[n, m] = B[n, mb];
      ++m;
    }
  }
  IntMatrix<> KA{K * A};
  std::cout << "A = " << A << "\nA * K = " << KA << "\n";
  EXPECT_TRUE(KA == I4);
}

auto isHNF(PtrMatrix<int64_t> A) -> bool {
  const auto [M, N] = shape(A);
  // l is lead
  Col<> l = {};
  for (ptrdiff_t m = 0; m < M; ++m) {
    // all entries must be 0
    for (ptrdiff_t n = 0; n < l; ++n)
      if (A[m, n]) return false;
    // now search for next lead
    while ((l < N) && A[m, l] == 0) ++l;
    if (l == N) continue;
    int64_t Aml = A[m, l];
    if (Aml < 0) return false;
    for (ptrdiff_t r = 0; r < m; ++r) {
      int64_t Arl = A[r, l];
      if ((Arl >= Aml) || (Arl < 0)) return false;
    }
  }
  return true;
}

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(Hermite, BasicAssertions) {
  {
    IntMatrix<> A43(DenseDims<>{row(4), col(3)});
    A43[0, 0] = 2;
    A43[1, 0] = 3;
    A43[2, 0] = 6;
    A43[3, 0] = 2;
    A43[0, 1] = 5;
    A43[1, 1] = 6;
    A43[2, 1] = 1;
    A43[3, 1] = 6;
    A43[0, 2] = 8;
    A43[1, 2] = 3;
    A43[2, 2] = 1;
    A43[3, 2] = 1;
    std::cout << "A=\n" << A43 << "\n";
    IntMatrix<> H = A43;
    SquareMatrix<int64_t> U{SquareDims{H.numRow()}};
    NormalForm::hermite(H, U);
    std::cout << "H=\n" << H << "\nU=\n" << U << "\n";

    EXPECT_TRUE(isHNF(H));
    EXPECT_TRUE(H == U * A43);

    for (ptrdiff_t i = 0; i < 3; ++i) A43[2, i] = A43[0, i] + A43[1, i];
    std::cout << "\n\n\n=======\n\nA=\n" << A43 << "\n";
    H << A43;
    NormalForm::hermite(H, U);
    std::cout << "H=\n" << H << "\nU=\n" << U << "\n";
    EXPECT_TRUE(isHNF(H));
    EXPECT_TRUE(H == U * A43);
  }
  {
    SquareMatrix<int64_t> A(SquareDims<>{math::row(4)});
    A[0, 0] = 3;
    A[1, 0] = -6;
    A[2, 0] = 7;
    A[3, 0] = 7;
    A[0, 1] = 7;
    A[1, 1] = -8;
    A[2, 1] = 10;
    A[3, 1] = 6;
    A[0, 2] = -5;
    A[1, 2] = 8;
    A[2, 2] = 7;
    A[3, 2] = 3;
    A[0, 3] = -5;
    A[1, 3] = -6;
    A[2, 3] = 8;
    A[3, 3] = -1;
    IntMatrix<> H = A;
    SquareMatrix<int64_t> U{SquareDims{H.numRow()}};
    NormalForm::hermite(H, U);
    std::cout << "\n\n\n====\n\nH=\n" << H << "\nU=\n" << U << "\n";
    EXPECT_TRUE(isHNF(H));
    EXPECT_TRUE(H == U * A);
  }
  {
    IntMatrix<> A{"[1 -3 0 -2 0 0 -1 -1 0 0 -1 0 0 0 0 0 0 "
                  "0 0 0 0 0; 0 1 0 1 0 0 0 1 0 "
                  "0 0 0 0 0 0 0 0 0 0 0 0 0; 0 1 0 0 0 0 "
                  "1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
                  "0; 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 "
                  "0 0 0 0; 0 -1 1 -1 1 0 0 -1 1 "
                  "0 0 0 0 0 0 0 0 0 0 0 0 0; 0 -1 1 0 0 1 "
                  "-1 0 0 0 0 0 0 0 0 0 0 0 0 0 "
                  "0 0; 0 -1 1 -1 1 0 0 0 0 1 -1 0 0 0 0 0 "
                  "0 0 0 0 0 0; -1 0 0 0 0 0 0 0 "
                  "0 0 0 1 0 0 0 0 0 0 0 0 0 0; 0 -1 0 0 0 "
                  "0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 "
                  "0 0; 0 0 -1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 "
                  "0 0 0 0 0; 0 0 0 -1 0 0 0 0 0 "
                  "0 0 0 0 0 1 0 0 0 0 0 0 0; 0 0 0 0 -1 0 "
                  "0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 "
                  "0; 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 1 0 "
                  "0 0 0 0; 0 0 0 0 0 0 -1 0 0 0 "
                  "0 0 0 0 0 0 0 1 0 0 0 0; 0 0 0 0 0 0 0 "
                  "-1 0 0 0 0 0 0 0 0 0 0 1 0 0 "
                  "0; 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 "
                  "0 1 0 0; 0 0 0 0 0 0 0 0 0 -1 "
                  "0 0 0 0 0 0 0 0 0 0 1 0; 0 0 0 0 0 0 0 "
                  "0 0 0 -1 0 0 0 0 0 0 0 0 0 0 "
                  "1]"_mat};
    IntMatrix<> H = A;
    SquareMatrix<int64_t> U{SquareDims{H.numRow()}};
    NormalForm::hermite(H, U);
    std::cout << "\n\n\n====\n\nH=" << H << "\nU=" << U << "\n";
    EXPECT_TRUE(isHNF(H));
    EXPECT_TRUE(H == U * A);
  }
  {
    IntMatrix<> A = "[-3 -1 1; 0 0 -2]"_mat;
    IntMatrix<> H = A;
    SquareMatrix<int64_t> U{SquareDims{H.numRow()}};
    NormalForm::hermite(H, U);
    EXPECT_TRUE(isHNF(H));
    EXPECT_TRUE(U * A == H);
    std::cout << "A = \n" << A << "\nH =\n" << H << "\nU =\n" << U << "\n";
  }
  {
    IntMatrix<> A =
      "[3 3 -3 1 0 -1 -2 1 1 2 -1; 3 3 -3 1 1 -3 2 0 3 0 -3; 2 -3 -2 -1 1 -2 3 3 3 3 -3]"_mat;
    IntMatrix<> H = A;
    SquareMatrix<int64_t> U{SquareDims{H.numRow()}};
    NormalForm::hermite(H, U);
    EXPECT_TRUE(isHNF(H));
    EXPECT_TRUE(U * A == H);
    std::cout << "A = \n" << A << "\nH =\n" << H << "\nU =\n" << U << "\n";
  }
}

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(NullSpaceTests, BasicAssertions) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(-10, 100);

  // ptrdiff_t numIters = 1000;
  ptrdiff_t num_iters = 1;
  for (ptrdiff_t num_col = 2; num_col < 11; num_col += 2) {
    IntMatrix<> B(DenseDims<>{row(8), col(num_col)});
    ptrdiff_t null_dim = 0;
    IntMatrix<> Z;
    DenseMatrix<int64_t> NS;
    for (ptrdiff_t i = 0; i < num_iters; ++i) {
      for (auto &&b : B) {
        b = distrib(gen);
        b = b > 10 ? 0 : b;
      }
      NS = NormalForm::nullSpace(B);
      null_dim += ptrdiff_t(NS.numRow());
      Z = NS * B;
      if (!allZero(Z)) {
        std::cout << "B = \n"
                  << B << "\nNS = \n"
                  << NS << "\nZ = \n"
                  << Z << "\n";
      }
      for (auto &z : Z) EXPECT_EQ(z, 0);
      EXPECT_EQ(NormalForm::nullSpace(std::move(NS)).numRow(), 0);
    }
    std::cout << "Average tested null dim = "
              << double(null_dim) / double(num_iters) << "\n";
  }
}

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(SimplifySystemTests, BasicAssertions) {
  IntMatrix<> A = "[2 4 5 5 -5; -4 3 -4 -3 -1; 1 0 -2 1 -4; -4 -2 3 -2 -1]"_mat;
  IntMatrix<> B = "[-6 86 -27 46 0 -15; -90 -81 91 44 -2 78; 4 -54 -98 "
                  "80 -10 82; -98 -15 -28 98 82 87]"_mat;
  NormalForm::solveSystem(A, B);
  IntMatrix<> sA = "[-3975 0 0 0 -11370; 0 -1325 0 0 -1305; "
                   "0 0 -265 0 -347; 0 0 0 265 -1124]"_mat;
  IntMatrix<> true_b =
    "[-154140 -128775 -205035 317580 83820 299760; -4910 -21400 -60890 "
    "44820 14480 43390; -1334 -6865 -7666 8098 -538 9191; -6548 -9165 "
    "-24307 26176 4014 23332]"_mat;

  EXPECT_EQ(sA, A);
  EXPECT_EQ(true_b, B);

  IntMatrix<> C = "[1 1 0; 0 1 1; 1 2 1]"_mat;
  IntMatrix<> D = "[1 0 0; 0 1 0; 0 0 1]"_mat;
  NormalForm::simplifySystem(C, D);
  IntMatrix<> true_c = "[1 0 -1; 0 1 1]"_mat;
  IntMatrix<> true_d = "[1 -1 0; 0 1 0]"_mat;
  EXPECT_EQ(true_c, C);
  EXPECT_EQ(true_d, D);
}

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(BareissTests, BasicAssertions) {
  IntMatrix<> A =
    "[-4 3 -2 2 -5; -5 1 -1 2 -5; -1 0 5 -3 2; -4 5 -4 -2 -4]"_mat;
  auto piv = NormalForm::bareiss(A);
  IntMatrix<> B =
    "[-4 3 -2 2 -5; 0 11 -6 2 -5; 0 0 56 -37 32; 0 0 0 -278 136]"_mat;
  EXPECT_EQ(A, B);
  Vector<ptrdiff_t> true_piv{std::array{0, 1, 2, 3}};
  EXPECT_EQ(piv, true_piv);

  IntMatrix<> C = "[-2 -2 -1 -2 -1; 1 1 2 2 -2; -2 2 2 -1 "
                  "-1; 0 0 -2 1 -1; -1 -2 2 1 -1]"_mat;
  IntMatrix<> D = "[-2 -2 -1 -2 -1; 0 -8 -6 -2 0; 0 0 -12 -8 "
                  "20; 0 0 0 -28 52; 0 0 0 0 -142]"_mat;
  auto pivots = NormalForm::bareiss(C);
  EXPECT_EQ(C, D);
  auto true_pivots = Vector<ptrdiff_t, 16>{"[0 2 2 3 4]"_mat};
  EXPECT_EQ(pivots, true_pivots);
}

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(InvTest, BasicAssertions) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(-10, 10);
  alloc::OwningArena<> alloc;
  const ptrdiff_t num_iters = 1000;
  for (ptrdiff_t dim = 1; dim < 5; ++dim) {
    auto s0 = alloc.scope();
    MutSquarePtrMatrix<int64_t> B{square_matrix<int64_t>(&alloc, dim)};
    for (ptrdiff_t i = 0; i < num_iters; ++i) {
      while (true) {
        for (ptrdiff_t n = 0; n < dim * dim; ++n) B.data()[n] = distrib(gen);
        if (NormalForm::rank(alloc, B) == dim) break;
      }
      auto s1 = alloc.scope();
      // Da * B^{-1} = Binv0
      // Da = Binv0 * B
      MutSquarePtrMatrix<int64_t> Da{square_matrix<int64_t>(&alloc, dim)};
      Da << B;
      auto Binv0 = NormalForm::inv(&alloc, Da);
      MutSquarePtrMatrix<int64_t> Bc{square_matrix<int64_t>(&alloc, dim)};
      Bc << B;
      auto [Binv1, s] = NormalForm::scaledInv(&alloc, Bc);
      EXPECT_TRUE(Da.isDiagonal());
      EXPECT_EQ((Binv0 * B), Da);
      Da.diag() << s;
      if (B * Binv1 != Da) {
        std::cout << "\nB = " << B << "\nDa = " << Da << "\nBinv0 = " << Binv0
                  << "\nBinv1 = " << Binv1 << "\ns = " << s << "\n";
      }
      EXPECT_EQ(B * Binv1, Da);
    }
  }
}
