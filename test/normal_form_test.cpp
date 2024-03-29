#include "Math/Array.hpp"
#include "Math/LinearAlgebra.hpp"
#include "Math/Math.hpp"
#include "Math/MatrixDimensions.hpp"
#include "Math/NormalForm.hpp"
#include "Utilities/MatrixStringParse.hpp"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
using namespace poly::math;
using poly::utils::operator""_mat;

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(OrthogonalizationTest, BasicAssertions) {
  SquareMatrix<int64_t> A(4);
  std::cout << "\n\n\n========\n========\n========\n\n";
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(-10, 10);
  ptrdiff_t orthAnyCount = 0;
  ptrdiff_t orthMaxCount = 0;
  ptrdiff_t orthCount = 0;
  ptrdiff_t luFailedCount = 0;
  ptrdiff_t invFailedCount = 0;
  ptrdiff_t numIters = 1000;
  IntMatrix<> B(DenseDims<>{{4}, {8}});
  SquareMatrix<int64_t> I4 = SquareMatrix<int64_t>::identity(4);
  for (ptrdiff_t i = 0; i < numIters; ++i) {
    for (ptrdiff_t n = 0; n < 4; ++n)
      for (ptrdiff_t m = 0; m < 8; ++m) B[n, m] = distrib(gen);
    // std::cout << "\nB = " << B << "\n";
    auto [K, included] = NormalForm::orthogonalize(B);
    orthCount += included.size();
    orthAnyCount += (!included.empty());
    orthMaxCount += (included.size() == 4);
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
        if (auto optA2 = (*optlu).inv()) {
          auto &A2 = *optA2;
          for (ptrdiff_t n = 0; n < 4; ++n) {
            for (ptrdiff_t j = 0; j < included.size(); ++j) {
              // std::cout << "A2(" << n << ", " << j << ") = " << A2(n, j)
              //              << "; B(" << n << ", " << included[j]
              //              << ") = " << B(n, included[j]) << "\n";
              EXPECT_EQ((A2[n, j]), (B[n, included[j]]));
            }
          }
        } else {
          ++invFailedCount;
        }
      } else {
        ++luFailedCount;
        std::cout << "B = " << B << "\nK = " << K << "\n";
        continue;
      }
    }
  }
  std::cout << "Mean orthogonalized: " << double(orthCount) / double(numIters)
            << "\nOrthogonalization succeeded on at least one: " << orthAnyCount
            << " / " << numIters
            << "\nOrthogonalization succeeded on 4: " << orthMaxCount << " / "
            << numIters << "\nLU fact failed count: " << luFailedCount << " / "
            << numIters << "\nInv fact failed count: " << invFailedCount
            << " / " << numIters << "\n";

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
  Col<> l = {0};
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
    IntMatrix<> A43(DenseDims<>{{4}, {3}});
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
    auto [H, U] = NormalForm::hermite(A43);
    std::cout << "H=\n" << H << "\nU=\n" << U << "\n";

    EXPECT_TRUE(isHNF(H));
    EXPECT_TRUE(H == U * A43);

    for (ptrdiff_t i = 0; i < 3; ++i) A43[2, i] = A43[0, i] + A43[1, i];
    std::cout << "\n\n\n=======\n\nA=\n" << A43 << "\n";
    auto [H2, U2] = NormalForm::hermite(A43);
    std::cout << "H=\n" << H2 << "\nU=\n" << U2 << "\n";
    EXPECT_TRUE(isHNF(H2));
    EXPECT_TRUE(H2 == U2 * A43);
  }
  {
    SquareMatrix<int64_t> A(4);
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
    auto [H3, U3] = NormalForm::hermite(A);
    std::cout << "\n\n\n====\n\nH=\n" << H3 << "\nU=\n" << U3 << "\n";
    EXPECT_TRUE(isHNF(H3));
    EXPECT_TRUE(H3 == U3 * A);
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
    auto [H3, U3] = NormalForm::hermite(A);
    std::cout << "\n\n\n====\n\nH=" << H3 << "\nU=" << U3 << "\n";
    EXPECT_TRUE(isHNF(H3));
    EXPECT_TRUE(H3 == U3 * A);
  }
  {
    IntMatrix<> A = "[-3 -1 1; 0 0 -2]"_mat;
    std::optional<std::pair<IntMatrix<>, SquareMatrix<int64_t>>> B =
      NormalForm::hermite(A);
    EXPECT_TRUE(B.has_value());
    auto &[H, U] = *B;
    EXPECT_TRUE(isHNF(H));
    EXPECT_TRUE(U * A == H);
    std::cout << "A = \n" << A << "\nH =\n" << H << "\nU =\n" << U << "\n";
  }
  {
    IntMatrix<> A =
      "[3 3 -3 1 0 -1 -2 1 1 2 -1; 3 3 -3 1 1 -3 2 0 3 0 -3; 2 -3 -2 -1 1 -2 3 3 3 3 -3]"_mat;
    auto [H, U] = NormalForm::hermite(A);
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
  ptrdiff_t numIters = 1;
  for (ptrdiff_t numCol = 2; numCol < 11; numCol += 2) {
    IntMatrix<> B(DenseDims<>{{8}, {numCol}});
    ptrdiff_t nullDim = 0;
    IntMatrix<> Z;
    DenseMatrix<int64_t> NS;
    for (ptrdiff_t i = 0; i < numIters; ++i) {
      for (auto &&b : B) {
        b = distrib(gen);
        b = b > 10 ? 0 : b;
      }
      NS = NormalForm::nullSpace(B);
      nullDim += ptrdiff_t(NS.numRow());
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
              << double(nullDim) / double(numIters) << "\n";
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
  IntMatrix<> trueB =
    "[-154140 -128775 -205035 317580 83820 299760; -4910 -21400 -60890 "
    "44820 14480 43390; -1334 -6865 -7666 8098 -538 9191; -6548 -9165 "
    "-24307 26176 4014 23332]"_mat;

  EXPECT_EQ(sA, A);
  EXPECT_EQ(trueB, B);

  IntMatrix<> C = "[1 1 0; 0 1 1; 1 2 1]"_mat;
  IntMatrix<> D = "[1 0 0; 0 1 0; 0 0 1]"_mat;
  NormalForm::simplifySystem(C, D);
  IntMatrix<> trueC = "[1 0 -1; 0 1 1]"_mat;
  IntMatrix<> trueD = "[1 -1 0; 0 1 0]"_mat;
  EXPECT_EQ(trueC, C);
  EXPECT_EQ(trueD, D);
}

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(BareissTests, BasicAssertions) {
  IntMatrix<> A =
    "[-4 3 -2 2 -5; -5 1 -1 2 -5; -1 0 5 -3 2; -4 5 -4 -2 -4]"_mat;
  auto piv = NormalForm::bareiss(A);
  IntMatrix<> B =
    "[-4 3 -2 2 -5; 0 11 -6 2 -5; 0 0 56 -37 32; 0 0 0 -278 136]"_mat;
  EXPECT_EQ(A, B);
  Vector<ptrdiff_t> truePiv{std::array{0, 1, 2, 3}};
  EXPECT_EQ(piv, truePiv);

  IntMatrix<> C = "[-2 -2 -1 -2 -1; 1 1 2 2 -2; -2 2 2 -1 "
                  "-1; 0 0 -2 1 -1; -1 -2 2 1 -1]"_mat;
  IntMatrix<> D = "[-2 -2 -1 -2 -1; 0 -8 -6 -2 0; 0 0 -12 -8 "
                  "20; 0 0 0 -28 52; 0 0 0 0 -142]"_mat;
  auto pivots = NormalForm::bareiss(C);
  EXPECT_EQ(C, D);
  auto truePivots = Vector<ptrdiff_t, 16>{std::array{0, 2, 2, 3, 4}};
  EXPECT_EQ(pivots, truePivots);
}

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(InvTest, BasicAssertions) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(-10, 10);
  const ptrdiff_t numIters = 1000;
  for (ptrdiff_t dim = 1; dim < 5; ++dim) {
    SquareMatrix<int64_t> B(SquareDims<>{{dim}});
    SquareMatrix<int64_t> Db = SquareMatrix<int64_t>::identity(dim);
    for (ptrdiff_t i = 0; i < numIters; ++i) {
      while (true) {
        for (ptrdiff_t n = 0; n < dim * dim; ++n) B.data()[n] = distrib(gen);
        if (NormalForm::rank(B) == dim) break;
      }
      // Da * B^{-1} = Binv0
      // Da = Binv0 * B
      auto [Da, Binv0] = NormalForm::inv(B);
      auto [Binv1, s] = NormalForm::scaledInv(B);
      EXPECT_TRUE(Da.isDiagonal());
      EXPECT_EQ((Binv0 * B), Da);
      Db.diag() << s;
      if (B * Binv1 != Db) {
        std::cout << "\nB = " << B << "\nDa = " << Da << "\nBinv0 = " << Binv0
                  << "\nBinv1 = " << Binv1 << "\ns = " << s << "\n";
      }
      EXPECT_EQ(B * Binv1, Db);
    }
  }
}
