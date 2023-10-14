#include "Math/Math.hpp"
#include "Math/Unimodularization.hpp"
#include "Utilities/MatrixStringParse.hpp"
#include <gtest/gtest.h>
#include <ostream>

using namespace poly::math;
using poly::utils::operator""_mat;

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(UnimodularizationTest, BasicAssertions) {
  IntMatrix<> VE{"[0 1; 1 0; 0 1; 1 0]"_mat};
  std::cout << "VE=\n" << VE << "\n";
  auto VB = unimodularize(VE);
  EXPECT_TRUE(VB.has_value());
  ASSERT(VB.has_value());
  std::cout << "VB:\n" << *VB << "\n";

  IntMatrix<> A23{"[9 5; -5 -2; 1 0]"_mat};
  auto B = unimodularize(A23);
  EXPECT_TRUE(B.has_value());
  ASSERT(B.has_value());
  std::cout << "B:\n" << *B << "\n";
  // EXPECT_EQ(j, length(bsc));
  // EXPECT_EQ(j, length(bs));

  IntMatrix<> A13{"[6; -5; 15]"_mat};
  auto test6_10_15 = unimodularize(A13); //, 1, 93, 1001);
  EXPECT_TRUE(test6_10_15.has_value());
  A13[0, 0] = 102;
  A13[1, 0] = 190;
  A13[2, 0] = 345;
  auto test102_190_345 = unimodularize(A13); //, 1, 93, 1001);
  EXPECT_TRUE(test102_190_345.has_value());
}
