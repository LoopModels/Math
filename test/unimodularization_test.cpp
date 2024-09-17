#include <gtest/gtest.h>
#ifndef USE_MODULE
#include "Math/ManagedArray.cxx"
#include "Math/Unimodularization.cxx"
#include "Utilities/MatrixStringParse.cxx"
#include <ostream>
#else
import ArrayParse;
import ManagedArray;
import STL;
import Unimodularization;
#endif

using namespace math;
using utils::operator""_mat;

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(UnimodularizationTest, BasicAssertions) {
  /*
  IntMatrix<> VE{"[0 1; 1 0; 0 1; 1 0]"_mat};
  std::cout << "VE=" << VE << "\n";
  auto VB = unimodularize(VE);
  ASSERT_TRUE(VB.has_value());
  std::cout << "VB=" << *VB << "\n";

  IntMatrix<> A23{"[9 5; -5 -2; 1 0]"_mat};
  auto B = unimodularize(A23);
  ASSERT_TRUE(B.has_value());
  std::cout << "B=" << *B << "\n";
  // EXPECT_EQ(j, length(bsc));
  // EXPECT_EQ(j, length(bs));
*/
  IntMatrix<> A13{"[6; -5; 15]"_mat};
  std::cout << "A13=" << A13 << "\n";
  EXPECT_EQ((A13[0, 0]), 6);
  EXPECT_EQ((A13[1, 0]), -5);
  EXPECT_EQ((A13[2, 0]), 15);
  auto test6_10_15 = unimodularize(A13); //, 1, 93, 1001);
  EXPECT_TRUE(test6_10_15.has_value());
  A13[0, 0] = 102;
  A13[1, 0] = 190;
  A13[2, 0] = 345;
  auto test102_190_345 = unimodularize(A13); //, 1, 93, 1001);
  EXPECT_TRUE(test102_190_345.has_value());
}
