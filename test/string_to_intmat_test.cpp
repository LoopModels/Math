#include <gtest/gtest.h>
#ifndef USE_MODULE
#include "Math/Array.cxx"
#include "Math/ManagedArray.cxx"
#include "Math/MatrixDimensions.cxx"
#include "Math/StaticArrays.cxx"
#include "Utilities/MatrixStringParse.cxx"
#include <cstddef>
#include <cstdint>
#include <ostream>
#else
import Array;
import ArrayParse;
import ManagedArray;
import MatDim;
import StaticArray;
import STL;
#endif

using namespace math;
using utils::operator""_mat;

auto autoConvert(math::PtrMatrix<int64_t> A) -> int64_t {
  int64_t s = 0;
  for (ptrdiff_t m = 0; m < A.numRow(); ++m)
    for (ptrdiff_t n = 0; n < A.numCol(); ++n) s += A[m, n];
  return s;
}

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(StringParse, BasicAssertions) {
  IntMatrix<> A{"[0 3 -2 1; 3 -1 -2 -2; 2 0 -3 0]"_mat};
  std::cout << "A = \n" << A << "\n";
  EXPECT_EQ((A[0, 0]), 0);
  EXPECT_EQ((A[0, 1]), 3);
  EXPECT_EQ((A[0, 2]), -2);
  EXPECT_EQ((A[0, 3]), 1);
  EXPECT_EQ((A[1, 0]), 3);
  EXPECT_EQ((A[1, 1]), -1);
  EXPECT_EQ((A[1, 2]), -2);
  EXPECT_EQ((A[1, 3]), -2);
  EXPECT_EQ((A[2, 0]), 2);
  EXPECT_EQ((A[2, 1]), 0);
  EXPECT_EQ((A[2, 2]), -3);
  EXPECT_EQ((A[2, 3]), 0);
#ifndef POLYMATHNOEXPLICITSIMDARRAY
  static_assert(std::same_as<math::StaticDims<int64_t, 2, 3, false>,
                             math::StridedDims<2, 3, 4>>);
#else
  static_assert(std::same_as<math::StaticDims<int64_t, 2, 3, false>,
                             math::DenseDims<2, 3>>);
#endif
  EXPECT_EQ(autoConvert("[1 2 3; 4 5 6]"_mat), 21);
}
TEST(StringParse2, BasicAssertions) {
  IntMatrix<> A = "[-1 0 1 0 0; 0 -1 1 0 0; 0 0 -1 1 0; 0 0 -1 0 1]"_mat;
  EXPECT_EQ((A[0, 0]), -1);
}
