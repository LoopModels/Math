#include "Math/Array.hpp"
#include "Math/StaticArrays.hpp"
#include <gtest/gtest.h>
#include <random>
#include <utility>

template <typename T>
constexpr void swap(poly::math::Reference<T> x, poly::math::Reference<T> y) {
  typename poly::math::Reference<T>::U temp = x;
  x = y;
  y = temp;
}

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(TypeCompressionTest, BasicAssertions) {
  using M = poly::math::DenseMatrix<poly::math::SVector<double, 7>>;
  M A(poly::math::DenseDims{3, 3});

  M B(poly::math::DenseDims{3, 3});

  std::mt19937_64 mt(1); // NOLINT(cert-msc51-cpp) // NOLINT(cert-msc32-c)
  for (ptrdiff_t i = 0; i < 3; ++i) {
    for (ptrdiff_t j = 0; j < 3; ++j) {
      poly::math::SVector<double, 7> a;
      for (ptrdiff_t k = 0; k < 7; ++k) {
        a[k] = std::uniform_real_distribution<double>(-2, 2)(mt);
        // b[k] = std::uniform_real_distribution<double>(-2, 2)(mt);
        // A(i, j)[k] = std::uniform_real_distribution<double>(-2, 2)(mt);
        B(i, j)[k] = std::uniform_real_distribution<double>(-2, 2)(mt);
      }
      A(i, j) = a;
      // B(i, j) = b;
    }
  }
  M C = A;
  EXPECT_TRUE(A == C);
  EXPECT_FALSE(B == C);
  for (ptrdiff_t i = 0; i < 3; ++i)
    for (ptrdiff_t j = 0; j < 3; ++j) swap(A(i, j), B(i, j));
  EXPECT_FALSE(A == C);
  EXPECT_TRUE(B == C);
}
