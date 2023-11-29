
#include "Math/GreatestCommonDivisor.hpp"
#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <random>
// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(SIMDGCDTest, BasicAssertions) {
  constexpr ptrdiff_t W = poly::simd::Width<int64_t>;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(std::numeric_limits<int>::min(),
                                          std::numeric_limits<int>::max());
  for (ptrdiff_t i = 0; i < 20000; ++i) {
    std::array<int64_t, size_t(W)> z;
    poly::simd::Vec<W, int64_t> a, b;
    int64_t rg = 0;
    for (ptrdiff_t j = 0; j < W; ++j) {
      int64_t w = distrib(gen), x = distrib(gen), y = std::gcd(w, x);
      EXPECT_EQ(y, poly::math::gcd(w, x));
      z[j] = y;
      a[j] = w;
      b[j] = x;
      rg = std::gcd(rg, y);
    }
    poly::simd::Vec<W, int64_t> g = poly::math::gcd<W>(a, b);
    for (ptrdiff_t j = 0; j < W; ++j) EXPECT_EQ(g[j], z[j]);
    EXPECT_EQ(rg, poly::math::gcdreduce<W>(g));
  }
}

