#include <gtest/gtest.h>

#ifndef USE_MODULE
#include <cmath>
#include <cstdint>
#include <random>

#include "Bit/Float.cxx"
#else

import BitHack;
import STL;
#endif

TEST(BitTest, BasicAssertions) {
  for (int i = 0; i < 63; ++i) {
    auto e2 = double(uint64_t(1) << i);
    EXPECT_EQ(bit::exp2unchecked(i), e2);
    EXPECT_EQ(bit::exp2unchecked(-i), 1.0 / e2);
  }
  std::random_device rd;
  std::mt19937 gen(rd());
  std::exponential_distribution<double> dist(15.6);
  for (int i = 0; i < 10000; ++i) {
    double x = dist(gen);
    EXPECT_EQ(bit::next_pow2(x), std::exp2(std::ceil(std::log2(x))));
  }
}
