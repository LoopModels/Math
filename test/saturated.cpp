#include <gtest/gtest.h>
#ifndef USE_MODULE
#include "Math/Saturated.cxx"
#include <algorithm>
#include <limits>
#include <random>
#else
import Saturated;
import STL;
#endif

TEST(SaturatedArithmetic, BasicAssertions) {

  constexpr int imax = std::numeric_limits<int>::max(),
                imin = std::numeric_limits<int>::min();
  EXPECT_EQ(math::add_sat(imax - 10, imax / 2), imax);
  EXPECT_EQ(math::add_sat(imin + 10, imin / 2), imin);
  EXPECT_EQ(math::sub_sat(imax - 10, imin / 2), imax);
  EXPECT_EQ(math::sub_sat(imin + 10, imax / 2), imin);
  EXPECT_EQ(math::mul_sat(imax - 10, imax / 2), imax);
  EXPECT_EQ(math::mul_sat(imin + 10, imax / 2), imin);

  constexpr auto iminlong = static_cast<long long>(imin),
                 imaxlong = static_cast<long long>(imax);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> idistrib(imin, imax);
  for (int i = 0; i < 1000; ++i) {
    int a = idistrib(gen), b = idistrib(gen);
    long long a64 = static_cast<long long>(a) + b,
              s64 = static_cast<long long>(a) - b,
              m64 = static_cast<long long>(a) * b;
    EXPECT_EQ(math::add_sat(a, b), std::clamp(a64, iminlong, imaxlong));
    EXPECT_EQ(math::sub_sat(a, b), std::clamp(s64, iminlong, imaxlong));
    EXPECT_EQ(math::mul_sat(a, b), std::clamp(m64, iminlong, imaxlong));
  }

  constexpr unsigned int umax = std::numeric_limits<unsigned int>::max();
  EXPECT_EQ(math::add_sat(umax - 10, umax / 2), umax);
  EXPECT_EQ(math::sub_sat(umax / 2, umax - 1), 0);
  EXPECT_EQ(math::mul_sat(umax - 10, umax / 2), umax);

  constexpr auto umaxlong = static_cast<long long>(umax);
  std::uniform_int_distribution<unsigned> udistrib(0, umax);
  for (int i = 0; i < 1000; ++i) {
    unsigned a = udistrib(gen), b = udistrib(gen);
    long long a64 = static_cast<long long>(a) + b,
              s64 = static_cast<long long>(a) - b;
    unsigned long long m64 = static_cast<unsigned long long>(a) * b;
    EXPECT_EQ(math::add_sat(a, b), std::clamp(a64, (long long)0, umaxlong));
    EXPECT_EQ(math::sub_sat(a, b), std::clamp(s64, (long long)0, umaxlong));
    EXPECT_EQ(math::mul_sat(a, b), std::clamp(m64, (unsigned long long)0,
                                              (unsigned long long)umaxlong));
  }
}
