#include <gtest/gtest.h>
#ifndef USE_MODULE
#include "Numbers/Int8.cxx"
#include <cstdint>
#include <limits>
#else

import Int8;
import STL;
#endif

using numbers::i8, numbers::u8, numbers::Flag8;

TEST(Int8Test, BasicAssertions) {
  for (uint8_t x = 0; x < std::numeric_limits<uint8_t>::max(); ++x) {
    auto fx = static_cast<Flag8>(x);
    if (x) EXPECT_TRUE(bool(fx));
    else EXPECT_FALSE(bool(fx));
    auto ux = static_cast<u8>(x);
    for (uint8_t y = 0; y < std::numeric_limits<uint8_t>::max(); ++y) {
      auto uy = static_cast<u8>(y);
      EXPECT_EQ(ux <=> uy, x <=> y);
      {
        u8 z = static_cast<u8>(x + y);
        EXPECT_TRUE(z == z);
        EXPECT_FALSE(z != z);
      }
      bool b = (ux > uy) == (x > y);
      EXPECT_TRUE(b);
      EXPECT_TRUE((ux == uy) == (x == y));
      EXPECT_TRUE((ux == y) == (x == y));
      EXPECT_TRUE((x == uy) == (x == y));
      EXPECT_TRUE((ux != uy) == (x != y));
      EXPECT_TRUE((ux != y) == (x != y));
      EXPECT_TRUE((x != uy) == (x != y));
      EXPECT_TRUE((ux > uy) == (x > y));
      EXPECT_TRUE((ux > y) == (x > y));
      EXPECT_TRUE((x > uy) == (x > y));
      EXPECT_TRUE((ux < uy) == (x < y));
      EXPECT_TRUE((ux < y) == (x < y));
      EXPECT_TRUE((x < uy) == (x < y));
      EXPECT_TRUE((ux >= uy) == (x >= y));
      EXPECT_TRUE((ux >= y) == (x >= y));
      EXPECT_TRUE((x >= uy) == (x >= y));
      EXPECT_TRUE((ux <= uy) == (x <= y));
      EXPECT_TRUE((ux <= y) == (x <= y));
      EXPECT_TRUE((x <= uy) == (x <= y));

      EXPECT_EQ(ux > uy, x > y);

      {
        int z = x + y;
        if ((z >= 0) && (z <= std::numeric_limits<uint8_t>::max())) {
          u8 uz = ux;
          uz += uy;
          EXPECT_EQ(uz, z);
          EXPECT_EQ(ux + uy, z);
          EXPECT_EQ(ux + y, z);
          EXPECT_EQ(x + uy, z);
        }
      }
      {
        int z = x * y;
        if ((z >= 0) && (z <= std::numeric_limits<uint8_t>::max())) {
          u8 uz = ux;
          uz *= uy;
          EXPECT_EQ(uz, z);
          EXPECT_EQ(ux * uy, z);
          EXPECT_EQ(ux * y, z);
          EXPECT_EQ(x * uy, z);
        }
      }
      {
        int z = x - y;
        if ((z >= 0) && (z <= std::numeric_limits<uint8_t>::max())) {
          u8 uz = ux;
          uz -= uy;
          EXPECT_EQ(uz, z);
          EXPECT_EQ(ux - uy, z);
          EXPECT_EQ(ux - y, z);
          EXPECT_EQ(x - uy, z);
        }
      }
      if (y) {
        int z = x / y;
        if ((z >= 0) && (z <= std::numeric_limits<uint8_t>::max())) {
          u8 uz = ux;
          uz /= uy;
          EXPECT_EQ(uz, z);
          EXPECT_EQ(ux / uy, z);
          EXPECT_EQ(ux / y, z);
          EXPECT_EQ(x / uy, z);
        }
      }
      auto fy = static_cast<Flag8>(y);
      EXPECT_EQ(fx & fy, static_cast<Flag8>(x & y));
      EXPECT_EQ(fx & y, static_cast<Flag8>(x & y));
      EXPECT_EQ(x & fy, static_cast<Flag8>(x & y));
      EXPECT_EQ(fx | fy, static_cast<Flag8>(x | y));
      EXPECT_EQ(fx | y, static_cast<Flag8>(x | y));
      EXPECT_EQ(x | fy, static_cast<Flag8>(x | y));
      EXPECT_EQ(fx ^ fy, static_cast<Flag8>(x ^ y));
      EXPECT_EQ(fx ^ y, static_cast<Flag8>(x ^ y));
      EXPECT_EQ(x ^ fy, static_cast<Flag8>(x ^ y));

      EXPECT_EQ(fx & fy, x & y);
      EXPECT_EQ(fx & y, x & y);
      EXPECT_EQ(x & fy, x & y);
      EXPECT_EQ(fx | fy, x | y);
      EXPECT_EQ(fx | y, x | y);
      EXPECT_EQ(x | fy, x | y);
      EXPECT_EQ(fx ^ fy, x ^ y);
      EXPECT_EQ(fx ^ y, x ^ y);
      EXPECT_EQ(x ^ fy, x ^ y);

      EXPECT_EQ(x & y, fx & fy);
      EXPECT_EQ(x & y, fx & y);
      EXPECT_EQ(x & y, x & fy);
      EXPECT_EQ(x | y, fx | fy);
      EXPECT_EQ(x | y, fx | y);
      EXPECT_EQ(x | y, x | fy);
      EXPECT_EQ(x ^ y, fx ^ fy);
      EXPECT_EQ(x ^ y, fx ^ y);
      EXPECT_EQ(x ^ y, x ^ fy);
    }
  }
  for (int8_t x = std::numeric_limits<int8_t>::min();
       x < std::numeric_limits<int8_t>::max(); ++x) {
    for (int8_t y = std::numeric_limits<int8_t>::min();
         y < std::numeric_limits<int8_t>::max(); ++y) {
      i8 ix = static_cast<i8>(x);
      i8 iy = static_cast<i8>(y);

      {
        int z = x + y;
        if ((z >= std::numeric_limits<int8_t>::min()) &&
            (z <= std::numeric_limits<int8_t>::max())) {
          i8 iz = ix;
          iz += iy;
          EXPECT_EQ(iz, z);
          EXPECT_EQ(ix + iy, z);
          EXPECT_EQ(ix + y, z);
          EXPECT_EQ(x + iy, z);
        }
      }
      {
        int z = x * y;
        if ((z >= std::numeric_limits<int8_t>::min()) &&
            (z <= std::numeric_limits<int8_t>::max())) {
          i8 iz = ix;
          iz *= iy;
          EXPECT_EQ(iz, z);
          EXPECT_EQ(ix * iy, z);
          EXPECT_EQ(ix * y, z);
          EXPECT_EQ(x * iy, z);
        }
      }
      {
        int z = x - y;
        if ((z >= std::numeric_limits<int8_t>::min()) &&
            (z <= std::numeric_limits<int8_t>::max())) {
          i8 iz = ix;
          iz -= iy;
          EXPECT_EQ(iz, z);
          EXPECT_EQ(ix - iy, z);
          EXPECT_EQ(ix - y, z);
          EXPECT_EQ(x - iy, z);
        }
      }
      if (y) {
        int z = x / y;
        if ((z >= std::numeric_limits<int8_t>::min()) &&
            (z <= std::numeric_limits<int8_t>::max())) {
          i8 iz = ix;
          iz /= iy;
          EXPECT_EQ(iz, z);
          EXPECT_EQ(ix / iy, z);
          EXPECT_EQ(ix / y, z);
          EXPECT_EQ(x / iy, z);
        }
      }
    }
  }
}
