#include <gtest/gtest.h>
#ifndef USE_MODULE
#include "Utilities/Optional.cxx"
#include <cstddef>
#include <limits>
#else
import Optional;
import STL;
#endif

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(Optional, BasicAssertions) {
  utils::Optional<ptrdiff_t> x{3}, y, z{std::numeric_limits<ptrdiff_t>::min()};
  EXPECT_TRUE(x);
  EXPECT_EQ(*x, 3);
  EXPECT_FALSE(y);
  EXPECT_FALSE(z);
  x = 14;
  EXPECT_TRUE(x);
  EXPECT_EQ(*x, 14);
  y = 33;
  EXPECT_TRUE(y);
  EXPECT_EQ(*y, 33);
  y = 0;
  EXPECT_TRUE(y);
  EXPECT_FALSE(*y);
  y = {};
  EXPECT_FALSE(y);

  ptrdiff_t a = 42, b = 11, c = 8;
  utils::Optional<ptrdiff_t *> p{&a}, q;
  static_assert(sizeof(utils::Optional<ptrdiff_t *>) == sizeof(ptrdiff_t *));
  EXPECT_TRUE(p);
  EXPECT_FALSE(q);
  **p += 10;
  EXPECT_EQ(**p, 52);
  q = &b;
  EXPECT_TRUE(q);
  p = &c;
  **p += 18;
  EXPECT_EQ(a, 52);
  EXPECT_EQ(**q, 11);
  EXPECT_EQ(c, 26);
  p = {};
  EXPECT_FALSE(p);
}
