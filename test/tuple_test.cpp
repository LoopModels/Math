
#include "Containers/Tuple.hpp"
#include <concepts>
#include <cstdint>
#include <gtest/gtest.h>

using poly::containers::Tuple, poly::containers::tie;

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(TupleTest, BasicAssertions) {
  double x = 2.3;
  double y = 4.5;
  long z = -5;
  ulong w = 15;
  Tuple t{w, x, y, z};
  {
    auto [a, b, c, d] = t;
    static_assert(std::same_as<decltype(a), ulong>);
    static_assert(std::same_as<decltype(x), double>);
    static_assert(std::same_as<decltype(y), double>);
    static_assert(std::same_as<decltype(z), long>);
    EXPECT_EQ(a, w);
    EXPECT_EQ(b, x);
    EXPECT_EQ(c, y);
    EXPECT_EQ(d, z);
  }
  Tuple t1 = t.map([](auto arg) { return 3 * arg; });
  {
    auto [a, b, c, d] = t1;
    static_assert(std::same_as<decltype(a), ulong>);
    static_assert(std::same_as<decltype(x), double>);
    static_assert(std::same_as<decltype(y), double>);
    static_assert(std::same_as<decltype(z), long>);
    EXPECT_EQ(a, 3 * w);
    EXPECT_EQ(b, 3 * x);
    EXPECT_EQ(c, 3 * y);
    EXPECT_EQ(d, 3 * z);
  }
  t1.apply([](auto &arg) { arg *= 2; });
  {
    auto [a, b, c, d] = t1;
    static_assert(std::same_as<decltype(a), ulong>);
    static_assert(std::same_as<decltype(x), double>);
    static_assert(std::same_as<decltype(y), double>);
    static_assert(std::same_as<decltype(z), long>);
    EXPECT_EQ(a, 6 * w);
    EXPECT_EQ(b, 6 * x);
    EXPECT_EQ(c, 6 * y);
    EXPECT_EQ(d, 6 * z);
    tie(w, x, y, z) = t1;
    EXPECT_EQ(a, w);
    EXPECT_EQ(b, x);
    EXPECT_EQ(c, y);
    EXPECT_EQ(d, z);
  }
}
