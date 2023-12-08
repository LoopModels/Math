#include "Containers/Tuple.hpp"
#include "Math/SOA.hpp"
#include <gtest/gtest.h>
// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(SOATest, BasicAssertions) {
  poly::containers::Tuple x{3, 2.0, 5.0F};
  static_assert(poly::math::CumSizeOf_v<0, decltype(x)> == 0);
  static_assert(poly::math::CumSizeOf_v<1, decltype(x)> == 4);
  static_assert(poly::math::CumSizeOf_v<2, decltype(x)> == 12);
  static_assert(poly::math::CumSizeOf_v<3, decltype(x)> == 16);
  // poly::math::ManagedSOA soa{std::type_identity<decltype(x)>{}, 5};
  poly::math::ManagedSOA soa(std::type_identity<decltype(x)>{}, ptrdiff_t(5));
  soa[0] = x;
  soa[1] = {5, 2.25, 5.5F};
  soa.template get<0>(2) = 7;
  soa.template get<1>(2) = 2.5;
  soa.template get<2>(2) = 6.0F;
  soa.template get<0>()[3] = 9;
  soa.template get<1>()[3] = 2.75;
  soa.template get<2>()[3] = 6.5F;
  soa[4] = {11, 3.0, 7.0F};
  for (ptrdiff_t j = 0; j < 5; ++j) {
    decltype(x) y = soa[j];
    auto [i, d, f] = y;
    static_assert(std::same_as<decltype(i), int>);
    static_assert(std::same_as<decltype(d), double>);
    static_assert(std::same_as<decltype(f), float>);
    EXPECT_EQ(i, 3 + 2 * j);
    EXPECT_EQ(d, 2.0 + 0.25 * j);
    EXPECT_EQ(f, 5.0F + 0.5F * j);
    EXPECT_EQ(i, soa.get<0>(j));
    EXPECT_EQ(d, soa.get<1>(j));
    EXPECT_EQ(f, soa.get<2>(j));
  }
  soa.resize(7);
  soa[5] = {13, 3.25, 7.5F};
  soa[6] = {15, 3.5, 8.0F};
  for (ptrdiff_t j = 0; j < 7; ++j) {
    decltype(x) y = soa[j];
    auto [i, d, f] = y;
    static_assert(std::same_as<decltype(i), int>);
    static_assert(std::same_as<decltype(d), double>);
    static_assert(std::same_as<decltype(f), float>);
    EXPECT_EQ(i, 3 + 2 * j);
    EXPECT_EQ(d, 2.0 + 0.25 * j);
    EXPECT_EQ(f, 5.0F + 0.5F * j);
    EXPECT_EQ(i, soa.get<0>(j));
    EXPECT_EQ(d, soa.get<1>(j));
    EXPECT_EQ(f, soa.get<2>(j));
  }
  for (int j = 7; j < 65; ++j) {
    int i = 3 + 2 * j;
    double d = 2.0 + 0.25 * j;
    float f = 5.0F + 0.5F * float(j);
    if (j & 1) soa.emplace_back(i, d, f);
    else soa.push_back(poly::containers::Tuple(i, d, f));
  }
  for (ptrdiff_t j = 0; j < 65; ++j) {
    decltype(x) y = soa[j];
    auto [i, d, f] = y;
    static_assert(std::same_as<decltype(i), int>);
    static_assert(std::same_as<decltype(d), double>);
    static_assert(std::same_as<decltype(f), float>);
    EXPECT_EQ(i, 3 + 2 * j);
    EXPECT_EQ(d, 2.0 + 0.25 * j);
    EXPECT_EQ(f, 5.0F + 0.5F * j);
    EXPECT_EQ(i, soa.get<0>(j));
    EXPECT_EQ(d, soa.get<1>(j));
    EXPECT_EQ(f, soa.get<2>(j));
  }
  EXPECT_EQ(soa.size(), 65);
}
