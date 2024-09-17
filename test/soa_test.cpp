#include <gtest/gtest.h>
#ifndef USE_MODULE
#include "Containers/Pair.cxx"
#include "Containers/Tuple.cxx"
#include "Math/Array.cxx"
#include "Math/AxisTypes.cxx"
#include "Math/ManagedArray.cxx"
#include "Math/Ranges.cxx"
#include "Math/SOA.cxx"
#include <concepts>
#include <cstddef>
#include <type_traits>
#else
import Array;
import AxisTypes;
import ManagedArray;
import Pair;
import Range;
import SOA;
import STL;
import Tuple;
#endif

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(SOATest, BasicAssertions) {
  containers::Tuple x{3, 2.0, 5.0F};
  // static_assert(math::CumSizeOf_v<0, decltype(x)> == 0);
  // static_assert(math::CumSizeOf_v<1, decltype(x)> == 4);
  // static_assert(math::CumSizeOf_v<2, decltype(x)> == 12);
  // static_assert(math::CumSizeOf_v<3, decltype(x)> == 16);
  // math::ManagedSOA soa{std::type_identity<decltype(x)>{}, 5};
  using T = decltype(x);
  static_assert(
    std::is_trivially_default_constructible_v<containers::Tuple<int, double>>);
  static_assert(std::is_trivially_default_constructible_v<T>);
  static_assert(std::is_trivially_destructible_v<T>);
  math::ManagedSOA soa(std::type_identity<decltype(x)>{}, math::length(5z));
  EXPECT_EQ(soa.capacity_.capacity_, 8);
  soa[0] = x;
  soa[1] = {5, 2.25, 5.5F};
  soa.template get<0>(2) = 7;
  soa.template get<1>(2) = 2.5;
  soa.template get<2>(2) = 6.0F;
  soa.template get<0>()[3] = 9;
  soa.template get<1>()[3] = 2.75;
  soa.template get<2>()[3] = 6.5F;
  soa[4] = {11, 3.0, 7.0F};
  {
    ptrdiff_t j = 0;
    for (auto [i, d, f] : soa) {
      static_assert(std::same_as<decltype(i), int>);
      static_assert(std::same_as<decltype(d), double>);
      static_assert(std::same_as<decltype(f), float>);
      EXPECT_EQ(i, 3 + 2 * j);
      EXPECT_EQ(d, 2.0 + 0.25 * j);
      EXPECT_EQ(f, 5.0F + 0.5F * j);
      EXPECT_EQ(i, soa.get<0>(j));
      EXPECT_EQ(d, soa.get<1>(j));
      EXPECT_EQ(f, soa.get<2>(j));
      ++j;
    }
  }
  soa.resize(7);
  soa[5] = {13, 3.25, 7.5F};
  soa[6] = {15, 3.5, 8.0F};
  for (ptrdiff_t j = 0; j < 7; ++j) {
    decltype(x) y = *(soa.begin() + j);
    // decltype(x) y = soa[j];
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
    else soa.push_back({i, d, f});
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
TEST(SOAPairTest, BasicAssertions) {
  containers::Pair x{3, 2.0};
  // static_assert(math::CumSizeOf_v<0, decltype(x)> == 0);
  // static_assert(math::CumSizeOf_v<1, decltype(x)> == 4);
  // static_assert(math::CumSizeOf_v<2, decltype(x)> == 12);
  // math::ManagedSOA soa{std::type_identity<decltype(x)>{}, 5};
  math::ManagedSOA<decltype(x)> soa;
  // math::ManagedSOA soa(std::type_identity<decltype(x)>{});
  EXPECT_EQ(soa.capacity_.capacity_, 0);
  soa.push_back(x);
  soa[0] = x;
  EXPECT_EQ(soa.capacity_.capacity_, 8);
  soa.resize(5);
  soa[1] = {5, 2.25};
  soa.template get<0>(2) = 7;
  soa.template get<1>(2) = 2.5;
  soa.template get<0>()[3] = 9;
  soa.template get<1>()[3] = 2.75;
  soa[4] = {11, 3.0};
  for (ptrdiff_t j = 0; j < 5; ++j) {
    decltype(x) y = soa[j];
    auto [i, d] = y;
    static_assert(std::same_as<decltype(i), int>);
    static_assert(std::same_as<decltype(d), double>);
    EXPECT_EQ(i, 3 + 2 * j);
    EXPECT_EQ(d, 2.0 + 0.25 * j);
    EXPECT_EQ(i, soa.get<0>(j));
    EXPECT_EQ(d, soa.get<1>(j));
  }
  soa.resize(7);
  soa[5] = {13, 3.25};
  soa[6] = {15, 3.5};
  for (ptrdiff_t j = 0; j < 7; ++j) {
    decltype(x) y = soa[j];
    auto [i, d] = y;
    static_assert(std::same_as<decltype(i), int>);
    static_assert(std::same_as<decltype(d), double>);
    EXPECT_EQ(i, 3 + 2 * j);
    EXPECT_EQ(d, 2.0 + 0.25 * j);
    EXPECT_EQ(i, soa.get<0>(j));
    EXPECT_EQ(d, soa.get<1>(j));
  }
  for (int j = 7; j < 65; ++j) {
    int i = 3 + 2 * j;
    double d = 2.0 + 0.25 * j;
    if (j & 1) soa.emplace_back(i, d);
    else soa.push_back({i, d});
  }
  for (ptrdiff_t j = 0; j < 65; ++j) {
    decltype(x) y = soa[j];
    auto [i, d] = y;
    static_assert(std::same_as<decltype(i), int>);
    static_assert(std::same_as<decltype(d), double>);
    EXPECT_EQ(i, 3 + 2 * j);
    EXPECT_EQ(d, 2.0 + 0.25 * j);
    EXPECT_EQ(i, soa.get<0>(j));
    EXPECT_EQ(d, soa.get<1>(j));
  }
  EXPECT_EQ(soa.size(), 65);
}
TEST(VecOfSOATest, BasicAssertions) {
  math::Vector<math::ManagedSOA<containers::Tuple<int, double, float>>> vsoa;
  vsoa.emplace_back();
  vsoa.emplace_back();
  vsoa.pop_back();
  vsoa.emplace_back();
  vsoa.emplace_back();
  vsoa.emplace_back();
  EXPECT_EQ(vsoa.size(), 4);
}
