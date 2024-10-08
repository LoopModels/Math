#include <gtest/gtest.h>

#ifndef USE_MODULE
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <print>

#include "Containers/BitSets.cxx"
#include "Math/ManagedArray.cxx"
#else

import BitSet;
import ManagedArray;
import STL;
#endif

using containers::BitSet, math::Vector;

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(BitSetTest, BasicAssertions) {
  BitSet bs(1000);
  bs[4] = true;
  bs[10] = true;
  bs[200] = true;
  bs[117] = true;
  bs[87] = true;
  bs[991] = true;
  bs[0] = true;
  std::cout << bs << "\n";
  // EXPECT_EQ(std::ranges::begin(bs), bs.begin());
  // EXPECT_EQ(std::ranges::end(bs), bs.end());
  Vector<size_t> bsc{std::array{0, 4, 10, 87, 117, 200, 991}};
  ptrdiff_t j = 0;
  for (auto J = bs.begin(); J != decltype(bs)::end(); ++J) {
    EXPECT_EQ(*J, bsc[j++]);
    EXPECT_TRUE(bs[*J]);
    std::println("We get: {}", *J);
  }
  j = 0;
  for (auto i : bs) {
    EXPECT_EQ(i, bsc[j++]);
    EXPECT_TRUE(bs[i]);
    std::println("We get: {}", i);
  }
  EXPECT_EQ(j, bsc.size());
  EXPECT_EQ(j, bs.size());
  BitSet empty;
  ptrdiff_t c = 0, d = 0;
  for (auto b : empty) {
    ++c;
    d += b;
  }
  EXPECT_FALSE(c);
  EXPECT_FALSE(d);
}
// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(BitSetInsert, BasicAssertions) {
  BitSet<std::array<uint64_t, 2>> bs;
  bs.insert(1);
  bs.insert(5);
  bs.insert(6);
  bs.insert(8);
  EXPECT_EQ(bs.data_[0], 354);
  EXPECT_EQ(bs.data_[1], 0);
  bs.insert(5);
  EXPECT_EQ(bs.data_[0], 354);
}
// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(DynSizeBitSetTest, BasicAssertions) {
  BitSet bs, bsd{BitSet<>::dense(11)};
  EXPECT_EQ(bs.data_.size(), 0);
  bs[4] = true;
  bs[10] = true;
  EXPECT_EQ(bs.data_.size(), 1);
  EXPECT_EQ(bs.data_.front(), 1040);
  for (ptrdiff_t i = 0; i < 11; ++i)
    if (!bs.contains(i)) {EXPECT_TRUE(bsd.remove(i));}
  EXPECT_EQ(bs, bsd);
  Vector<size_t> sv;
  for (auto i : bs) sv.push_back(i);
  EXPECT_EQ(sv.size(), 2);
  EXPECT_EQ(sv[0], 4);
  EXPECT_EQ(sv[1], 10);
}
// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(FixedSizeBitSetTest, BasicAssertions) {
  BitSet<std::array<uint64_t, 2>> bs;
  bs[4] = true;
  bs[10] = true;
  EXPECT_EQ(bs.data_[0], 1040);
  EXPECT_EQ(bs.data_[1], 0);
  Vector<size_t> sv;
  for (auto i : bs) sv.push_back(i);
  EXPECT_EQ(sv.size(), 2);
  EXPECT_EQ(sv[0], 4);
  EXPECT_EQ(sv[1], 10);
}
// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(FixedSizeSmallBitSetTest, BasicAssertions) {
  using SB = BitSet<std::array<uint16_t, 1>>;
  static_assert(sizeof(SB) == 2);
  SB bs;
  bs[4] = true;
  bs[10] = true;
  bs[7] = true;
  bs.insert(5);
  EXPECT_EQ(bs.data_[0], 1200);
  Vector<size_t> sv;
  for (auto i : bs) sv.push_back(i);
  EXPECT_EQ(sv.size(), 4);
  EXPECT_EQ(sv[0], 4);
  EXPECT_EQ(sv[1], 5);
  EXPECT_EQ(sv[2], 7);
  EXPECT_EQ(sv[3], 10);
  EXPECT_EQ(SB::fromMask(1200).data_[0], 1200);
}
