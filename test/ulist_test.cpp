
#include "Alloc/Arena.hpp"
#include "Containers/UnrolledList.hpp"
#include <cstdint>
#include <gtest/gtest.h>

using namespace poly;

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(UListTest, BasicAssertions) {
  alloc::OwningArena<> alloc;
  auto *list = alloc.create<containers::UList<int64_t>>();
  for (int64_t i = 0; i < 100; ++i) {
    list = list->push(&alloc, i);
    int64_t s = i * (i + 1) / 2;
    EXPECT_EQ(list->reduce(0, [](int64_t a, int64_t b) { return a + b; }), s);
    int64_t s2 = i * (i + 1) * (2 * i + 1) / 6;
    EXPECT_EQ(list->transform_reduce(0,
                                     [](int64_t a, int64_t &b) {
                                       b *= 2;
                                       return a + b * b;
                                     }),
              s2 * 4);
    // undo the *2;
    list->forEachRev([](int64_t &a) { a /= 2; });
    const auto *constList = list;
    int64_t c = 0;
    for (auto j : *constList) c += j;
    EXPECT_EQ(c, s);
    c = 0;
    for (auto &&j : *list) c += (j += 3);
    EXPECT_EQ(c - (3 * (i + 1)), s);
    list->forEach([](int64_t &a) { a -= 3; });
  }
}
