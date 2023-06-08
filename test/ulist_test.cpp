
#include "Containers/UnrolledList.hpp"
#include "Utilities/Allocators.hpp"
#include <cstdint>
#include <gtest/gtest.h>

using namespace poly;

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(UListTest, BasicAssertions) {
  utils::BumpAlloc<> alloc;
  auto *list = alloc.create<containers::UList<int64_t>>();
  for (int64_t i = 0; i < 100; ++i) {
    list = list->push(alloc, i);
    int64_t s = i * (i + 1) / 2;
    EXPECT_EQ(list->reduce(0, [](int64_t a, int64_t b) { return a + b; }), s);
    int64_t s2 = i * (i + 1) * (2 * i + 1) / 6;
    EXPECT_EQ(list->reduce(0, [](int64_t a, int64_t b) { return a + b * b; }),
              s2);
  }
}
