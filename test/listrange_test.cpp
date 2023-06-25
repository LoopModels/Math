#include "Utilities/Allocators.hpp"
#include "Utilities/ListRanges.hpp"
#include <bits/ranges_base.h>
#include <concepts>
#include <gtest/gtest.h>
#include <ranges>
#include <vector>
using namespace poly;

template <typename T> class List {
  T data;
  List *next{nullptr};
  List *prev{nullptr};

public:
  List(const List &) = delete;
  constexpr List(T d) : data(d) {}
  constexpr auto getData() -> T & { return data; }
  constexpr auto getNext() -> List * { return next; }
  constexpr auto getPrev() -> List * { return prev; }
  constexpr void setData(T d) { data = d; }
  constexpr auto setNext(List *n) -> List * {
    next = n;
    if (n) n->prev = this;
    return this;
  }
  constexpr auto setPrev(List *p) -> List * {
    prev = p;
    if (p) p->next = this;
    return this;
  }
};

auto f = [](List<List<int> *> *l) -> List<int> * { return l->getData(); };
static_assert(
  std::input_iterator<utils::ListIterator<List<int>, utils::GetNext>>);

static_assert(
  std::forward_iterator<utils::ListIterator<List<int>, utils::GetNext>>);
static_assert(
  std::ranges::input_range<utils::ListRange<List<int>, utils::GetNext>>);
static_assert(
  std::ranges::forward_range<utils::ListRange<List<int>, utils::GetNext>>);

static_assert(
  std::input_iterator<
    utils::NestedListIterator<List<List<int> *>, List<int>, utils::GetNext,
                              utils::GetNext, decltype(f)>>);
static_assert(
  std::forward_iterator<
    utils::NestedListIterator<List<List<int> *>, List<int>, utils::GetNext,
                              utils::GetNext, decltype(f)>>);
static_assert(
  std::ranges::input_range<utils::NestedListRange<
    List<List<int> *>, utils::GetNext, utils::GetNext, decltype(f)>>);
static_assert(
  std::ranges::forward_range<utils::NestedListRange<
    List<List<int> *>, utils::GetNext, utils::GetNext, decltype(f)>>);

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(ListRangeTest, BasicAssertions) {
  utils::OwningArena<> arena;
  List<int> *list = nullptr;
  for (int i = 0; i < 10; ++i) list = arena.create<List<int>>(i)->setNext(list);
  {
    int s = 0;
    for (auto *v : utils::ListRange{list, utils::GetNext{}}) s += v->getData();
    EXPECT_EQ(s, 45);
  }
  List<int> *list100 = nullptr;
  for (int i = 0; i < 1000; i += 100)
    list100 = arena.create<List<int>>(i)->setNext(list100);

  List<int> *list10000 = nullptr;
  for (int i = 0; i < 100000; i += 10000)
    list10000 = arena.create<List<int>>(i)->setNext(list10000);

  auto *listList = arena.create<List<List<int> *>>(list);
  listList = arena.create<List<List<int> *>>(list100)->setNext(listList);
  listList = arena.create<List<List<int> *>>(list10000)->setNext(listList);
  {
    int s = 0;
    for (auto *outer = listList; outer; outer = outer->getNext())
      for (auto *v : utils::ListRange{outer->getData(), utils::GetNext{}})
        s += v->getData();
    EXPECT_EQ(s, 454545);
  }
  {
    int s = 0;
    utils::NestedListRange nlr{listList, utils::GetNext{}, utils::GetNext{}, f};
    static_assert(std::ranges::input_range<decltype(nlr)>);
    static_assert(std::input_iterator<decltype(nlr.begin())>);
    for (auto *v : nlr) s += v->getData();
    EXPECT_EQ(s, 454545);
  }
  {
    std::vector<int> destination;
    std::ranges::transform(
      utils::NestedListRange{listList, utils::GetNext{}, utils::GetNext{}, f},
      std::back_inserter(destination), [](auto *v) { return v->getData(); });
    // std::ostream_iterator<int>{std::cout, ", "},
    EXPECT_EQ(destination.size(), 3 * 10);
    int s = 0;
    for (auto v : destination) s += v;
    EXPECT_EQ(s, 454545);
  }
}
