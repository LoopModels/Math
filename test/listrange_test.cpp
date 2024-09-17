#include <gtest/gtest.h>

#ifndef USE_MODULE
#include <algorithm>
#include <concepts>
#include <ranges>

#include "Alloc/Arena.cxx"
#include "Math/ManagedArray.cxx"
#include "Utilities/ListRanges.cxx"
#else

import Arena;
import ListRange;
import ManagedArray;
import STL;
#endif

template <typename T> class List {
  T data_;
  List *next_{nullptr};
  List *prev_{nullptr};

public:
  List(const List &) = delete;
  constexpr List(T d) : data_(d) {}
  constexpr auto getData() -> T & { return data_; }
  [[nodiscard]] constexpr auto getNext() const -> List * { return next_; }
  constexpr auto getPrev() -> List * { return prev_; }
  constexpr void setData(T d) { data_ = d; }
  constexpr auto setNext(List *n) -> List * {
    next_ = n;
    if (n) n->prev_ = this;
    return this;
  }
  constexpr auto setPrev(List *p) -> List * {
    prev_ = p;
    if (p) p->next_ = this;
    return this;
  }
};

namespace {
auto g = [](List<List<int> *> *l) {
  return utils::ListRange{l->getData(), utils::GetNext{}};
};
} // namespace
using LI = utils::ListIterator<List<int>, utils::GetNext, utils::Identity>;
using LR = utils::ListRange<List<int>, utils::GetNext, utils::Identity>;
using LIO = utils::ListIterator<List<List<int> *>, utils::GetNext, decltype(g)>;
static_assert(std::forward_iterator<LIO>);

static_assert(std::input_iterator<LI>);
static_assert(std::forward_iterator<LI>);
static_assert(std::ranges::input_range<LR>);
static_assert(std::ranges::forward_range<LR>);
static_assert(std::ranges::range<LR>);

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(ListRangeTest, BasicAssertions) {
  alloc::OwningArena<> arena;
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

  auto *list_list = arena.create<List<List<int> *>>(list);
  list_list = arena.create<List<List<int> *>>(list100)->setNext(list_list);
  list_list = arena.create<List<List<int> *>>(list10000)->setNext(list_list);

  {
    int s = 0;
    for (auto *outer = list_list; outer; outer = outer->getNext())
      for (auto *v : utils::ListRange{outer->getData(), utils::GetNext{}})
        s += v->getData();
    EXPECT_EQ(s, 454545);
  }
  {
    int s = 0;
    for (auto *outer = list_list; outer; outer = outer->getNext())
      for (auto v : utils::ListRange{outer->getData(), utils::GetNext{},
                                     [](auto *v) { return 2 * v->getData(); }})
        s += v;
    EXPECT_EQ(s, 909090);
  }
  {
    int s = 0;
    List<List<int> *> *zll =
      arena.create<List<List<int> *>>(nullptr)->setNext(list_list);
    utils::ListRange outer{zll, utils::GetNext{}};
    utils::NestedList nlr{outer, g};
    static_assert(std::input_iterator<decltype(nlr.begin())>);
    static_assert(std::forward_iterator<decltype(nlr.begin())>);
    static_assert(std::incrementable<decltype(nlr.begin())>);
    static_assert(std::ranges::input_range<decltype(nlr)>);
    static_assert(std::ranges::enable_borrowed_range<decltype(outer)>);
    static_assert(
      std::ranges::enable_borrowed_range<typename decltype(nlr)::InnerType>);
    static_assert(std::ranges::enable_borrowed_range<decltype(nlr)>);
    static_assert(std::ranges::borrowed_range<decltype(nlr)>);
    static_assert(std::ranges::range<decltype(nlr)>);
    for (auto *v : nlr) s += v->getData();
    EXPECT_EQ(s, 454545);
  }
  {
    int s = 0;
    auto tmp =
      std::ranges::owning_view{utils::ListRange{list_list, utils::GetNext{}}};
    static_assert(std::ranges::input_range<decltype(tmp)>);
    static_assert(std::is_trivially_copyable_v<decltype(tmp)>);
    static_assert(std::is_trivially_destructible_v<decltype(tmp)>);
    static_assert(std::ranges::enable_borrowed_range<decltype(tmp)>);
    auto pred = [](List<List<int> *> *v) -> bool {
      return v->getData()->getData() != 900;
    };
    static_assert(std::is_trivially_copyable_v<decltype(pred)>);
    static_assert(std::ranges::enable_borrowed_range<decltype(tmp)>);
    static_assert(std::ranges::view<
                  std::ranges::filter_view<decltype(tmp), decltype(pred)>>);
    // static_assert(std::is_trivially_copyable_v<
    //               std::ranges::filter_view<decltype(tmp), decltype(pred)>>);
    auto outer =
      utils::ListRange{list_list, utils::GetNext{}} | std::views::filter(pred);
    static_assert(std::ranges::input_range<decltype(outer)>);
    static_assert(std::is_trivially_destructible_v<decltype(outer)>);
    static_assert(std::ranges::view<decltype(outer)>);
    utils::NestedList nlr{outer, g};
    static_assert(std::input_iterator<decltype(nlr.begin())>);
    static_assert(std::ranges::input_range<decltype(nlr)>);
    static_assert(std::ranges::view<decltype(outer)>);
    static_assert(
      std::ranges::enable_borrowed_range<typename decltype(nlr)::InnerType>);
    static_assert(std::ranges::view<decltype(nlr)>);
    for (auto *v : nlr) s += v->getData();
    EXPECT_EQ(s, 450045);
  }
  {
    math::Vector<int> destination;
    utils::NestedList nlr{utils::ListRange{list_list, utils::GetNext{}}, g};
    static_assert(std::input_iterator<decltype(nlr.begin())>);
    static_assert(std::ranges::input_range<decltype(nlr)>);
    std::ranges::transform(nlr, std::back_inserter(destination),
                           [](auto *v) { return v->getData(); });
    // std::ostream_iterator<int>{std::cout, ", "},
    EXPECT_EQ(destination.size(), 3 * 10);
    int s = 0;
    for (auto v : destination) s += v;
    EXPECT_EQ(s, 454545);
  }
}
