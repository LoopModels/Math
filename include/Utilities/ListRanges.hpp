#pragma once

#include <cstddef>
#include <ranges>
#include <type_traits>
namespace poly::utils {

class GetNext {
public:
  template <typename T> constexpr auto operator()(T *state) const noexcept {
    return state->getNext();
  }
};
class GetPrev {
public:
  template <typename T> constexpr auto operator()(T *state) const noexcept {
    return state->getPrev();
  }
};
class GetChild {
public:
  template <typename T> constexpr auto operator()(T *state) const noexcept {
    return state->getChild();
  }
};
class GetParent {
public:
  template <typename T> constexpr auto operator()(T *state) const noexcept {
    return state->getParent();
  }
};
class Identity {
public:
  template <typename T> constexpr auto operator()(T *state) const noexcept {
    return state;
  }
};

class End {
public:
  template <typename T> constexpr operator T *() const noexcept {
    return nullptr;
  }
};

/// Safe for removing current iter from `list` while iterating.
template <typename T, class Op, class Proj> class ListIterator {
  T *state_;
  T *next_{nullptr};
  [[no_unique_address]] Op op_{};
  [[no_unique_address]] Proj p_{};

public:
  using value_type = decltype(p_(state_));
  constexpr auto operator*() const noexcept -> value_type { return p_(state_); }
  // constexpr auto operator->() const noexcept -> T * { return state_; }
  constexpr auto getState() const noexcept -> T * { return state_; }
  constexpr auto operator++() noexcept -> ListIterator & {
    state_ = next_;
    if (next_) next_ = op_(next_);
    return *this;
  }
  constexpr auto operator++(int) noexcept -> ListIterator {
    ListIterator tmp{*this};
    state_ = next_;
    if (next_) next_ = op_(next_);
    return tmp;
  }
  constexpr auto operator-(ListIterator const &other) const noexcept
    -> ptrdiff_t {
    ptrdiff_t count = 0;
    for (auto iter = other; iter != *this; ++iter) ++count;
    return count;
  }
  constexpr auto operator==(ListIterator const &other) const noexcept -> bool {
    return state_ == other.state_;
  }
  constexpr auto operator==(End) const noexcept -> bool {
    return state_ == nullptr;
  }
  constexpr ListIterator(T *state) noexcept
    : state_{state}, next_{state ? Op{}(state) : nullptr} {}
  constexpr ListIterator(T *state, Op op, Proj p) noexcept
    : state_{state}, next_{state ? op(state) : nullptr}, op_{op}, p_{p} {}
  constexpr ListIterator() noexcept = default;
  constexpr ListIterator(const ListIterator &) noexcept = default;
  constexpr ListIterator(ListIterator &&) noexcept = default;
  constexpr auto operator=(const ListIterator &) noexcept
    -> ListIterator & = default;
  constexpr auto operator=(ListIterator &&) noexcept
    -> ListIterator & = default;
};
template <typename T, class Op, class Proj>
ListIterator(T *, Op, Proj) -> ListIterator<T, Op, Proj>;
template <typename T, class Op>
ListIterator(T *, Op) -> ListIterator<T, Op, Identity>;

template <typename T, class Op, class Proj>
class ListRange
  : public std::ranges::view_interface<ListIterator<T, Op, Proj>> {
  T *begin_;
  [[no_unique_address]] Op next_{};
  [[no_unique_address]] Proj projection_{};

public:
  constexpr auto begin() noexcept -> ListIterator<T, Op, Proj> {
    return {begin_, next_, projection_};
  }
  constexpr auto begin() const noexcept -> ListIterator<const T, Op, Proj> {
    return {begin_, next_, projection_};
  }
  static constexpr auto end() noexcept -> End { return {}; }
  constexpr ListRange(T *begin, Op next, Proj projection) noexcept
    : begin_{begin}, next_{next}, projection_{projection} {}
  constexpr ListRange(T *begin, Op next) noexcept
    : begin_{begin}, next_{next} {}
  constexpr ListRange(T *begin) noexcept : begin_{begin} {}
  constexpr ListRange() noexcept = default;
  constexpr ListRange(const ListRange &) noexcept = default;
  constexpr ListRange(ListRange &&) noexcept = default;
  constexpr auto operator=(const ListRange &) noexcept -> ListRange & = default;
  constexpr auto operator=(ListRange &&) noexcept -> ListRange & = default;
};
template <typename T, class Op, class Proj>
ListRange(T *, Op, Proj) -> ListRange<T, Op, Proj>;
template <typename T> class NotNull;
template <typename T, class Op, class Proj>
ListRange(NotNull<T>, Op, Proj) -> ListRange<T, Op, Proj>;

template <typename T, class Op>
ListRange(T *, Op) -> ListRange<T, Op, Identity>;
template <typename T> class NotNull;
template <typename T, class Op>
ListRange(NotNull<T>, Op) -> ListRange<T, Op, Identity>;

template <std::forward_iterator O, std::forward_iterator I, class P, class J,
          class F>
class NestedIterator {
  [[no_unique_address]] O outer;
  [[no_unique_address]] I inner;
  [[no_unique_address]] P outerend;
  [[no_unique_address]] J innerend;
  [[no_unique_address]] F innerfun;

public:
  using value_type = decltype(*inner);

  constexpr auto operator==(NestedIterator const &other) const noexcept
    -> bool {
    return outer == other.outer && inner == other.inner;
  }
  constexpr auto operator==(End) const noexcept -> bool {
    return outer == outerend;
  }
  constexpr auto operator++() noexcept -> NestedIterator & {
    if (++inner != innerend) return *this;
    if (++outer != outerend) {
      auto iter = innerfun(*outer);
      inner = iter.begin();
      innerend = iter.end();
    }
    return *this;
  }
  constexpr auto operator++(int) noexcept -> NestedIterator {
    NestedIterator tmp{*this};
    ++*this;
    return tmp;
  }
  constexpr auto operator*() const noexcept -> value_type { return *inner; }
  constexpr auto operator->() const noexcept -> I { return inner; }
  constexpr auto operator-(NestedIterator const &other) const noexcept
    -> ptrdiff_t {
    ptrdiff_t count = 0;
    for (auto iter = other; iter != *this; ++iter) ++count;
    return count;
  }
  constexpr NestedIterator() noexcept = default;
  constexpr NestedIterator(const NestedIterator &) noexcept = default;
  constexpr NestedIterator(O o, I i, P p, J j, F f) noexcept
    : outer{o}, inner{i}, outerend{p}, innerend{j}, innerfun{f} {}
  constexpr NestedIterator(NestedIterator &&) noexcept = default;
  constexpr auto operator=(const NestedIterator &) noexcept
    -> NestedIterator & = default;
  constexpr auto operator=(NestedIterator &&) noexcept
    -> NestedIterator & = default;
};

/// NestedList
/// A range that lets us iterate over a graph via nesting ranges.
/// This is called a "nested list" because the output of each
/// range level is the head of the next iteration, i.e. it is useful
/// for nested list-like data structures, rather than (for example)
/// cartesian product iterators.
template <std::ranges::forward_range O, class F>
class NestedList : public std::ranges::view_interface<NestedList<O, F>> {
  static_assert(std::ranges::view<O>);
  static_assert(std::is_trivially_destructible_v<O>);
  [[no_unique_address]] O outer;
  [[no_unique_address]] F inner;

public:
  using InnerType = decltype(inner(*(outer.begin())));
  static_assert(std::ranges::view<InnerType>);
  static_assert(std::is_trivially_destructible_v<InnerType>);

private:
  using InnerBegin = decltype(std::declval<InnerType>().begin());
  using InnerEnd = decltype(std::declval<InnerType>().end());
  using IteratorType = NestedIterator<decltype(outer.begin()), InnerBegin,
                                      decltype(outer.end()), InnerEnd, F>;

public:
  constexpr auto begin() noexcept -> IteratorType {
    auto out = outer.begin();
    if (out == outer.end())
      return IteratorType{out, InnerBegin{}, outer.end(), InnerEnd{}, inner};
    InnerType inn{inner(*out)};
    return IteratorType{out, inn.begin(), outer.end(), inn.end(), inner};
  }
  static constexpr auto end() noexcept -> End { return {}; }
  constexpr NestedList(O out, F inn) noexcept
    : outer{std::move(out)}, inner{std::move(inn)} {}
  constexpr NestedList(const NestedList &) noexcept = default;
  constexpr NestedList(NestedList &&) noexcept = default;
  constexpr auto operator=(const NestedList &) noexcept
    -> NestedList & = default;
  constexpr auto operator=(NestedList &&) noexcept -> NestedList & = default;
};
template <std::ranges::forward_range O, class F>
NestedList(O, F) -> NestedList<O, F>;

// constexpr auto nest(auto &&outer, auto &&f) noexcept {
//   return NestedList{std::forward<decltype(outer)>(outer),
//                     std::forward<decltype(f)>(f)};
// }

// static_assert(std::ranges::range<ListRange

} // namespace poly::utils

template <typename T, class Op, class Proj>
inline constexpr bool
  std::ranges::enable_borrowed_range<poly::utils::ListRange<T, Op, Proj>> =
    true;
template <std::ranges::forward_range O, class F>
inline constexpr bool
  std::ranges::enable_borrowed_range<poly::utils::NestedList<O, F>> =
    std::ranges::enable_borrowed_range<O> &&
    std::ranges::enable_borrowed_range<
      typename poly::utils::NestedList<O, F>::InnerType>;
