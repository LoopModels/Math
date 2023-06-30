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

template <typename T, class Op, class Proj> class ListIterator {
  T *state_;
  [[no_unique_address]] Op op_{};
  [[no_unique_address]] Proj p_{};

public:
  using value_type = decltype(p_(state_));
  constexpr auto operator*() const noexcept -> value_type { return p_(state_); }
  // constexpr auto operator->() const noexcept -> T * { return state_; }
  constexpr auto getState() const noexcept -> T * { return state_; }
  constexpr auto operator++() noexcept -> ListIterator & {
    state_ = op_(state_);
    return *this;
  }
  constexpr auto operator++(int) noexcept -> ListIterator {
    ListIterator tmp{*this};
    state_ = op_(state_);
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
  constexpr ListIterator(T *state) noexcept : state_{state} {}
  constexpr ListIterator(T *state, Op op, Proj p) noexcept
    : state_{state}, op_{op}, p_{p} {}
  constexpr ListIterator() = default;
};
template <typename T, class Op, class Proj>
ListIterator(T *, Op, Proj) -> ListIterator<T, Op, Proj>;
template <typename T, class Op>
ListIterator(T *, Op) -> ListIterator<T, Op, Identity>;

template <typename T, class Op, class Proj> class ListRange {
  T *begin_;
  [[no_unique_address]] Op op_{};
  [[no_unique_address]] Proj p_{};

public:
  constexpr auto begin() noexcept -> ListIterator<T, Op, Proj> {
    return {begin_, op_, p_};
  }
  constexpr auto begin() const noexcept -> ListIterator<const T, Op, Proj> {
    return {begin_, op_, p_};
  }
  static constexpr auto end() noexcept -> End { return {}; }
  constexpr ListRange(T *begin, Op op, Proj p) noexcept
    : begin_{begin}, op_{op}, p_{p} {}
  constexpr ListRange(T *begin, Op op) noexcept : begin_{begin}, op_{op} {}
  constexpr ListRange(T *begin) noexcept : begin_{begin} {}
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
// actually, implement via composition!
// Implementation detail: the current inner state corresponds to the current
// outer state. When we run out of inner states, we increment outer and set
// inner.
template <class O, class I, class OpOuter, class OpInner, class NewInner,
          class Proj>
class NestedListIterator {
  ListIterator<O, OpOuter, NewInner> outer_state_;
  ListIterator<I, OpInner, Proj> inner_state_;

public:
  using value_type = decltype(*inner_state_);
  constexpr auto operator*() const noexcept -> value_type {
    return *inner_state_;
  }
  // constexpr auto operator->() const noexcept -> I * {
  //   return inner_state_.getState();
  // }
  constexpr auto operator++() noexcept -> NestedListIterator & {
    ++inner_state_;
    if (inner_state_ == End{}) {
      ++outer_state_;
      if (outer_state_ != End{})
        inner_state_ = {*outer_state_, OpInner{}, Proj{}};
    }
    return *this;
  }
  constexpr auto operator++(int) noexcept -> NestedListIterator {
    NestedListIterator tmp{*this};
    ++*this;
    return tmp;
  }
  constexpr auto operator-(NestedListIterator const &other) const noexcept
    -> ptrdiff_t {
    ptrdiff_t count = 0;
    for (auto iter = other; iter != *this; ++iter) ++count;
    return count;
  }
  constexpr auto operator==(NestedListIterator const &other) const noexcept
    -> bool {
    return outer_state_ == other.outer_state_ &&
           inner_state_ == other.inner_state_;
  }
  constexpr auto operator==(End) const noexcept -> bool {
    return (inner_state_ == End{}) && (outer_state_ == End{});
  }
  constexpr NestedListIterator(
    ListIterator<O, OpOuter, NewInner> outer_state,
    ListIterator<I, OpInner, Proj> inner_state) noexcept
    : outer_state_{outer_state}, inner_state_{inner_state} {}
  constexpr NestedListIterator() = default;
};
template <class O, class I, class OpOuter, class OpInner, class NewInner,
          class Proj>
NestedListIterator(ListIterator<O, OpOuter, NewInner>,
                   ListIterator<I, OpInner, Proj>)
  -> NestedListIterator<O, I, OpOuter, OpInner, NewInner, Proj>;

template <typename T, class OpOuter, class OpInner, class NewInner, class Proj>
class NestedListRange {
  T *begin_;
  [[no_unique_address]] OpOuter nextOuter_{};
  [[no_unique_address]] OpInner nextInner_{};
  [[no_unique_address]] NewInner newInner_{};
  [[no_unique_address]] Proj p_{};

public:
  constexpr auto begin() noexcept {
    ListIterator outer{begin_, nextOuter_, newInner_};
    decltype(newInner_(begin_)) innerState =
      begin_ ? newInner_(begin_) : nullptr;
    ListIterator inner{innerState, nextInner_, p_};
    return NestedListIterator{outer, inner};
  }
  constexpr auto begin() const noexcept {
    const T *b = begin_;
    ListIterator outer{b, nextOuter_, newInner_};
    const decltype(newInner_(b)) innerState = b ? newInner_(b) : nullptr;
    ListIterator inner{innerState, nextInner_, p_};
    return NestedListIterator{outer, inner};
  }
  static constexpr auto end() noexcept -> End { return {}; }
  constexpr NestedListRange(T *begin, OpOuter outer, OpInner inner,
                            NewInner newInner, Proj p) noexcept
    : begin_{begin}, nextOuter_{outer}, nextInner_{inner}, newInner_{newInner},
      p_{p} {}
  constexpr NestedListRange(T *begin, OpOuter outer, OpInner inner,
                            NewInner newInner) noexcept
    : begin_{begin}, nextOuter_{outer}, nextInner_{inner}, newInner_{newInner} {
  }
  constexpr NestedListRange(T *begin) noexcept : begin_{begin} {}
};

template <typename T, class OpOuter, class OpInner, class NewInner>
NestedListRange(T *, OpOuter, OpInner, NewInner)
  -> NestedListRange<T, OpOuter, OpInner, NewInner, Identity>;
template <typename T, class OpOuter, class OpInner, class NewInner>
NestedListRange(NotNull<T>, OpOuter, OpInner, NewInner)
  -> NestedListRange<T, OpOuter, OpInner, NewInner, Identity>;
template <typename T, class OpOuter, class OpInner, class NewInner, class Proj>
NestedListRange(T *, OpOuter, OpInner, NewInner, Proj)
  -> NestedListRange<T, OpOuter, OpInner, NewInner, Proj>;
template <typename T, class OpOuter, class OpInner, class NewInner, class Proj>
NestedListRange(NotNull<T>, OpOuter, OpInner, NewInner, Proj)
  -> NestedListRange<T, OpOuter, OpInner, NewInner, Proj>;

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
  constexpr NestedIterator(O o, I i, P p, J j, F f) noexcept
    : outer{o}, inner{i}, outerend{p}, innerend{j}, innerfun(f) {}
};

/// NestedList
/// A range that lets us iterate over a graph via nesting ranges.
/// This is called a "nested list" because the output of each
/// range level is the head of the next iteration, i.e. it is useful
/// for nested list-like data structures, rather than (for example)
/// cartesian product iterators.
template <std::ranges::forward_range O, class F> class NestedList {
  static_assert(std::is_trivially_copyable_v<O>);
  static_assert(std::is_trivially_destructible_v<O>);
  [[no_unique_address]] O outer;
  [[no_unique_address]] F inner;

public:
  using InnerType = decltype(inner(*(outer.begin())));
  static_assert(std::is_trivially_copyable_v<InnerType>);
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
