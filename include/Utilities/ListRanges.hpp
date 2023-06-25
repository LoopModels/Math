#pragma once

#include <cstddef>
#include <ranges>

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

class ListEnd {};

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
  constexpr auto operator==(ListEnd) const noexcept -> bool {
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
  static constexpr auto end() noexcept -> ListEnd { return {}; }
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
    if (inner_state_ == ListEnd{}) {
      ++outer_state_;
      if (outer_state_ != ListEnd{})
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
  constexpr auto operator==(ListEnd) const noexcept -> bool {
    return (inner_state_ == ListEnd{}) && (outer_state_ == ListEnd{});
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
  static constexpr auto end() noexcept -> ListEnd { return {}; }
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

// static_assert(std::ranges::range<ListRange

} // namespace poly::utils
