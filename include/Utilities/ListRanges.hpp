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

class ListEnd {};

template <typename T, class Op> class ListIterator {
  T *state_;
  [[no_unique_address]] Op op_{};

public:
  using value_type = T *;
  constexpr auto operator*() const noexcept -> T * { return state_; }
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
  constexpr ListIterator(T *state, Op op) noexcept : state_{state}, op_{op} {}
  constexpr ListIterator() = default;
};
template <typename T, class Op> ListIterator(T *, Op) -> ListIterator<T, Op>;

template <typename T, class Op> class ListRange {
  T *begin_;
  [[no_unique_address]] Op op_{};

public:
  constexpr auto begin() noexcept -> ListIterator<T, Op> {
    return {begin_, op_};
  }
  constexpr auto begin() const noexcept -> ListIterator<const T, Op> {
    return {begin_, op_};
  }
  static constexpr auto end() noexcept -> ListEnd { return {}; }
  constexpr ListRange(T *begin, Op op) noexcept : begin_{begin}, op_{op} {}
  constexpr ListRange(T *begin) noexcept : begin_{begin} {}
};
template <typename T, class Op> ListRange(T *, Op) -> ListRange<T, Op>;
// actually, implement via composition!
// Implementation detail: the current inner state corresponds to the current
// outer state. When we run out of inner states, we increment outer and set
// inner.
template <class O, class I, class OpOuter, class OpInner, class NewInner>
class NestedListIterator {
  ListIterator<O, OpOuter> outer_state_;
  ListIterator<I, OpInner> inner_state_;
  [[no_unique_address]] NewInner newInner_{};

public:
  using value_type = I *;
  constexpr auto operator*() const noexcept -> I * { return *inner_state_; }
  // constexpr auto operator->() const noexcept -> I * {
  //   return inner_state_.getState();
  // }
  constexpr auto operator++() noexcept -> NestedListIterator & {
    ++inner_state_;
    if (inner_state_ == ListEnd{}) {
      ++outer_state_;
      if (outer_state_ != ListEnd{})
        inner_state_ = {newInner_(outer_state_.getState()), OpInner{}};
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
  constexpr NestedListIterator(O *outer_state, I *inner_state, OpOuter outer,
                               OpInner inner, NewInner newInner) noexcept
    : outer_state_{outer_state, outer}, inner_state_{inner_state, inner},
      newInner_{newInner} {}
  constexpr NestedListIterator(ListIterator<O, OpOuter> outer_state,
                               ListIterator<I, OpInner> inner_state,
                               NewInner newInner) noexcept
    : outer_state_{outer_state}, inner_state_{inner_state},
      newInner_{newInner} {}
  constexpr NestedListIterator() = default;
};
template <class O, class I, class OpOuter, class OpInner, class NewInner>
NestedListIterator(O *, I *, OpOuter, OpInner, NewInner)
  -> NestedListIterator<O, I, OpOuter, OpInner, NewInner>;
template <class O, class I, class OpOuter, class OpInner, class NewInner>
NestedListIterator(ListIterator<O, OpOuter>, ListIterator<I, OpInner>, NewInner)
  -> NestedListIterator<O, I, OpOuter, OpInner, NewInner>;

template <typename T, class OpOuter, class OpInner, class NewInner>
class NestedListRange {
  T *begin_;
  [[no_unique_address]] OpOuter nextOuter_{};
  [[no_unique_address]] OpInner nextInner_{};
  [[no_unique_address]] NewInner newInner_{};

public:
  constexpr auto begin() noexcept {
    ListIterator<T, OpOuter> outer{begin_, nextOuter_};
    decltype(newInner_(begin_)) innerState =
      begin_ ? newInner_(begin_) : nullptr;
    ListIterator inner{innerState, nextInner_};
    NestedListIterator iter{outer, inner, newInner_};
    return iter;
  }
  constexpr auto begin() const noexcept {
    const T *b = begin_;
    ListIterator<T, OpOuter> outer{b, nextOuter_};
    const decltype(newInner_(b)) innerState = b ? newInner_(b) : nullptr;
    ListIterator inner{innerState, nextInner_};
    NestedListIterator iter{outer, inner, newInner_};
    return iter;
  }
  static constexpr auto end() noexcept -> ListEnd { return {}; }
  constexpr NestedListRange(T *begin, OpOuter outer, OpInner inner,
                            NewInner newInner) noexcept
    : begin_{begin}, nextOuter_{outer}, nextInner_{inner}, newInner_{newInner} {
  }
  constexpr NestedListRange(T *begin) noexcept : begin_{begin} {}
};

// static_assert(std::ranges::range<ListRange

} // namespace poly::utils
