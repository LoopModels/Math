#pragma once

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

template <typename T, class Op> class ListIterator {
  T *state_;
  [[no_unique_address]] Op op_;
  constexpr auto operator*() const noexcept -> T & { return *state_; }
  constexpr auto operator->() const noexcept -> T * { return state_; }
  constexpr auto operator++() noexcept -> ListIterator & {
    state_ = op_(state_);
    return *this;
  }
};
class ListEnd {};

template <typename T, class Op> class ListRange {
  T *begin_;
  [[no_unique_address]] Op op_;
  constexpr auto begin() noexcept -> ListIterator<T, Op> {
    return {begin_, op_};
  }
  constexpr auto begin() const noexcept -> ListIterator<const T, Op> {
    return {begin_, op_};
  }
  static constexpr auto end() noexcept -> ListEnd { return {}; }
};

// actually, implement via composition!
template <class O, class I, class OpOuter, class OpInner, class NewInner>
class NestedListIterator {
  O *outer_state_;
  I *inner_state_;
  [[no_unique_address]] OpOuter outer_;
  [[no_unique_address]] OpInner inner_;
  [[no_unique_address]] NewInner newInner_;
  constexpr auto operator*() const noexcept -> I & { return *inner_state_; }
  constexpr auto operator->() const noexcept -> I * { return inner_state_; }
  constexpr auto operator++() noexcept -> ListIterator & {
    state_ = op_(state_);
    return *this;
  }
};
template <class O, class I, class OpOuter, class OpInner, class NewInner>
NestedListIterator(O *outer_state, I *inner_state, OpOuter outer, OpInner inner,
                   NewInner newInner)
  -> NestedListIterator<O, I, OpOuter, OpInner, NewInner>;

template <typename T, class OpOuter, class OpInner, class NewInner>
class NestedListRange {
  T *begin_;
  [[no_unique_address]] OpOuter nextOuter_;
  [[no_unique_address]] OpInner nextInner_;
  [[no_unique_address]] NewInner newInner_;
  constexpr auto begin() noexcept {
    NestedListIterator iter{begin_, newInner_(begin_), nextOuter_, nextInner_,
                            newInner_};
    return iter;
  }
  constexpr auto begin() const noexcept {
    const T *b = begin_;
    NestedListIterator iter{b, newInner_(b), nextOuter_, nextInner_, newInner_};
    return iter;
  }
  static constexpr auto end() noexcept -> ListEnd { return {}; }
};

// static_assert(std::ranges::range<ListRange

} // namespace poly::utils
