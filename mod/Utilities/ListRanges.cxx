#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifndef USE_MODULE
#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>

#include "Utilities/Invariant.cxx"
#include "Utilities/Valid.cxx"
#else
export module ListRange;

import Invariant;
import STL;
import Valid;
#endif

#ifdef USE_MODULE
export namespace utils {
#else
namespace utils {
#endif

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
  constexpr auto operator*() const noexcept -> value_type {
    invariant(state_ != next_);
    return p_(state_);
  }
  // constexpr auto operator->() const noexcept -> T * { return state_; }
  constexpr auto getState() const noexcept -> T * { return state_; }
  constexpr auto operator++() noexcept -> ListIterator & {
    invariant(state_ != next_);
    state_ = next_;
    if (next_) next_ = op_(next_);
    return *this;
  }
  constexpr auto operator++(int) noexcept -> ListIterator {
    invariant(state_ != next_);
    ListIterator tmp{*this};
    state_ = next_;
    if (next_) next_ = op_(next_);
    return tmp;
  }
  constexpr auto
  operator-(ListIterator const &other) const noexcept -> ptrdiff_t {
    ptrdiff_t count = 0;
    for (auto iter = other; iter != *this; ++iter) ++count;
    return count;
  }
  constexpr auto operator==(ListIterator const &other) const noexcept -> bool {
    return state_ == other.state_;
  }
  constexpr auto operator==(End) const noexcept -> bool {
    invariant((state_ == nullptr) || (state_ != next_));
    return state_ == nullptr;
  }
  constexpr ListIterator(T *state) noexcept
    : state_{state}, next_{state ? Op{}(state) : nullptr} {}
  constexpr ListIterator(T *state, Op op, Proj p) noexcept
    : state_{state}, next_{state ? op(state) : nullptr}, op_{op}, p_{p} {}
  constexpr ListIterator() noexcept = default;
  constexpr ListIterator(const ListIterator &) noexcept = default;
  constexpr ListIterator(ListIterator &&) noexcept = default;
  constexpr auto
  operator=(const ListIterator &) noexcept -> ListIterator & = default;
  constexpr auto
  operator=(ListIterator &&) noexcept -> ListIterator & = default;
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
  // constexpr auto operator|(std::invocable<ListRange> auto&&x) {
  //     return x(*this);
  // }
  template <std::invocable<ListRange> F> constexpr auto operator|(F &&f) const {
    return std::forward<F>(f)(*this);
  }
};
template <typename T, class Op, class Proj>
ListRange(T *, Op, Proj) -> ListRange<T, Op, Proj>;
template <typename T, class Op, class Proj>
ListRange(Valid<T>, Op, Proj) -> ListRange<T, Op, Proj>;

template <typename T, class Op>
ListRange(T *, Op) -> ListRange<T, Op, Identity>;
template <typename T, class Op>
ListRange(Valid<T>, Op) -> ListRange<T, Op, Identity>;

struct NoInnerObj {};
struct NoInnerState {};
struct NoInnerEnd {};
template <std::forward_iterator O, std::forward_iterator I, class P, class J,
          class F, class L>
class NestedIterator {
  [[no_unique_address]] O outer;
  [[no_unique_address]] P outerend;
  [[no_unique_address]] F innerfun;
  union {
    [[no_unique_address]] NoInnerObj noobj{};
    [[no_unique_address]] L innerobj; // keep the inner object alive!
  };
  union {
    [[no_unique_address]] NoInnerState nostate{};
    [[no_unique_address]] I inner;
  };
  union {
    [[no_unique_address]] NoInnerEnd noend{};
    [[no_unique_address]] J innerend;
  };

  constexpr void initInner() {
    for (;;) {
      innerobj = innerfun(*outer);
      inner = innerobj.begin();
      innerend = innerobj.end();
      if (inner != innerend) break;
      if (++outer == outerend) break;
    }
  }
  constexpr void initInnerConstructor() {
    ::new (&innerobj) L{innerfun(*outer)};
    ::new (&inner) I{innerobj.begin()};
    ::new (&innerend) J{innerobj.end()};
    if ((inner == innerend) && (++outer != outerend)) initInner();
  }
  constexpr void destroyInner() {
    if constexpr (!std::is_trivially_destructible_v<J>) innerend.~J();
    if constexpr (!std::is_trivially_destructible_v<I>) inner.~I();
    if constexpr (!std::is_trivially_destructible_v<L>) innerobj.~L();
    noend = {};
    nostate = {};
    noobj = {};
  }

public:
  using value_type = decltype(*inner);

  constexpr auto
  operator==(NestedIterator const &other) const noexcept -> bool {
    return outer == other.outer && inner == other.inner;
  }
  constexpr auto operator==(End) const noexcept -> bool {
    return outer == outerend;
  }
  constexpr auto operator++() noexcept -> NestedIterator & {
    if (++inner != innerend) return *this;
    if (++outer != outerend) initInner();
    else destroyInner(); // outer == outerend means undef
    return *this;
  }
  constexpr auto operator++(int) noexcept -> NestedIterator {
    NestedIterator tmp{*this};
    ++*this;
    return tmp;
  }
  constexpr auto operator*() const noexcept -> value_type { return *inner; }
  constexpr auto operator->() const noexcept -> I { return inner; }
  constexpr auto
  operator-(NestedIterator const &other) const noexcept -> ptrdiff_t {
    ptrdiff_t count = 0;
    for (auto iter = other; iter != *this; ++iter) ++count;
    return count;
  }
  constexpr NestedIterator() noexcept = default;
  // constexpr NestedIterator(const NestedIterator &) noexcept = default;
  // constexpr NestedIterator(NestedIterator &&) noexcept = default;
  // constexpr auto operator=(const NestedIterator &) noexcept
  //   -> NestedIterator & = default;
  // constexpr auto operator=(NestedIterator &&) noexcept
  //   -> NestedIterator & = default;

  constexpr NestedIterator(auto &out, auto &innerfun_) noexcept
    : outer{out.begin()}, outerend{out.end()}, innerfun{innerfun_} {
    if (outer != outerend) initInnerConstructor();
  }
  constexpr NestedIterator(const auto &out, const auto &innerfun_) noexcept
    : outer{out.begin()}, outerend{out.end()}, innerfun{innerfun_} {
    if (outer != outerend) initInnerConstructor();
  }
  constexpr NestedIterator(const NestedIterator &other) noexcept
    : outer{other.outer}, outerend{other.outerend}, innerfun{other.innerfun} {
    if (outer != outerend) {
      ::new (&innerobj) L{other.innerobj};
      ::new (&inner) I{other.inner};
      ::new (&innerend) J{other.innerend};
    }
  }
  constexpr NestedIterator(NestedIterator &&other) noexcept
    : outer{std::move(other.outer)}, outerend{std::move(other.outerend)},
      innerfun{std::move(other.innerfun)} {
    if (outer != outerend) {
      ::new (&innerobj) L{std::move(other.innerobj)};
      ::new (&inner) I{std::move(other.inner)};
      ::new (&innerend) J{std::move(other.innerend)};
    }
  }
  constexpr auto
  operator=(const NestedIterator &other) noexcept -> NestedIterator & {
    if (this == &other) return *this;
    if (outer != outerend) destroyInner();
    outer = other.outer;
    outerend = other.outerend;
    innerfun = other.innerfun;
    if (outer != outerend) {
      ::new (&innerobj) L{other.innerobj};
      ::new (&inner) I{other.inner};
      ::new (&innerend) J{other.innerend};
    }
  }
  constexpr auto
  operator=(NestedIterator &&other) noexcept -> NestedIterator & {
    if (this == &other) return *this;
    if (outer != outerend) destroyInner();
    outer = std::move(other.outer);
    outerend = std::move(other.outerend);
    innerfun = std::move(other.innerfun);
    if (outer != outerend) {
      ::new (&innerobj) L{std::move(other.innerobj)};
      ::new (&inner) I{std::move(other.inner)};
      ::new (&innerend) J{std::move(other.innerend)};
    }
  }
  ~NestedIterator() {
    if (outer != outerend) destroyInner();
  }
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
  using IteratorType =
    NestedIterator<decltype(outer.begin()), InnerBegin, decltype(outer.end()),
                   InnerEnd, F, InnerType>;

public:
  constexpr auto begin() noexcept -> IteratorType {
    return IteratorType(outer, inner);
  }
  static constexpr auto end() noexcept -> End { return {}; }
  constexpr NestedList(O out, F inn) noexcept
    : outer{std::move(out)}, inner{std::move(inn)} {}
  constexpr NestedList(const NestedList &) noexcept = default;
  constexpr NestedList(NestedList &&) noexcept = default;
  constexpr auto
  operator=(const NestedList &) noexcept -> NestedList & = default;
  constexpr auto operator=(NestedList &&) noexcept -> NestedList & = default;
};
template <std::ranges::forward_range O, class F>
NestedList(O, F) -> NestedList<O, F>;

} // namespace utils

template <typename T, class Op, class Proj>
inline constexpr bool
  std::ranges::enable_borrowed_range<utils::ListRange<T, Op, Proj>> = true;
template <std::ranges::forward_range O, class F>
inline constexpr bool
  std::ranges::enable_borrowed_range<utils::NestedList<O, F>> =
    std::ranges::enable_borrowed_range<O> &&
    std::ranges::enable_borrowed_range<
      typename utils::NestedList<O, F>::InnerType>;
