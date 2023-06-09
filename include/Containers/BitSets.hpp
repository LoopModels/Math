#pragma once
#include "Math/Array.hpp"
#include "Utilities/Invariant.hpp"
#include <bits/ranges_base.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ios>
#include <istream>
#include <iterator>
#include <limits>
#include <ostream>
#include <string>

namespace poly::containers {

struct EndSentinel {
  [[nodiscard]] constexpr auto operator-(auto it) -> ptrdiff_t {
    ptrdiff_t i = 0;
    for (; it != EndSentinel{}; ++it, ++i) {}
    return i;
  }
  // overloaded operator== cannot be a static member function
  constexpr auto operator==(EndSentinel) const -> bool { return true; }
};

template <typename T>
concept CanResize = requires(T t) { t.resize(0); };

class BitSetIterator {
  [[no_unique_address]] const uint64_t *it;
  [[no_unique_address]] const uint64_t *end;
  [[no_unique_address]] uint64_t istate;
  [[no_unique_address]] ptrdiff_t cstate0{-1};
  [[no_unique_address]] ptrdiff_t cstate1{0};

public:
  constexpr explicit BitSetIterator(const uint64_t *_it, const uint64_t *_end,
                                    uint64_t _istate)
    : it{_it}, end{_end}, istate{_istate} {}
  using value_type = ptrdiff_t;
  using difference_type = ptrdiff_t;
  constexpr auto operator*() const -> ptrdiff_t { return cstate0 + cstate1; }
  constexpr auto operator++() -> BitSetIterator & {
    while (istate == 0) {
      if (++it == end) return *this;
      istate = *it;
      cstate0 = -1;
      cstate1 += 64;
    }
    ptrdiff_t tzp1 = std::countr_zero(istate);
    cstate0 += ++tzp1;
    istate >>= tzp1;
    return *this;
  }
  constexpr auto operator++(int) -> BitSetIterator {
    BitSetIterator temp = *this;
    ++*this;
    return temp;
  }
  constexpr auto operator==(EndSentinel) const -> bool {
    return it == end && (istate == 0);
  }
  constexpr auto operator!=(EndSentinel) const -> bool {
    return it != end || (istate != 0);
  }
  constexpr auto operator==(BitSetIterator j) const -> bool {
    return (it == j.it) && (istate == j.istate);
  }
  friend constexpr auto operator==(EndSentinel, const BitSetIterator &bt)
    -> bool {
    return bt.it == bt.end && (bt.istate == 0);
  }
  friend constexpr auto operator!=(EndSentinel, const BitSetIterator &bt)
    -> bool {
    return bt.it != bt.end || (bt.istate != 0);
  }
};

/// A set of `ptrdiff_t` elements.
/// Initially constructed
template <typename T = math::Vector<uint64_t, 1>> struct BitSet {
  [[no_unique_address]] T data{};
  // ptrdiff_t operator[](ptrdiff_t i) const {
  //     return data[i];
  // } // allow `getindex` but not `setindex`
  constexpr explicit BitSet() = default;
  static constexpr auto numElementsNeeded(ptrdiff_t N) -> unsigned {
    return unsigned(((N + 63) >> 6));
  }
  constexpr explicit BitSet(ptrdiff_t N) : data{numElementsNeeded(N), 0} {}
  constexpr void resize64(ptrdiff_t N) {
    if constexpr (CanResize<T>) data.resize(N);
    else ASSERT(N <= std::ssize(data));
  }
  constexpr void resize(ptrdiff_t N) {
    if constexpr (CanResize<T>) data.resize(numElementsNeeded(N));
    else ASSERT(N <= std::ssize(data) * 64);
  }
  constexpr void resize(ptrdiff_t N, uint64_t x) {
    if constexpr (CanResize<T>) data.resize(numElementsNeeded(N), x);
    else {
      ASSERT(N <= std::ssize(data) * 64);
      std::fill(data.begin(), data.end(), x);
    }
  }
  constexpr void maybeResize(ptrdiff_t N) {
    if constexpr (CanResize<T>) {
      ptrdiff_t M = numElementsNeeded(N);
      if (M > std::ssize(data)) data.resize(M);
    } else ASSERT(N <= std::ssize(data) * 64);
  }
  static constexpr auto dense(ptrdiff_t N) -> BitSet {
    BitSet b;
    ptrdiff_t M = numElementsNeeded(N);
    if (!M) return b;
    uint64_t maxval = std::numeric_limits<uint64_t>::max();
    if constexpr (CanResize<T>) b.data.resize(M, maxval);
    else
      for (ptrdiff_t i = 0; i < M - 1; ++i) b.data[i] = maxval;
    if (ptrdiff_t rem = N & 63) b.data[M - 1] = (ptrdiff_t(1) << rem) - 1;
    return b;
  }
  [[nodiscard]] constexpr auto maxValue() const -> ptrdiff_t {
    ptrdiff_t N = std::ssize(data);
    return N ? (64 * N - std::countl_zero(data[N - 1])) : 0;
  }
  // BitSet::Iterator(std::vector<std::uint64_t> &seta)
  //     : set(seta), didx(0), offset(0), state(seta[0]), count(0) {};
  [[nodiscard]] constexpr auto begin() const -> BitSetIterator {
    const uint64_t *b{data.begin()};
    const uint64_t *e{data.end()};
    if (b == e) return BitSetIterator{b, e, 0};
    BitSetIterator it{b, e, *b};
    return ++it;
  }
  [[nodiscard]] static constexpr auto end() -> EndSentinel {
    return EndSentinel{};
  };
  [[nodiscard]] constexpr auto front() const -> ptrdiff_t {
    for (ptrdiff_t i = 0; i < std::ssize(data); ++i)
      if (data[i]) return 64 * i + std::countr_zero(data[i]);
    return std::numeric_limits<ptrdiff_t>::max();
  }
  static constexpr auto contains(math::PtrVector<uint64_t> data, ptrdiff_t x)
    -> uint64_t {
    if (data.empty()) return 0;
    ptrdiff_t d = x >> ptrdiff_t(6);
    uint64_t r = uint64_t(x) & uint64_t(63);
    uint64_t mask = uint64_t(1) << r;
    return (data[d] & (mask));
  }
  [[nodiscard]] constexpr auto contains(ptrdiff_t i) const -> uint64_t {
    return contains(data, i);
  }
  struct Contains {
    const T &d;
    constexpr auto operator()(ptrdiff_t i) const -> uint64_t {
      return contains(d, i);
    }
  };
  [[nodiscard]] constexpr auto contains() const -> Contains {
    return Contains{data};
  }
  constexpr auto insert(ptrdiff_t x) -> bool {
    ptrdiff_t d = x >> ptrdiff_t(6);
    uint64_t r = uint64_t(x) & uint64_t(63);
    uint64_t mask = uint64_t(1) << r;
    if (d >= std::ssize(data)) resize64(d + 1);
    bool contained = ((data[d] & mask) != 0);
    if (!contained) data[d] |= (mask);
    return contained;
  }
  constexpr void uncheckedInsert(ptrdiff_t x) {
    ptrdiff_t d = x >> ptrdiff_t(6);
    uint64_t r = uint64_t(x) & uint64_t(63);
    uint64_t mask = uint64_t(1) << r;
    if (d >= std::ssize(data)) resize64(d + 1);
    data[d] |= (mask);
  }

  constexpr auto remove(ptrdiff_t x) -> bool {
    ptrdiff_t d = x >> ptrdiff_t(6);
    uint64_t r = uint64_t(x) & uint64_t(63);
    uint64_t mask = uint64_t(1) << r;
    bool contained = ((data[d] & mask) != 0);
    if (contained) data[d] &= (~mask);
    return contained;
  }
  static constexpr void set(uint64_t &d, ptrdiff_t r, bool b) {
    uint64_t mask = uint64_t(1) << r;
    if (b == ((d & mask) != 0)) return;
    if (b) d |= mask;
    else d &= (~mask);
  }
  static constexpr void set(math::MutPtrVector<uint64_t> data, ptrdiff_t x,
                            bool b) {
    ptrdiff_t d = x >> ptrdiff_t(6);
    uint64_t r = uint64_t(x) & uint64_t(63);
    set(data[d], r, b);
  }

  class Reference {
    [[no_unique_address]] math::MutPtrVector<uint64_t> data;
    [[no_unique_address]] ptrdiff_t i;

  public:
    constexpr explicit Reference(math::MutPtrVector<uint64_t> dd, ptrdiff_t ii)
      : data(dd), i(ii) {}
    constexpr explicit operator bool() const { return contains(data, i); }
    constexpr auto operator=(bool b) -> Reference & {
      BitSet::set(data, i, b);
      return *this;
    }
  };

  constexpr auto operator[](ptrdiff_t i) const -> bool {
    return contains(data, i);
  }
  constexpr auto operator[](ptrdiff_t i) -> Reference {
    maybeResize(i + 1);
    math::MutPtrVector<uint64_t> d{data};
    return Reference{d, i};
  }
  [[nodiscard]] constexpr auto size() const -> ptrdiff_t {
    ptrdiff_t s = 0;
    for (auto u : data) s += std::popcount(u);
    return s;
  }
  [[nodiscard]] constexpr auto empty() const -> bool {
    return std::ranges::all_of(data, [](auto u) { return u == 0; });
  }
  [[nodiscard]] constexpr auto any() const -> bool {
    return std::ranges::any_of(data, [](auto u) { return u != 0; });
    // for (auto u : data)
    //   if (u) return true;
    // return false;
  }
  constexpr void setUnion(const BitSet &bs) {
    ptrdiff_t O = std::ssize(bs.data), N = std::ssize(data);
    if (O > N) resize64(O);
    for (ptrdiff_t i = 0; i < O; ++i) {
      uint64_t d = data[i] | bs.data[i];
      data[i] = d;
    }
  }
  constexpr auto operator&=(const BitSet &bs) -> BitSet & {
    if (std::ssize(bs.data) < std::ssize(data)) resize64(std::ssize(bs.data));
    for (ptrdiff_t i = 0; i < std::ssize(data); ++i) data[i] &= bs.data[i];
    return *this;
  }
  // &!
  constexpr auto operator-=(const BitSet &bs) -> BitSet & {
    if (std::ssize(bs.data) < std::ssize(data)) resize64(std::ssize(bs.data));
    for (ptrdiff_t i = 0; i < std::ssize(data); ++i) data[i] &= (~bs.data[i]);
    return *this;
  }
  constexpr auto operator|=(const BitSet &bs) -> BitSet & {
    if (std::ssize(bs.data) > std::ssize(data)) resize64(std::ssize(bs.data));
    for (ptrdiff_t i = 0; i < std::ssize(bs.data); ++i) data[i] |= bs.data[i];
    return *this;
  }
  constexpr auto operator&(const BitSet &bs) const -> BitSet {
    BitSet r = *this;
    return r &= bs;
  }
  constexpr auto operator|(const BitSet &bs) const -> BitSet {
    BitSet r = *this;
    return r |= bs;
  }
  constexpr auto operator==(const BitSet &bs) const -> bool {
    return data == bs.data;
  }

  friend inline auto operator<<(std::ostream &os, BitSet const &x)
    -> std::ostream & {
    os << "BitSet[";
    auto it = x.begin();
    constexpr EndSentinel e = BitSet::end();
    if (it != e) {
      os << *(it++);
      for (; it != e; ++it) os << ", " << *it;
    }
    os << "]";
    return os;
  }
  [[nodiscard]] constexpr auto isEmpty() const -> bool {
    return std::ranges::all_of(data, [](auto u) { return u == 0; });
    // for (auto u : data)
    //   if (u) return false;
    // return true;
  }
};

template <unsigned N> using FixedSizeBitSet = BitSet<std::array<uint64_t, N>>;
// BitSet with length 64
using BitSet64 = FixedSizeBitSet<1>;
static_assert(std::is_trivially_destructible_v<BitSet64>);
static_assert(std::is_trivially_destructible_v<FixedSizeBitSet<2>>);
// static_assert(std::input_or_output_iterator<
//               decltype(std::declval<FixedSizeBitSet<2>>().begin())>);
static_assert(std::ranges::range<FixedSizeBitSet<2>>);

template <typename T, typename B = BitSet<>> struct BitSliceView {
  [[no_unique_address]] math::MutPtrVector<T> a;
  [[no_unique_address]] const B &i;
  struct Iterator {
    [[no_unique_address]] math::MutPtrVector<T> a;
    [[no_unique_address]] BitSetIterator it;
    constexpr auto operator==(EndSentinel) const -> bool {
      return it == EndSentinel{};
    }
    constexpr auto operator++() -> Iterator & {
      ++it;
      return *this;
    }
    constexpr auto operator++(int) -> Iterator {
      Iterator temp = *this;
      ++it;
      return temp;
    }
    constexpr auto operator*() -> T & { return a[*it]; }
    constexpr auto operator*() const -> const T & { return a[*it]; }
    constexpr auto operator->() -> T * { return &a[*it]; }
    constexpr auto operator->() const -> const T * { return &a[*it]; }
  };
  constexpr auto begin() -> Iterator { return {a, i.begin()}; }
  struct ConstIterator {
    [[no_unique_address]] math::PtrVector<T> a;
    [[no_unique_address]] BitSetIterator it;
    constexpr auto operator==(EndSentinel) const -> bool {
      return it == EndSentinel{};
    }
    constexpr auto operator==(ConstIterator c) const -> bool {
      return (it == c.it) && (a.data() == c.a.data());
    }
    constexpr auto operator++() -> ConstIterator & {
      ++it;
      return *this;
    }
    constexpr auto operator++(int) -> ConstIterator {
      ConstIterator temp = *this;
      ++it;
      return temp;
    }
    constexpr auto operator*() const -> const T & { return a[*it]; }
    constexpr auto operator->() const -> const T * { return &a[*it]; }
  };
  [[nodiscard]] constexpr auto begin() const -> ConstIterator {
    return {a, i.begin()};
  }
  [[nodiscard]] constexpr auto end() const -> EndSentinel { return {}; }
  [[nodiscard]] constexpr auto size() const -> ptrdiff_t { return i.size(); }
};
[[nodiscard]] constexpr auto operator-(EndSentinel,
                                       BitSliceView<int64_t>::Iterator v)
  -> ptrdiff_t {
  return EndSentinel{} - v.it;
}
[[nodiscard]] constexpr auto operator-(EndSentinel,
                                       BitSliceView<int64_t>::ConstIterator v)
  -> ptrdiff_t {
  return EndSentinel{} - v.it;
}
template <typename T, typename B>
BitSliceView(math::MutPtrVector<T>, const B &) -> BitSliceView<T, B>;

static_assert(std::movable<BitSliceView<int64_t>::Iterator>);
static_assert(std::movable<BitSliceView<int64_t>::ConstIterator>);
} // namespace poly::containers
