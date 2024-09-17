#ifdef USE_MODULE
module;
#else
#pragma once
#endif
#ifndef USE_MODULE
#include <algorithm>
#include <array>
#include <bit>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>
#include <ostream>
#include <ranges>
#include <string>
#include <type_traits>

#include "Math/Array.cxx"
#include "Math/AxisTypes.cxx"
#include "Math/ManagedArray.cxx"
#include "Math/MatrixDimensions.cxx"
#include "Math/Ranges.cxx"
#include "Utilities/Invariant.cxx"
#else
export module BitSet;

import Array;
import AxisTypes;
import Invariant;
import ManagedArray;
import MatDim;
import Range;
import STL;
#endif

#ifdef USE_MODULE
export namespace containers {
#else
namespace containers {
#endif
using utils::invariant;

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

template <std::unsigned_integral U> class BitSetIterator {
  [[no_unique_address]] const U *it_;
  [[no_unique_address]] const U *end_;
  [[no_unique_address]] U istate_;
  ptrdiff_t cstate0_{-1};
  ptrdiff_t cstate1_{0};

public:
  constexpr explicit BitSetIterator(const U *_it, const U *_end, U _istate)
    : it_{_it}, end_{_end}, istate_{_istate} {}
  using value_type = ptrdiff_t;
  using difference_type = ptrdiff_t;
  constexpr auto operator*() const -> ptrdiff_t { return cstate0_ + cstate1_; }
  constexpr auto operator++() -> BitSetIterator & {
    while (istate_ == 0) {
      if (++it_ == end_) return *this;
      istate_ = *it_;
      cstate0_ = -1;
      cstate1_ += 8 * sizeof(U);
    }
    ptrdiff_t tzp1 = std::countr_zero(istate_);
    cstate0_ += ++tzp1;
    istate_ >>= tzp1;
    return *this;
  }
  constexpr auto operator++(int) -> BitSetIterator {
    BitSetIterator temp = *this;
    ++*this;
    return temp;
  }
  constexpr auto operator==(EndSentinel) const -> bool {
    return it_ == end_ && (istate_ == 0);
  }
  constexpr auto operator!=(EndSentinel) const -> bool {
    return it_ != end_ || (istate_ != 0);
  }
  constexpr auto operator==(BitSetIterator j) const -> bool {
    return (it_ == j.it_) && (istate_ == j.istate_);
  }
  friend constexpr auto operator==(EndSentinel,
                                   const BitSetIterator &bt) -> bool {
    return bt.it_ == bt.end_ && (bt.istate_ == 0);
  }
  friend constexpr auto operator!=(EndSentinel,
                                   const BitSetIterator &bt) -> bool {
    return bt.it_ != bt.end_ || (bt.istate_ != 0);
  }
};

template <typename T>
concept Collection = requires(T t) {
  { std::size(t) } -> std::convertible_to<size_t>;
  { std::ssize(t) } -> std::convertible_to<ptrdiff_t>;
  { *t.begin() } -> std::convertible_to<uint64_t>;
};

/// A set of `ptrdiff_t` elements.
/// Initially constructed
template <Collection T = math::Vector<uint64_t, 1>> struct BitSet {
  using U = utils::eltype_t<T>;
  static constexpr U usize = 8 * sizeof(U);
  static constexpr U umask = usize - 1;
  static constexpr U ushift = std::countr_zero(usize);
  [[no_unique_address]] T data_{};
  // ptrdiff_t operator[](ptrdiff_t i) const {
  //     return data[i];
  // } // allow `getindex` but not `setindex`
  constexpr explicit BitSet() = default;
  constexpr explicit BitSet(T &&_data) : data_{std::move(_data)} {}
  constexpr explicit BitSet(const T &_data) : data_{_data} {}
  static constexpr auto numElementsNeeded(ptrdiff_t N) -> math::Length<> {
    return math::length(((N + usize - 1) >> ushift));
  }
  constexpr explicit BitSet(ptrdiff_t N) : data_{numElementsNeeded(N), 0} {}
  static constexpr auto fromMask(U u) -> BitSet { return BitSet{T{u}}; }
  constexpr void resizeData(ptrdiff_t N) {
    if constexpr (CanResize<T>) data_.resize(N);
    else invariant(N <= std::ssize(data_));
  }
  constexpr void resize(ptrdiff_t N) {
    if constexpr (CanResize<T>) data_.resize(numElementsNeeded(N));
    else invariant(N <= std::ssize(data_) * usize);
  }
  constexpr void maybeResize(ptrdiff_t N) {
    if constexpr (CanResize<T>) {
      math::Length<> M = numElementsNeeded(N);
      if (M > std::ssize(data_)) data_.resize(M);
    } else invariant(N <= std::ssize(data_) * ptrdiff_t(usize));
  }
  static constexpr auto dense(ptrdiff_t N) -> BitSet {
    BitSet b{};
    math::Length M = numElementsNeeded(N);
    if (!M) return b;
    U maxval = std::numeric_limits<U>::max();
    if constexpr (CanResize<T>) b.data_.resizeForOverwrite(M);
    --M;
    for (ptrdiff_t i = 0z; i < M; ++i) b.data_[i] = maxval;
    if (ptrdiff_t rem = N & (usize - 1))
      b.data_[ptrdiff_t(M)] = (1z << rem) - 1z;
    return b;
  }
  [[nodiscard]] constexpr auto maxValue() const -> ptrdiff_t {
    ptrdiff_t N = std::ssize(data_);
    return N ? ((usize * N) - std::countl_zero(data_[N - 1])) : 0;
  }
  // BitSet::Iterator(std::vector<std::U> &seta)
  //     : set(seta), didx(0), offset(0), state(seta[0]), count(0) {};
  [[nodiscard]] constexpr auto begin() const -> BitSetIterator<U> {
    auto be = data_.begin();
    auto de = data_.end();
    const U *b{be};
    const U *e{de};
    if (b == e) return BitSetIterator<U>{b, e, 0};
    BitSetIterator it{b, e, *b};
    return ++it;
  }
  [[nodiscard]] static constexpr auto end() -> EndSentinel {
    return EndSentinel{};
  };
  [[nodiscard]] constexpr auto front() const -> ptrdiff_t {
    for (ptrdiff_t i = 0; i < std::ssize(data_); ++i)
      if (data_[i]) return (usize * i) + std::countr_zero(data_[i]);
    return std::numeric_limits<ptrdiff_t>::max();
  }
  static constexpr auto contains(math::PtrVector<U> data, ptrdiff_t x) -> U {
    if (data.empty()) return 0;
    ptrdiff_t d = x >> ptrdiff_t(ushift);
    U r = U(x) & umask;
    U mask = U(1) << r;
    return (data[d] & (mask));
  }
  /// Returns `true` if `i` is in the `BitSet`
  [[nodiscard]] constexpr auto contains(ptrdiff_t i) const -> U {
    return contains(data_, i);
  }
  struct Contains {
    const T &d_;
    constexpr auto operator()(ptrdiff_t i) const -> U {
      return contains(d_, i);
    }
  };
  [[nodiscard]] constexpr auto contains() const -> Contains {
    return Contains{data_};
  }
  constexpr auto insert(ptrdiff_t x) -> bool {
    ptrdiff_t d = x >> ptrdiff_t(ushift);
    U r = U(x) & umask;
    U mask = U(1) << r;
    if (d >= std::ssize(data_)) resizeData(d + 1);
    bool contained = ((data_[d] & mask) != 0);
    if (!contained) data_[d] |= (mask);
    return contained;
  }
  constexpr void uncheckedInsert(ptrdiff_t x) {
    ptrdiff_t d = x >> ushift;
    U r = U(x) & umask;
    U mask = U(1) << r;
    if (d >= std::ssize(data_)) resizeData(d + 1);
    data_[d] |= (mask);
  }
  // returns `true` the bitset contained `x`, i.e. if the
  // removal was succesful.
  constexpr auto remove(ptrdiff_t x) -> bool {
    ptrdiff_t d = x >> ushift;
    U r = U(x) & umask;
    U mask = U(1) << r;
    bool contained = ((data_[d] & mask) != 0);
    if (contained) data_[d] &= (~mask);
    return contained;
  }
  static constexpr void set(U &d, ptrdiff_t r, bool b) {
    U mask = U(1) << r;
    if (b == ((d & mask) != 0)) return;
    if (b) d |= mask;
    else d &= (~mask);
  }
  static constexpr void set(math::MutPtrVector<U> data, ptrdiff_t x, bool b) {
    ptrdiff_t d = x >> ushift;
    U r = U(x) & umask;
    set(data[d], r, b);
  }

  class Reference {
    [[no_unique_address]] math::MutPtrVector<U> data_;
    [[no_unique_address]] ptrdiff_t i_;

  public:
    constexpr explicit Reference(math::MutPtrVector<U> dd, ptrdiff_t ii)
      : data_(dd), i_(ii) {}
    constexpr operator bool() const { return contains(data_, i_); }
    constexpr auto operator=(bool b) -> Reference & {
      BitSet::set(data_, i_, b);
      return *this;
    }
  };

  constexpr auto operator[](ptrdiff_t i) const -> bool {
    return contains(data_, i);
  }
  constexpr auto operator[](ptrdiff_t i) -> Reference {
    maybeResize(i + 1);
    math::MutPtrVector<U> d{data_};
    return Reference{d, i};
  }
  [[nodiscard]] constexpr auto size() const -> ptrdiff_t {
    ptrdiff_t s = 0;
    for (auto u : data_) s += std::popcount(u);
    return s;
  }
  [[nodiscard]] constexpr auto empty() const -> bool {
    return std::ranges::all_of(data_, [](auto u) { return u == 0; });
  }
  [[nodiscard]] constexpr auto any() const -> bool {
    return std::ranges::any_of(data_, [](auto u) { return u != 0; });
    // for (auto u : data)
    //   if (u) return true;
    // return false;
  }
  constexpr void setUnion(const BitSet &bs) {
    ptrdiff_t O = std::ssize(bs.data_), N = std::ssize(data_);
    if (O > N) resizeData(O);
    for (ptrdiff_t i = 0; i < O; ++i) {
      U d = data_[i] | bs.data_[i];
      data_[i] = d;
    }
  }
  constexpr auto operator&=(const BitSet &bs) -> BitSet & {
    if (std::ssize(bs.data_) < std::ssize(data_))
      resizeData(std::ssize(bs.data_));
    for (ptrdiff_t i = 0; i < std::ssize(data_); ++i) data_[i] &= bs.data_[i];
    return *this;
  }
  // &!
  constexpr auto operator-=(const BitSet &bs) -> BitSet & {
    if (std::ssize(bs.data_) < std::ssize(data_))
      resizeData(std::ssize(bs.data_));
    for (ptrdiff_t i = 0; i < std::ssize(data_); ++i)
      data_[i] &= (~bs.data_[i]);
    return *this;
  }
  constexpr auto operator|=(const BitSet &bs) -> BitSet & {
    if (std::ssize(bs.data_) > std::ssize(data_))
      resizeData(std::ssize(bs.data_));
    for (ptrdiff_t i = 0; i < std::ssize(bs.data_); ++i)
      data_[i] |= bs.data_[i];
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
    return data_ == bs.data_;
  }
  // Ranks higher elements as more important, thus iterating
  // backwards.
  constexpr auto
  operator<=>(const BitSet &other) const -> std::strong_ordering {
    ptrdiff_t ntd = data_.size(), nod = other.data_.size();
    if (ntd != nod) {
      bool larger = ntd > nod;
      ptrdiff_t l = std::min(ntd, nod), L = std::max(ntd, nod);
      const T &d = larger ? data_ : other.data_;
      // The other is effectively all `0`, thus we may as well iterate forwards.
      for (ptrdiff_t i = l; i < L; ++i)
        if (d[i])
          return larger ? std::strong_ordering::greater
                        : std::strong_ordering::less;
      ntd = l;
    }
    for (ptrdiff_t i = ntd; i--;)
      if (auto cmp = data_[i] <=> other.data_[i]; cmp != 0) return cmp;
    return std::strong_ordering::equal;
  }

  friend auto operator<<(std::ostream &os, BitSet const &x) -> std::ostream & {
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
  constexpr void clear() { std::fill_n(data_.begin(), std::ssize(data_), 0); }
  [[nodiscard]] constexpr auto isEmpty() const -> bool {
    return std::ranges::all_of(data_, [](auto u) { return u == 0; });
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
    [[no_unique_address]] BitSetIterator<uint64_t> it;
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
    [[no_unique_address]] BitSetIterator<uint64_t> it;
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
  [[nodiscard]] friend constexpr auto operator-(EndSentinel,
                                                Iterator v) -> ptrdiff_t {
    return EndSentinel{} - v.it;
  }
  [[nodiscard]] friend constexpr auto operator-(EndSentinel,
                                                ConstIterator v) -> ptrdiff_t {
    return EndSentinel{} - v.it;
  }
};
template <typename T, typename B>
BitSliceView(math::MutPtrVector<T>, const B &) -> BitSliceView<T, B>;

static_assert(std::movable<BitSliceView<int64_t>::Iterator>);
static_assert(std::movable<BitSliceView<int64_t>::ConstIterator>);
} // namespace containers
