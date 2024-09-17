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
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "Alloc/Mallocator.cxx"
#include "Containers/Pair.cxx"
#include "Containers/Tuple.cxx"
#include "Math/Array.cxx"
#include "Math/AxisTypes.cxx"
#include "Math/MatrixDimensions.cxx"
#else
export module SOA;

import Allocator;
import Array;
import AxisTypes;
import MatDim;
import Pair;
import STL;
import Tuple;
#endif

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif

template <typename... T> struct Types {};

static_assert(std::tuple_size_v<containers::Tuple<int, double>> == 2);
static_assert(std::tuple_size_v<containers::Pair<int, double>> == 2);

template <typename T,
          typename I = std::make_index_sequence<std::tuple_size_v<T>>>
struct TupleTypes {
  using type = void;
};

template <typename T, size_t... I>
struct TupleTypes<T, std::index_sequence<I...>> {
  using type = Types<std::tuple_element_t<I, T>...>;
};

template <typename T> using TupleTypes_t = typename TupleTypes<T>::type;

static_assert(std::is_same_v<TupleTypes_t<containers::Tuple<int, double>>,
                             Types<int, double>>);

template <typename T, typename U = TupleTypes<T>::type>
struct CanConstructFromMembers {
  static constexpr bool value = false;
};
template <typename T, typename... Elts>
struct CanConstructFromMembers<T, Types<Elts...>> {
  static constexpr bool value = std::is_constructible_v<T, Elts...>;
};

static_assert(CanConstructFromMembers<containers::Tuple<int, double>>::value);

namespace CapacityCalculators {

struct Length {
  static constexpr auto operator()(auto sz) -> ptrdiff_t {
    return static_cast<ptrdiff_t>(sz);
  }
  constexpr Length() = default;
  constexpr Length(auto) {};
  /// args are old nz, new nz
  /// returns `0` if not growing
  constexpr auto oldnewcap(auto osz, auto nsz) -> std::array<ptrdiff_t, 2> {
    return {(*this)(osz), (*this)(nsz)};
  }
  constexpr auto operator=(ptrdiff_t) -> Length & { return *this; }
};
constexpr auto nextpow2(ptrdiff_t x) -> ptrdiff_t {
  return static_cast<ptrdiff_t>(std::bit_ceil(static_cast<size_t>(x)));
}
struct NextPow2 {
  static constexpr auto operator()(auto sz) -> ptrdiff_t {
    ptrdiff_t i = ptrdiff_t(sz);
    return i ? std::max(8z, nextpow2(i)) : 0z;
  }
  constexpr NextPow2() = default;
  constexpr NextPow2(auto) {};
  /// args are old nz, new nz
  /// returns `0` if not growing
  constexpr auto oldnewcap(auto osz, auto nsz) -> std::array<ptrdiff_t, 2> {
    return {(*this)(osz), (*this)(nsz)};
  }
  constexpr auto operator=(ptrdiff_t) -> NextPow2 & { return *this; }
};
struct Explicit {
  ptrdiff_t capacity_{0z};
  constexpr Explicit() = default;
  constexpr Explicit(ptrdiff_t sz)
    : capacity_{((sz > 8z) ? nextpow2(sz) : 8z * (sz > 0))} {}
  constexpr Explicit(::math::Length<> sz) : Explicit(ptrdiff_t(sz)) {}
  constexpr auto operator()(auto) const -> ptrdiff_t { return capacity_; }
  constexpr auto operator=(ptrdiff_t cap) -> Explicit & {
    capacity_ = cap;
    return *this;
  }
  /// args are old nz, new nz
  /// returns `0` if not growing
  constexpr auto oldnewcap(auto, auto sz) -> std::array<ptrdiff_t, 2> {
    return {capacity_, nextpow2(sz)};
  }
};

} // namespace CapacityCalculators

template <size_t I, typename T> struct CumSizeOf {
  static constexpr size_t value = 0;
};

template <size_t I, typename T, typename... Ts>
struct CumSizeOf<I, Types<T, Ts...>> {
  static constexpr size_t value =
    sizeof(T) + CumSizeOf<I - 1, Types<Ts...>>::value;
};
template <typename T, typename... Ts> struct CumSizeOf<0, Types<T, Ts...>> {
  static constexpr size_t value = 0;
};
template <> struct CumSizeOf<0, Types<>> {
  static constexpr size_t value = 0;
};
template <size_t I, typename T>
inline constexpr size_t CumSizeOf_v = CumSizeOf<I, TupleTypes_t<T>>::value;

template <typename T>
concept ConstructibleFromMembers = CanConstructFromMembers<T>::value;

template <ConstructibleFromMembers T, size_t... II> struct SOAReference {
  char *ptr_;
  ptrdiff_t stride_;
  ptrdiff_t i_;
  operator T() const {
    char *p = std::assume_aligned<16>(ptr_);
    return T(*reinterpret_cast<std::tuple_element_t<II, T> *>(
      p + (CumSizeOf_v<II, T> * stride_) +
      (sizeof(std::tuple_element_t<II, T>) * i_))...);
  }
  template <size_t I> void assign(const T &x) {
    char *p = std::assume_aligned<16>(ptr_);
    *reinterpret_cast<std::tuple_element_t<I, T> *>(
      p + (CumSizeOf_v<I, T> * stride_) +
      (sizeof(std::tuple_element_t<I, T>) * i_)) = x.template get<I>();
  }
  constexpr SOAReference(char *p, ptrdiff_t s, ptrdiff_t idx)
    : ptr_{p}, stride_{s}, i_{idx} {}
  constexpr SOAReference(const SOAReference &) = default;
  auto operator=(const T &x) -> SOAReference & {
    ((void)assign<II>(x), ...);
    // assign<II...>(x);
    return *this;
  }
  auto operator=(SOAReference x) -> SOAReference & {
    (*this) = T(x);
    return *this;
  }
  template <size_t I> auto get() -> std::tuple_element_t<I, T> & {
    invariant(i_ >= 0);
    return *reinterpret_cast<std::tuple_element_t<I, T> *>(
      ptr_ + (CumSizeOf_v<I, T> * stride_) +
      (sizeof(std::tuple_element_t<I, T>) * i_));
  }
  template <size_t I> auto get() const -> const std::tuple_element_t<I, T> & {
    invariant(i_ >= 0);
    return *reinterpret_cast<const std::tuple_element_t<I, T> *>(
      ptr_ + (CumSizeOf_v<I, T> * stride_) +
      (sizeof(std::tuple_element_t<I, T>) * i_));
  }
};
// std::conditional_t<MatrixDimension<S>, CapacityCalculators::Length,
//                                CapacityCalculators::NextPow2>
/// requires 16 byte alignment of allocated pointer
template <typename T, Dimension S = Length<>,
          typename C = CapacityCalculators::Explicit,
          typename TT = TupleTypes_t<T>,
          typename II = std::make_index_sequence<std::tuple_size_v<T>>>
struct SOA {
  static_assert(false, "Requires `T` to be constructible from members.");
};
template <typename T, typename S, typename C, typename... Elts, size_t... II>
requires(CanConstructFromMembers<T>::value)
struct SOA<T, S, C, Types<Elts...>, std::index_sequence<II...>> {
  char *data_;
  [[no_unique_address]] S sz_;
  [[no_unique_address]] C capacity_;
  static constexpr bool trivial =
    std::is_trivially_default_constructible_v<T> &&
    std::is_trivially_destructible_v<T>;
  static_assert(trivial);
  using value_type = T;
  using reference_type = SOAReference<T, II...>;
  auto operator[](ptrdiff_t i) const -> T {
    char *p = std::assume_aligned<16>(data_);
    ptrdiff_t stride = capacity_(sz_);
    return T(*reinterpret_cast<std::tuple_element_t<II, T> *>(
      p + (CumSizeOf_v<II, T> * stride) +
      (sizeof(std::tuple_element_t<II, T>) * i))...);
  }
  auto operator[](ptrdiff_t i) -> reference_type {
    return {data_, capacity_(sz_), i};
  }
  [[nodiscard]] constexpr auto size() const -> ptrdiff_t {
    return ptrdiff_t(sz_);
  }
  template <size_t I> auto get(ptrdiff_t i) -> std::tuple_element_t<I, T> & {
    invariant(i >= 0);
    invariant(i < size());
    return *reinterpret_cast<std::tuple_element_t<I, T> *>(
      data_ + (CumSizeOf_v<I, T> * capacity_(sz_)) +
      (sizeof(std::tuple_element_t<I, T>) * i));
  }
  template <size_t I>
  auto get(ptrdiff_t i) const -> const std::tuple_element_t<I, T> & {
    invariant(i >= 0);
    invariant(i < size());
    return *reinterpret_cast<const std::tuple_element_t<I, T> *>(
      data_ + (CumSizeOf_v<I, T> * capacity_(sz_)) +
      (sizeof(std::tuple_element_t<I, T>) * i));
  }
  template <size_t I>
  auto get() -> math::MutArray<std::tuple_element_t<I, T>, S> {
    return {reinterpret_cast<std::tuple_element_t<I, T> *>(
              data_ + (CumSizeOf_v<I, T> * capacity_(sz_))),
            sz_};
  }
  template <size_t I>
  auto get() const -> math::Array<std::tuple_element_t<I, T>, S> {
    return {reinterpret_cast<const std::tuple_element_t<I, T> *>(
              data_ + (CumSizeOf_v<I, T> * capacity_(sz_))),
            sz_};
  }
  template <bool Const> struct Iterator {
    using RefType = std::conditional_t<Const, const SOA &, SOA &>;
    RefType soa_;
    ptrdiff_t i_;

    constexpr auto operator*() { return soa_[i_]; }
    constexpr auto operator->() { return &soa_[i_]; }
    constexpr auto operator++() -> Iterator {
      ++i_;
      return *this;
    }
    constexpr auto operator--() -> Iterator {
      --i_;
      return *this;
    }
    constexpr auto operator++(int) -> Iterator {
      Iterator it = *this;
      ++i_;
      return it;
    }
    constexpr auto operator--(int) -> Iterator {
      Iterator it = *this;
      --i_;
      return it;
    }

  private:
    friend constexpr auto operator+(Iterator x, ptrdiff_t y) -> Iterator {
      x.i_ += y;
      return x;
    }
    friend constexpr auto operator+(ptrdiff_t y, Iterator x) -> Iterator {
      x.i_ += y;
      return x;
    }
    friend constexpr auto operator-(Iterator x, ptrdiff_t y) -> Iterator {
      x.i_ -= y;
      return x;
    }
    friend constexpr auto operator==(Iterator x, Iterator y) -> bool {
      return x.i_ == y.i_;
    }
    friend constexpr auto operator<=>(Iterator x,
                                      Iterator y) -> std::strong_ordering {
      return x.i_ <=> y.i_;
    }
  };
  constexpr auto begin() -> Iterator<false> { return {*this, 0z}; }
  constexpr auto end() -> Iterator<false> { return {*this, ptrdiff_t(sz_)}; }
  constexpr auto begin() const -> Iterator<true> { return {*this, 0z}; }
  constexpr auto end() const -> Iterator<true> {
    return {*this, ptrdiff_t(sz_)};
  }

protected:
  static constexpr auto totalSizePer() -> size_t {
    return CumSizeOf_v<sizeof...(II), T>;
  }
  void destroy(ptrdiff_t i) {
    char *p = std::assume_aligned<16>(data_);
    ptrdiff_t stride = capacity_(sz_);
    (std::destroy_at(reinterpret_cast<std::tuple_element_t<II, T> *>(
       p + (CumSizeOf_v<II, T> * stride) +
       (sizeof(std::tuple_element_t<II, T>) * i))),
     ...);
  }
};

template <typename T, typename S = Length<>,
          typename C = CapacityCalculators::Explicit,
          typename TT = TupleTypes_t<T>,
          typename II = std::make_index_sequence<std::tuple_size_v<T>>,
          class A = alloc::Mallocator<char>>
struct ManagedSOA : public SOA<T, S, C, TT, II> {
  using Base = SOA<T, S, C, TT, II>;
  /// uninitialized allocation
  constexpr ManagedSOA() : Base{nullptr, S{}, C{}} {}
  ManagedSOA(S nsz) {
    this->sz_ = nsz;
    this->capacity_ = C{ptrdiff_t(nsz)};
    ptrdiff_t stride = this->capacity_(ptrdiff_t(this->sz_));
    this->data_ = nsz ? alloc(stride) : nullptr;
  }
  constexpr ManagedSOA(std::type_identity<T>) : ManagedSOA() {}
  ManagedSOA(std::type_identity<T>, S nsz) : ManagedSOA(nsz) {}
  ManagedSOA(const ManagedSOA &other) {
    this->sz_ = other.sz_;
    this->capacity_ = C{this->sz_};
    if (ptrdiff_t L = ptrdiff_t(this->sz_)) {
      this->data_ = alloc(this->capacity_(L));
      for (ptrdiff_t i = 0z; i < L; ++i) (*this)[i] = other[i];
    } else this->data_ = nullptr;
  }

  static auto alloc(ptrdiff_t stride) -> char * {
    return A::allocate(stride * Base::totalSizePer());
  }
  ~ManagedSOA() { free(this->capacity_(this->sz_)); }
  constexpr ManagedSOA(ManagedSOA &&other)
    : SOA<T, S, C, TT, II>{other.data_, other.sz_, other.capacity_} {
    other.data_ = nullptr;
    other.sz_ = {};
    other.capacity_ = {};
  }
  constexpr auto operator=(ManagedSOA &&other) -> ManagedSOA & {
    std::swap(this->data_, other.data_);
    std::swap(this->sz_, other.sz_);
    std::swap(this->capacity_, other.capacity_);
    return *this;
  }
  constexpr auto operator=(const ManagedSOA &other) -> ManagedSOA & {
    if (this == &other) return *this;
    auto L = static_cast<ptrdiff_t>(other.sz_);
    resizeForOverwrite(other.sz_);
    ManagedSOA &self{*this};
    for (ptrdiff_t i = 0z; i < L; ++i) self[i] = other[i];
    return *this;
  }
  void resizeForOverwrite(S nsz) {
    auto [ocap, ncap] = this->capacity_.oldnewcap(this->sz_, ptrdiff_t(nsz));
    this->sz_ = nsz;
    if (ocap >= ncap) return;
    free(ocap);
    this->data_ = A::allocate(ncap * this->totalSizePer());
    this->capacity_ = ncap;
  }
  void resize(S nsz) {
    auto [ocap, ncap] =
      this->capacity_.oldnewcap(ptrdiff_t(this->sz_), ptrdiff_t(nsz));
    if (ocap >= ncap) {
      this->sz_ = nsz;
      return;
    }
    ManagedSOA other{nsz};
    ManagedSOA &self{*this};
    std::swap(self.data_, other.data_);
    std::swap(self.sz_, other.sz_);
    std::swap(self.capacity_, other.capacity_);
    // FIXME: only accepts non-copyable
    for (ptrdiff_t i = 0,
                   L = std::min(ptrdiff_t(self.sz_), ptrdiff_t(other.sz_));
         i < L; ++i)
      self[i] = other[i];
  }
  void resize(ptrdiff_t nsz)
  requires(std::same_as<S, Length<>>)
  {
    resize(length(nsz));
  }
  constexpr void clear() { this->sz_ = {}; }
  template <typename... Args> void emplace_back(Args &&...args) {
    push_back(T(args...));
  }
  void push_back(T arg)
  requires(std::same_as<S, Length<>>)
  {
    S osz = this->sz_;
    auto i = ptrdiff_t(osz);
    resize(i + 1z);
    (*this)[i] = arg;
  }
  void erase(ptrdiff_t pos)
  requires(std::same_as<S, Length<>> &&
           std::same_as<C, CapacityCalculators::Explicit>)
  {
    invariant(pos >= 0);
    ptrdiff_t N = (this->size()) - 1;
    invariant(pos <= N);
    auto &self = *this;
    for (ptrdiff_t i = pos; i < N;) {
      ptrdiff_t j = i++;
      self[j] = self[i];
    }
    if constexpr (!(std::is_trivially_default_constructible_v<T> &&
                    std::is_trivially_destructible_v<T>)) {
      this->destroy(N);
    }
    resize(N);
  }

private:
  void free(ptrdiff_t stride) {
    if (this->data_) A::deallocate(this->data_, stride * Base::totalSizePer());
  }
};
template <typename T, typename S>
ManagedSOA(std::type_identity<T>, S) -> ManagedSOA<T, S>;

} // namespace math

template <math::ConstructibleFromMembers T, size_t... II>
struct std::tuple_size<math::SOAReference<T, II...>>
  : public std::tuple_size<T> {};

template <size_t I, math::ConstructibleFromMembers T, size_t... II>
struct std::tuple_element<I, math::SOAReference<T, II...>> {
  using type = typename std::tuple_element<I, T>::type;
};
