#ifdef USE_MODULE
module;
#else
#pragma once
#endif
#ifndef USE_MODULE
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "Containers/Pair.cxx"
#else
export module Tuple;

import Pair;
import STL;
#endif

#ifdef USE_MODULE
export namespace containers {
#else
namespace containers {
#endif

template <typename T, typename... Ts> struct Tuple;
template <typename T, typename... Ts>
[[gnu::always_inline]] constexpr auto cattuple(T,
                                               Tuple<Ts...>) -> Tuple<T, Ts...>;

template <typename T, typename... Ts, typename U, typename... Us>
constexpr void copyFrom(Tuple<T, Ts...> &dst, const Tuple<U, Us...> &src)
requires(sizeof...(Ts) == sizeof...(Us) && std::assignable_from<T, U> &&
         (... && std::assignable_from<Ts, Us>))
{
  dst = src;
}

template <typename T, typename... Ts> struct Tuple {
  [[no_unique_address]] T head_;
  [[no_unique_address]] Tuple<Ts...> tail_;
  constexpr Tuple() = default;
  // template <std::convertible_to<T> U, std::convertible_to<Ts>... Us>
  // constexpr Tuple(U head, Us... tail)
  //   : head_(std::forward<U>(head)), tail_(std::forward<Us>(tail)...){};
  constexpr Tuple(T head, Ts... tail) : head_(head), tail_(tail...) {};
  constexpr Tuple(T head, Tuple<Ts...> tail) : head_(head), tail_(tail) {};

  constexpr Tuple(const Tuple &) = default;
  template <size_t I> [[gnu::always_inline]] constexpr auto get() -> auto & {
    if constexpr (I == 0) return head_;
    else if constexpr (I == 1) return tail_.head_;
    else if constexpr (I == 2) return tail_.tail_.head_;
    else return tail_.tail_.tail_.template get<I - 3>();
  }
  template <size_t I>
  [[nodiscard, gnu::always_inline]] constexpr auto get() const -> const auto & {
    if constexpr (I == 0) return head_;
    else if constexpr (I == 1) return tail_.head_;
    else if constexpr (I == 2) return tail_.tail_.head_;
    else return tail_.tail_.tail_.template get<I - 3>();
  }
  constexpr void apply(const auto &f) {
    f(head_);
    tail_.apply(f);
  }
  template <typename U, typename... Us>
  constexpr void apply(const Tuple<U, Us...> &x, const auto &f)
  requires(sizeof...(Ts) == sizeof...(Us))
  {
    f(head_, x.head_);
    tail_.apply(x.tail_, f);
  }
  constexpr auto mutmap(const auto &f) {
    return cattuple(f(head_), tail_.mutmap(f));
  }
  constexpr auto map(const auto &f) const {
    return cattuple(f(head_), tail_.map(f));
  }
  template <typename U, typename... Us>
  constexpr auto map(const Tuple<U, Us...> &x, const auto &f) const
  requires(sizeof...(Ts) == sizeof...(Us))
  {
    return cattuple(f(head_, x.head_), tail_.map(x.tail_, f));
  }
  template <typename U, typename... Us>
  [[gnu::always_inline, gnu::flatten]] constexpr void
  operator+=(const Tuple<U, Us...> &src)
  requires(sizeof...(Ts) == sizeof...(Us))
  {
    (*this) << map(src, [](const auto &d, const auto &s) { return d + s; });
  }
  template <typename U, typename... Us>
  [[gnu::always_inline, gnu::flatten]] constexpr void
  operator-=(const Tuple<U, Us...> &src)
  requires(sizeof...(Ts) == sizeof...(Us))
  {
    (*this) << map(src, [](const auto &d, const auto &s) { return d - s; });
  }
  template <typename U, typename... Us>
  [[gnu::always_inline, gnu::flatten]] constexpr void
  operator*=(const Tuple<U, Us...> &src)
  requires(sizeof...(Ts) == sizeof...(Us))
  {
    (*this) << map(src, [](const auto &d, const auto &s) { return d * s; });
  }
  template <typename U, typename... Us>
  [[gnu::always_inline, gnu::flatten]] constexpr void
  operator/=(const Tuple<U, Us...> &src)
  requires(sizeof...(Ts) == sizeof...(Us))
  {
    (*this) << map(src, [](const auto &d, const auto &s) { return d / s; });
  }
  constexpr auto operator=(const Tuple &) -> Tuple & = default;
  constexpr auto operator=(Tuple &&) -> Tuple & = default;
  template <typename U, typename... Us>
  constexpr auto operator=(Tuple<U, Us...> x)
    -> Tuple &requires(
      std::is_assignable_v<T, U> &&... &&std::is_assignable_v<Ts, Us>) {
    head_ = std::move(x.head_);
    tail_ = std::move(x.tail_);
    return *this;
  }

  template <typename U, typename V>
  constexpr auto operator=(Pair<U, V> x)
    -> Tuple &requires((sizeof...(Ts) == 1) &&
                       (std::is_assignable_v<T, U> && ... &&
                        std::is_assignable_v<Ts, V>)) {
    head_ = std::move(x.first);
    tail_.head_ = std::move(x.second);
    return *this;
  }

  template <typename U, typename... Us>
  constexpr void operator<<(const Tuple<U, Us...> &src)
  requires(sizeof...(Ts) == sizeof...(Us))
  {
    copyFrom(*this, src);
  }
};
template <typename T, typename... Ts>
[[gnu::always_inline]] constexpr auto
cattuple(T x, Tuple<Ts...> y) -> Tuple<T, Ts...> {
  return {x, y};
}
template <typename T> struct Tuple<T> {
  [[no_unique_address]] T head_;
  constexpr Tuple() = default;
  constexpr Tuple(T head) : head_(head) {};
  // template <std::convertible_to<T> U>
  // constexpr Tuple(U &&head) : head_(std::forward<U>(head)){};
  constexpr Tuple(const Tuple &) = default;
  template <size_t I> constexpr auto get() -> T & {
    static_assert(I == 0);
    return head_;
  }
  template <size_t I> [[nodiscard]] constexpr auto get() const -> const T & {
    static_assert(I == 0);
    return head_;
  }
  constexpr auto operator=(const Tuple &) -> Tuple & = default;
  constexpr auto operator=(Tuple &&) -> Tuple & = default;
  constexpr void apply(const auto &f) { f(head_); }
  template <typename U> constexpr void apply(const Tuple<U> &x, const auto &f) {
    f(head_, x.head_);
  }
  constexpr auto mutmap(const auto &f) -> Tuple<decltype(f(head_))> {
    return {f(head_)};
  }
  constexpr auto map(const auto &f) const -> Tuple<decltype(f(head_))> {
    return {f(head_)};
  }
  template <typename U>
  constexpr auto map(const Tuple<U> &x, const auto &f) const
    -> Tuple<decltype(f(head_, x.head_))> {
    return {f(head_, x.head_)};
  }
  template <typename U> constexpr void operator+=(const Tuple<U> &);
  template <typename U> constexpr void operator-=(const Tuple<U> &);
  template <typename U> constexpr void operator*=(const Tuple<U> &);
  template <typename U> constexpr void operator/=(const Tuple<U> &);

template <typename U>
constexpr auto operator=(Tuple<U> x)
  -> Tuple &requires((!std::same_as<T, U>) && std::is_assignable_v<T, U>) {
  head_ = std::move(x.head_);
  return *this;
}

private : template <typename U>
          friend constexpr void operator<<(Tuple<T> &dst, const Tuple<U> &src) {
    dst << src;
  }
  template <typename U>
  friend constexpr void operator<<(Tuple<T> &&dst, const Tuple<U> &src) {
    dst << src;
  }
};

template <typename... Ts> Tuple(Ts...) -> Tuple<Ts...>;

template <typename T> struct Add {
  T &x_;
  constexpr auto operator=(T y) -> Add & {
    x_ += y;
    return *this;
  }
};
template <typename T> struct And {
  T &x_;
  constexpr auto operator=(T y) -> And & {
    x_ &= y;
    return *this;
  }
};
template <typename T> struct Or {
  T &x_;
  constexpr auto operator=(T y) -> Or & {
    x_ |= y;
    return *this;
  }
};
template <typename T> struct Max {
  T &x_;
  constexpr auto operator=(T y) -> Max & {
    x_ = x_ >= y ? x_ : y;
    // x_ = std::max(x_, y);
    return *this;
  }
};
template <typename T> struct Min {
  T &x_;
  constexpr auto operator=(T y) -> Min & {
    x_ = x_ >= y ? y : x_;
    // x_ = std::min(x_, y);
    return *this;
  }
};
inline constexpr struct IgnoreImpl {
  // NOLINTNEXTLINE
  constexpr auto operator=(const auto &) const -> const IgnoreImpl & {
    return *this;
  }
} Ignore;

template <typename... Ts> constexpr auto tie(Ts &&...x) -> Tuple<Ts...> {
  return {x...};
}

// template <typename F, typename S>
// using Pair = Tuple<F,S>;

} // namespace containers

template <typename T, typename... Ts>
struct std::tuple_size<containers::Tuple<T, Ts...>>
  : public std::integral_constant<size_t, 1 + sizeof...(Ts)> {};

template <typename T, typename... Ts>
struct std::tuple_element<0, containers::Tuple<T, Ts...>> {
  using type = T;
};
template <size_t I, typename T, typename... Ts>
struct std::tuple_element<I, containers::Tuple<T, Ts...>> {
  using type =
    typename std::tuple_element<I - 1, containers::Tuple<Ts...>>::type;
};
