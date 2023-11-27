#pragma once
#ifndef POLY_CONTAIERS_Tuple_hpp_INCLUDED
#define POLY_CONTAIERS_Tuple_hpp_INCLUDED

#include <concepts>
#include <cstddef>

namespace poly::containers {
template <typename T, typename... Ts> struct Tuple;
template <typename T, typename... Ts>
[[gnu::always_inline]] constexpr auto cattuple(T, Tuple<Ts...>)
  -> Tuple<T, Ts...>;

template <typename T, typename... Ts> struct Tuple {
  [[no_unique_address]] T head;
  [[no_unique_address]] Tuple<Ts...> tail;
  constexpr Tuple(T head_, Ts... tail_) : head(head_), tail(tail_...){};
  constexpr Tuple(T head_, Tuple<Ts...> tail_) : head(head_), tail(tail_){};

  constexpr Tuple(const Tuple &) = default;
  template <size_t I> auto get() -> auto & {
    if constexpr (I == 0) return head;
    else return tail.template get<I - 1>();
  }
  template <size_t I> [[nodiscard]] auto get() const -> const auto & {
    if constexpr (I == 0) return head;
    else return tail.template get<I - 1>();
  }
  constexpr void apply(const auto &f) {
    f(head);
    tail.apply(f);
  }
  template <typename U, typename... Us>
  constexpr void apply(const Tuple<U, Us...> &x, const auto &f)
  requires(sizeof...(Ts) == sizeof...(Us))
  {
    f(head, x.head);
    tail.apply(x.tail, f);
  }
  constexpr auto map(const auto &f) const {
    return cattuple(f(head), tail.map(f));
  }
  template <typename U, typename... Us>
  constexpr auto map(const Tuple<U, Us...> &x, const auto &f) const
  requires(sizeof...(Ts) == sizeof...(Us))
  {
    return cattuple(f(head, x.head), tail.map(x.tail, f));
  }
  template <typename U, typename... Us>
  inline constexpr void operator<<(const Tuple<U, Us...> &)
  requires(sizeof...(Ts) == sizeof...(Us));
  template <typename U, typename... Us>
  inline constexpr void operator+=(const Tuple<U, Us...> &)
  requires(sizeof...(Ts) == sizeof...(Us));
  template <typename U, typename... Us>
  inline constexpr void operator-=(const Tuple<U, Us...> &)
  requires(sizeof...(Ts) == sizeof...(Us));
  template <typename U, typename... Us>
  inline constexpr void operator*=(const Tuple<U, Us...> &)
  requires(sizeof...(Ts) == sizeof...(Us));
  template <typename U, typename... Us>
  inline constexpr void operator/=(const Tuple<U, Us...> &)
  requires(sizeof...(Ts) == sizeof...(Us));
  constexpr auto operator=(const Tuple &) -> Tuple & = default;
  template <typename U, typename... Us>
  constexpr auto operator=(Tuple<U, Us...> x) -> Tuple &requires(
    std::assignable_from<T, U> &&... &&std::assignable_from<Ts, Us>) {
    head = x.head;
    tail = x.tail;
    return *this;
  }
};
template <typename T, typename... Ts>
[[gnu::always_inline]] constexpr auto cattuple(T x, Tuple<Ts...> y)
  -> Tuple<T, Ts...> {
  return {x, y};
}
template <typename T> struct Tuple<T> {
  [[no_unique_address]] T head;
  constexpr Tuple(T head_) : head(head_){};
  constexpr Tuple(const Tuple &) = default;
  template <size_t I> auto get() -> T & {
    static_assert(I == 0);
    return head;
  }
  template <size_t I> [[nodiscard]] auto get() const -> const T & {
    static_assert(I == 0);
    return head;
  }
  constexpr auto operator=(const Tuple &) -> Tuple & = default;
  constexpr void apply(const auto &f) { f(head); }
  template <typename U> constexpr void apply(const Tuple<U> &x, const auto &f) {
    f(head, x.head);
  }
  constexpr auto map(const auto &f) const -> Tuple<decltype(f(head))> {
    return {f(head)};
  }
  template <typename U>
  constexpr auto map(const Tuple<U> &x, const auto &f) const
    -> Tuple<decltype(f(head, x.head))> {
    return {f(head, x.head)};
  }
  template <typename U> inline constexpr void operator<<(const Tuple<U> &);
  template <typename U> inline constexpr void operator+=(const Tuple<U> &);
  template <typename U> inline constexpr void operator-=(const Tuple<U> &);
  template <typename U> inline constexpr void operator*=(const Tuple<U> &);
  template <typename U> inline constexpr void operator/=(const Tuple<U> &);

  template <typename U>
  constexpr auto operator=(Tuple<U> x)
    -> Tuple &requires((!std::same_as<T, U>)&&std::assignable_from<T, U>) {
      head = x.head;
      return *this;
    }
};

template <typename... Ts> Tuple(Ts...) -> Tuple<Ts...>;

template <typename... Ts> constexpr auto tie(Ts &...x) -> Tuple<Ts &...> {
  return {x...};
}
} // namespace poly::containers

template <typename T, typename... Ts>
struct std::tuple_size<poly::containers::Tuple<T, Ts...>>
  : public std::integral_constant<size_t, 1 + sizeof...(Ts)> {};

template <typename T, typename... Ts>
struct std::tuple_element<0, poly::containers::Tuple<T, Ts...>> {
  using type = T;
};
template <size_t I, typename T, typename... Ts>
struct std::tuple_element<I, poly::containers::Tuple<T, Ts...>> {
  using type =
    typename std::tuple_element<I - 1, poly::containers::Tuple<Ts...>>::type;
};

#endif // Tuple_hpp_INCLUDED
