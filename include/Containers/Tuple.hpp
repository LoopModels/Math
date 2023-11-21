#pragma once
#ifndef POLY_CONTAIERS_Tuple_hpp_INCLUDED
#define POLY_CONTAIERS_Tuple_hpp_INCLUDED

namespace poly::containers {
template <typename T, typename... Ts> struct Tuple {
  [[no_unique_address]] T head;
  [[no_unique_address]] Tuple<Ts...> tail;
  constexpr Tuple(T head_, Ts... tail_) : head(head_), tail(tail_...){};

  template <size_t I> auto get() -> auto & {
    if constexpr (I == 0) return head;
    else return tail.template get<I - 1>();
  }
  template <size_t I> [[nodiscard]] auto get() const -> const auto & {
    if constexpr (I == 0) return head;
    else return tail.template get<I - 1>();
  }
  template <typename U, typename... Us>
  constexpr auto operator=(Tuple<U, Us...> x) -> Tuple &requires(
    std::assignable_from<T, U> &&... &&std::assignable_from<Ts, Us>) {
    head = x.head;
    tail = x.tail;
    return *this;
  }
};
template <typename T> struct Tuple<T> {
  [[no_unique_address]] T head;
  constexpr Tuple(T head_) : head(head_){};
  template <size_t I> auto get() -> T & {
    static_assert(I == 0);
    return head;
  }
  template <size_t I> [[nodiscard]] auto get() const -> const T & {
    static_assert(I == 0);
    return head;
  }
  constexpr auto operator=(const Tuple &) -> Tuple & = default;
  template <typename U>
  constexpr auto operator=(Tuple<U> x)
    -> Tuple &requires(std::assignable_from<T, U>) {
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
