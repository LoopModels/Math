#ifdef USE_MODULE
module;
#else
#pragma once
#endif
#ifndef USE_MODULE
#include <cstddef>
#include <ostream>
#include <type_traits>
#include <utility>

#else
export module Pair;
import STL;
#endif

#ifdef USE_MODULE
export namespace containers {
#else
namespace containers {
#endif

template <typename To, typename From>
concept ConvertibleFrom = std::is_convertible_v<From, To>;

template <class F, class S> struct Pair {
  [[no_unique_address]] F first;
  [[no_unique_address]] S second;
  template <ConvertibleFrom<F> A, ConvertibleFrom<S> B>
  constexpr operator Pair<A, B>() {
    return {A{first}, B{second}};
  }
  template <size_t I> constexpr auto get() -> auto & {
    if constexpr (I == 0) return first;
    else return second;
  }
  template <size_t I> constexpr auto get() const -> const auto & {
    if constexpr (I == 0) return first;
    else return second;
  }
  // template <typename T, typename U>
  // constexpr auto operator=(Pair<T, U> x)
  //   -> Pair &requires(std::assignable_from<F, T> &&std::assignable_from<S,
  //   U>) { first = x.first; second = x.second; return *this;
  // }
private:
  friend void print_obj(std::ostream &os, const containers::Pair<F, S> &x) {
    os << "(" << x.first << ", " << x.second << ")";
  };
};
} // namespace containers

template <typename F, typename S>
struct std::tuple_size<containers::Pair<F, S>>
  : public std::integral_constant<size_t, 2> {};

template <typename F, typename S>
struct std::tuple_element<0, containers::Pair<F, S>> {
  using type = F;
};
template <typename F, typename S>
struct std::tuple_element<1, containers::Pair<F, S>> {
  using type = S;
};
