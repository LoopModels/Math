#pragma once
#ifndef Pair_hpp_INCLUDED
#define Pair_hpp_INCLUDED
#include <type_traits>

namespace poly::containers {

template <typename To, typename From>
concept ConvertibleFrom = std::is_convertible_v<From, To>;
    
template <class F, class S> struct Pair {
  [[no_unique_address]] F first;
  [[no_unique_address]] S second;
  template <ConvertibleFrom<F> A, ConvertibleFrom<S> B> constexpr operator Pair<A, B>() {
    return {A{first}, B{second}};
  }
};
} // namespace poly::containers

#endif // Pair_hpp_INCLUDED
