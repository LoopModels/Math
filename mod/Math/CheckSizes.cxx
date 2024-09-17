#ifdef USE_MODULE
module;
#else
#pragma once
#endif
#ifndef USE_MODULE
#include <concepts>
#include <cstddef>
#include <type_traits>

#include "Utilities/Invariant.cxx"
#include "Utilities/Parameters.cxx"
#else
export module CheckSizes;

import Invariant;
import Param;
import STL;
#endif

template <typename T>
concept ShouldView = requires(const T &x) {
  { x.view() } -> utils::TriviallyCopyable;
};

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif
using utils::invariant;
// inputs must be `ptrdiff_t` or `std::integral_constant<ptrdiff_t,value>`
template <typename X, typename Y>
[[gnu::always_inline]] constexpr auto check_sizes(X x, Y y) {
  if constexpr (std::same_as<ptrdiff_t, X>) {
    if constexpr (std::same_as<ptrdiff_t, Y>) {
      invariant(x, y);
      return x;
    } else if constexpr (y > 1) {
      constexpr ptrdiff_t L = y;
      invariant(x, L);
      return std::integral_constant<ptrdiff_t, L>{};
    } else return x;
  } else if constexpr (x <= 1) return y;
  else if constexpr (std::same_as<ptrdiff_t, Y>) {
    constexpr ptrdiff_t L = x;
    invariant(L, y);
    return std::integral_constant<ptrdiff_t, L>{};
  } else if constexpr (y <= 1) return x;
  else {
    static_assert(x == y);
    return std::integral_constant<ptrdiff_t, ptrdiff_t(x)>{};
  }
}

template <typename T> [[gnu::always_inline]] constexpr auto view(const T &x) {
  if constexpr (ShouldView<T>) return x.view();
  else return x;
}
} // namespace math
