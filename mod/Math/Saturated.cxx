#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifndef USE_MODULE
#include <concepts>
#include <limits>

#include "Utilities/Widen.cxx"
#else
export module Saturated;

import STL;
import Widen;
#endif

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif
template <std::unsigned_integral T> constexpr auto add_sat(T x, T y) -> T {
  T res = x + y;
  return res | -(res < x);
}
template <std::unsigned_integral T> constexpr auto sub_sat(T x, T y) -> T {
  T res = x - y;
  return res & -(res <= x);
}

template <std::unsigned_integral T> constexpr auto mul_sat(T x, T y) -> T {
  auto prod = utils::widen(x) * utils::widen(y);
  T hi = prod >> (8 * sizeof(T));
  T lo = prod;
  return lo | -!!hi;
}

template <std::signed_integral T> constexpr auto add_sat(T x, T y) -> T {
  T res;
  // workaround for modules
#ifndef USE_MODULE
  if (!__builtin_add_overflow(x, y, &res)) return res;
#else
  if constexpr (sizeof(T) < sizeof(int)) {
    int sx = x, sy = y, sres = sx + sy;
    if ((sres >= std::numeric_limits<T>::min()) &&
        (sres <= std::numeric_limits<T>::max()))
      return res = sres;
  } else if constexpr (std::same_as<T, int>) {
    if (!__builtin_sadd_overflow(x, y, &res)) return res;
  } else if constexpr (std::same_as<T, long>) {
    if (!__builtin_saddl_overflow(x, y, &res)) return res;
  } else {
    if (!__builtin_saddll_overflow(x, y, &res)) return res;
  }
#endif
  return x > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
}
template <std::signed_integral T> constexpr auto sub_sat(T x, T y) -> T {
  T res;
#ifndef USE_MODULE
  if (!__builtin_sub_overflow(x, y, &res)) return res;
#else
  if constexpr (sizeof(T) < sizeof(int)) {
    int sx = x, sy = y, sres = sx - sy;
    if ((sres >= std::numeric_limits<T>::min()) &&
        (sres <= std::numeric_limits<T>::max()))
      return res = sres;
  } else if constexpr (std::same_as<T, int>) {
    if (!__builtin_ssub_overflow(x, y, &res)) return res;
  } else if constexpr (std::same_as<T, long>) {
    if (!__builtin_ssubl_overflow(x, y, &res)) return res;
  } else {
    if (!__builtin_ssubll_overflow(x, y, &res)) return res;
  }
#endif
  return x > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
}
template <std::signed_integral T> constexpr auto mul_sat(T x, T y) -> T {
  T res;
#ifndef USE_MODULE
  if (!__builtin_mul_overflow(x, y, &res)) return res;
#else
  if constexpr (sizeof(T) < sizeof(int)) {
    int sx = x, sy = y, sres = sx * sy;
    if ((sres >= std::numeric_limits<T>::min()) &&
        (sres <= std::numeric_limits<T>::max()))
      return res = sres;
  } else if constexpr (std::same_as<T, int>) {
    if (!__builtin_smul_overflow(x, y, &res)) return res;
  } else if constexpr (std::same_as<T, long>) {
    if (!__builtin_smull_overflow(x, y, &res)) return res;
  } else {
    if (!__builtin_smulll_overflow(x, y, &res)) return res;
  }
#endif
  return ((x > 0) == (y > 0)) ? std::numeric_limits<T>::max()
                              : std::numeric_limits<T>::min();
}
} // namespace math
