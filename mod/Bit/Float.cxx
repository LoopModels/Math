#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifndef USE_MODULE
#include <bit>
#include <concepts>
#include <cstdint>

#else
export module BitHack;
import STL;
#endif

#ifdef USE_MODULE
export namespace bit {
#else
namespace bit {
#endif

constexpr auto exp2unchecked(std::integral auto x) {
  return std::bit_cast<double>(static_cast<int64_t>(1023 + x) << 52);
}

constexpr auto next_pow2(double x) -> double {
  static constexpr int64_t mantissa_mask = ~((int64_t(1) << 52) - 1);
  int64_t i = std::bit_cast<int64_t>(x), j = i & mantissa_mask;
  return j != i ? std::bit_cast<double>(j + (int64_t(1) << 52)) : x;
}

} // namespace bit
