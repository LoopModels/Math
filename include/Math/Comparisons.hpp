#pragma once
#include <algorithm>

namespace poly::math {
constexpr auto allZero(const auto &x) -> bool {
  return std::all_of(x.begin(), x.end(), [](auto a) { return a == 0; });
  // return std::ranges::all_of(x, [](auto x) { return x == 0; });
}
constexpr auto allGEZero(const auto &x) -> bool {
  return std::all_of(x.begin(), x.end(), [](auto a) { return a >= 0; });
  // return std::ranges::all_of(x, [](auto x) { return x >= 0; });
}
constexpr auto allLEZero(const auto &x) -> bool {
  return std::all_of(x.begin(), x.end(), [](auto a) { return a <= 0; });
  // return std::ranges::all_of(x, [](auto x) { return x <= 0; });
}

constexpr auto anyNEZero(const auto &x) -> bool {
  return std::any_of(x.begin(), x.end(), [](int64_t y) { return y != 0; });
}

} // namespace poly::math
