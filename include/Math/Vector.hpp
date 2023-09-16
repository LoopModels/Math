#pragma once
#include "Utilities/TypePromotion.hpp"
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace poly::math {

template <typename T, typename S = utils::eltype_t<T>>
concept LinearlyIndexable = requires(T t, ptrdiff_t i) {
  { t[i] } -> std::convertible_to<S>;
};
template <typename T, typename S>
concept LinearlyIndexableOrConvertible =
  LinearlyIndexable<T, S> || std::convertible_to<T, S>;

template <typename T>
concept AbstractVector =
  utils::HasEltype<T> && LinearlyIndexable<T> && requires(T t) {
    { t.size() } -> std::convertible_to<ptrdiff_t>;
    { t.view() };
  };

// This didn't work: #include "Math/Vector.hpp" NOLINT(unused-includes)
// so I moved some code from "Math/Array.hpp" here instead.
template <class T>
concept SizeMultiple8 = (sizeof(T) % 8) == 0;

template <class S> struct DefaultCapacityType {
  using type = unsigned int;
};
template <SizeMultiple8 S> struct DefaultCapacityType<S> {
  using type = std::size_t;
};
template <class S>
using default_capacity_type_t = typename DefaultCapacityType<S>::type;

static_assert(!SizeMultiple8<uint32_t>);
static_assert(SizeMultiple8<uint64_t>);
static_assert(std::is_same_v<default_capacity_type_t<uint32_t>, uint32_t>);
static_assert(std::is_same_v<default_capacity_type_t<uint64_t>, uint64_t>);

template <class T> consteval auto PreAllocStorage() -> size_t {
  constexpr ptrdiff_t totalBytes = 128;
  constexpr ptrdiff_t remainingBytes =
    totalBytes - sizeof(T *) - 2 * sizeof(unsigned);
  constexpr ptrdiff_t N = remainingBytes / sizeof(T);
  return std::max<ptrdiff_t>(1, N);
}
constexpr auto log2Floor(uint64_t x) -> uint64_t {
  return 63 - std::countl_zero(x);
}
constexpr auto log2Ceil(uint64_t x) -> uint64_t {
  return 64 - std::countl_zero(x - 1);
}
// NOLINTNEXTLINE(misc-no-recursion)
consteval auto bisectFindSquare(uint64_t l, uint64_t h, uint64_t N)
  -> uint64_t {
  if (l == h) return l;
  uint64_t m = (l + h) / 2;
  if (m * m >= N) return bisectFindSquare(l, m, N);
  return bisectFindSquare(m + 1, h, N);
}
template <class T> consteval auto PreAllocSquareStorage() -> size_t {
  // 2* because we want to allow more space for matrices
  // also removes need for other checks; log2Floor(2)==1
  constexpr uint64_t N = 2 * PreAllocStorage<T>();
  // a fairly naive algorirthm for computing the next square `N`
  // sqrt(x) = x^(1/2) = exp2(log2(x)/2)
  constexpr uint64_t L = 1 << (log2Floor(N) / 2);
  constexpr uint64_t H = 1 << ((log2Ceil(N) + 1) / 2);
  return bisectFindSquare(L, H, N);
}

constexpr auto selfDot(const auto &a) {
  utils::eltype_t<decltype(a)> sum = 0;
  for (auto x : a) sum += x * x;
  return sum;
}

} // namespace poly::math
