#pragma once
#include "Math/MatrixDimensions.hpp"
#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace poly::containers {
template <typename T, ptrdiff_t N> struct Storage {
  static_assert(N > 0);
  alignas(T) char mem[N * sizeof(T)]; // NOLINT (modernize-avoid-c-style-arrays)
  constexpr auto data() -> T * { return reinterpret_cast<T *>(mem); }
  constexpr auto data() const -> const T * {
    return reinterpret_cast<const T *>(mem);
  }
  constexpr Storage() {} // NOLINT (modernize-use-equals-default)
};
template <typename T> struct alignas(T) Storage<T, 0> {
  static constexpr auto data() -> T * { return nullptr; }
};

template <class T>
concept SizeMultiple8 = (sizeof(T) % 8) == 0;

template <class S> struct DefaultCapacityType {
  using type = int;
};
template <SizeMultiple8 S> struct DefaultCapacityType<S> {
  using type = std::ptrdiff_t;
};
template <class S>
using default_capacity_type_t = typename DefaultCapacityType<S>::type;

static_assert(!SizeMultiple8<uint32_t>);
static_assert(SizeMultiple8<uint64_t>);
static_assert(std::is_same_v<default_capacity_type_t<uint32_t>, int32_t>);
static_assert(std::is_same_v<default_capacity_type_t<uint64_t>, int64_t>);

consteval auto log2Floor(uint64_t x) -> uint64_t {
  return 63 - std::countl_zero(x);
}
consteval auto log2Ceil(uint64_t x) -> uint64_t {
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
template <class T, class S> consteval auto PreAllocStorage() -> ptrdiff_t {
  constexpr ptrdiff_t totalBytes = 128;
  // constexpr ptrdiff_t remainingBytes =
  //   totalBytes - sizeof(T *) - sizeof(S) -
  //   sizeof(default_capacity_type_t<S>);
  // constexpr ptrdiff_t N = remainingBytes / ptrdiff_t(sizeof(T));
  constexpr ptrdiff_t N = totalBytes / ptrdiff_t(sizeof(T));
  static_assert(N <= 128);
  if constexpr (N <= 0) return 0;
  // else if constexpr (!math::MatrixDimension<S>) return N;
  else if constexpr (!std::convertible_to<S, math::SquareDims<>>) return N;
  else {
    constexpr auto UN = uint64_t(N);
    // a fairly naive algorirthm for computing the next square `N`
    // sqrt(x) = x^(1/2) = exp2(log2(x)/2)
    constexpr uint64_t R = log2Floor(UN) / 2;
    static_assert(R < 63);
    constexpr uint64_t L = uint64_t(1) << R;
    constexpr uint64_t H = uint64_t(1) << ((log2Ceil(N) + 1) / 2);
    return ptrdiff_t(bisectFindSquare(L, H, UN));
  }
}
} // namespace poly::containers
