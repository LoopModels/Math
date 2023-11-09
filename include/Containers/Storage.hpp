#pragma once
#include <cstddef>
#include <cstdint>
#include <bit>

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

template <class T, class S> consteval auto PreAllocStorage() -> ptrdiff_t {
  constexpr ptrdiff_t totalBytes = 128;
  constexpr ptrdiff_t remainingBytes =
    totalBytes - sizeof(T *) - sizeof(S) - sizeof(default_capacity_type_t<S>);
  constexpr ptrdiff_t N = remainingBytes / sizeof(T);
  return std::max<ptrdiff_t>(0, N);
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
template <class T, class S>
consteval auto PreAllocSquareStorage() -> ptrdiff_t {
  // 2* because we want to allow more space for matrices
  // also removes need for other checks; log2Floor(2)==1
  constexpr uint64_t N = 2 * PreAllocStorage<T, S>();
  if (!N) return 0;
  // a fairly naive algorirthm for computing the next square `N`
  // sqrt(x) = x^(1/2) = exp2(log2(x)/2)
  constexpr uint64_t L = 1 << (log2Floor(N) / 2);
  constexpr uint64_t H = 1 << ((log2Ceil(N) + 1) / 2);
  return ptrdiff_t(bisectFindSquare(L, H, N));
}

} // namespace poly::containers
