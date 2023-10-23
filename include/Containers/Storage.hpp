#pragma once
#include <cstddef>

namespace poly::containers {
template <typename T, ptrdiff_t N> struct Storage {
  static_assert(N > 0);
  alignas(T) char mem[N * sizeof(T)]; // NOLINT (modernize-avoid-c-style-arrays)
  constexpr auto data() -> T * {
    void *p = mem;
    return (T *)p;
  }
  constexpr auto data() const -> const T * {
    const void *p = mem;
    return (T *)p;
  }
  constexpr Storage() {} // NOLINT (modernize-use-equals-default)
};
template <typename T> struct alignas(T) Storage<T, 0> {
  static constexpr auto data() -> T * { return nullptr; }
};
} // namespace poly::containers
