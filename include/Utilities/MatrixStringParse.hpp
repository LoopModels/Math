#pragma once

#include "Math/Array.hpp"
#include "Math/MatrixDimensions.hpp"
#include "Math/StaticArrays.hpp"
#include <cstddef>
#include <cstdint>

namespace poly::utils {

constexpr auto cstoll(const char *s, ptrdiff_t &cur) -> int64_t {
  int64_t res = 0;
  bool neg = false;
  while (s[cur] == ' ') ++cur;
  if (s[cur] == '-') {
    neg = true;
    ++cur;
  }
  while (s[cur] >= '0' && s[cur] <= '9') {
    res = res * 10 + (s[cur] - '0');
    ++cur;
  }
  return neg ? -res : res;
}

// static_assert(__cpp_nontype_template_args >= 201911);
template <std::size_t N> struct String {
  char data[N];

  static constexpr auto size() -> size_t { return N; }
  constexpr String(char const (&p)[N]) { std::copy_n(p, N, data); }
  constexpr auto operator[](ptrdiff_t i) const -> char { return data[i]; }
};

// returns an array {nrows, ncols}
#if defined(__clang__)
template <String S> constexpr auto dims_eltype() -> std::array<ptrdiff_t, 2> {
#else
template <String S> consteval auto dims_eltype() -> std::array<ptrdiff_t, 2> {
#endif
  ptrdiff_t numRows = 1, numCols = 0;
  // count numCols
  const char *s = S.data + 1; // skip `[`
  // while (*s != ';'){
  while (true) {
    char c = *s;
    while (c == ' ') c = *(++s);
    if (c == ';') break;
    if (c == ']') return {numRows, numCols};
    while (true) {
      c = *(++s);
      if (c == '-') continue;
      if (c >= '0' && c <= '9') continue;
      break;
    }
    ++numCols;
  }
  ++numRows;
  while (true) {
    char c = *(++s);
    if (c == ']') return {numRows, numCols};
    if (c == ';') ++numRows;
  }
}

#if defined(__clang__)
template <String S> constexpr auto matrix_from_string() {
#else
template <String S> consteval auto matrix_from_string() {
#endif
  constexpr std::array<ptrdiff_t, 2> dims = dims_eltype<S>();
  math::StaticArray<int64_t, dims[0], dims[1]> A(int64_t(0));
  ptrdiff_t cur = 1, i = 0, j = 0;
  const char *s = S.data;
  while (s[cur] != ']') {
    switch (s[cur]) {
    case ';': [[fallthrough]];
    case ' ': ++cur; break;
    default: {
      int64_t x = cstoll(s, cur);
      A.set(x, i, j);
      if (++j == dims[1]) {
        j = 0;
        ++i;
      }
    }
    }
  }
  return A;
}

#if defined(__clang__)
template <String S> [[nodiscard]] constexpr auto operator"" _mat() {
#else
template <String S> [[nodiscard]] consteval auto operator"" _mat() {
#endif
  return matrix_from_string<S>();
}

} // namespace poly::utils
