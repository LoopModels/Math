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

// returns an array [nrows, ncols]
template <String S> consteval auto dims_eltype() -> std::array<ptrdiff_t, 2> {
  std::vector<int64_t> content;
  ptrdiff_t cur = 1, numRows = 1;
  const char *s = S.data;
  while (s[cur] != ']') {
    switch (s[cur]) {
    case ';': ++numRows; [[fallthrough]];
    case ' ': ++cur; break;
    default: content.push_back(cstoll(s, cur));
    }
  }
  ptrdiff_t numCols = ptrdiff_t(content.size()) / numRows;
  if (content.size() % numRows != 0) __builtin_trap();
  return {numRows, numCols};
}

template <String S> constexpr auto matrix_from_string() {
  constexpr std::array<ptrdiff_t, 2> dims = dims_eltype<S>();
  math::StaticArray<int64_t, dims[0], dims[1]> A;
  ptrdiff_t cur = 1, i = 0;
  const char *s = S.data;
  while (s[cur] != ']') {
    switch (s[cur]) {
    case ';': [[fallthrough]];
    case ' ': ++cur; break;
    default: A.data()[i++] = cstoll(s, cur);
    }
  }
  return A;
}

template <String S> [[nodiscard]] constexpr auto operator"" _mat() {
  return matrix_from_string<S>();
}

} // namespace poly::utils
