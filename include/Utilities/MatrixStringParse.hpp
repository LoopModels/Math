#pragma once

#include "Math/Array.hpp"
#include "Math/MatrixDimensions.hpp"
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

[[nodiscard]] constexpr auto operator"" _mat(const char *s, size_t)
  -> math::DenseMatrix<int64_t, 0> {
  invariant(s[0] == '[');
  math::ManagedArray<int64_t, unsigned, 0> content;
  ptrdiff_t cur = 1;
  ptrdiff_t numRows = 1;
  while (s[cur] != ']') {
    switch (s[cur]) {
    case ';': ++numRows; [[fallthrough]];
    case ' ': ++cur; break;
    default: content.push_back(cstoll(s, cur));
    }
  }
  ptrdiff_t numCols = content.size() / numRows;
  if (content.size() % numRows != 0) __builtin_trap();
  math::DenseMatrix<int64_t, 0> A(
    std::move(content),
    math::DenseDims{math::Row{numRows}, math::Col{numCols}});
  return A;
}

} // namespace poly::utils
