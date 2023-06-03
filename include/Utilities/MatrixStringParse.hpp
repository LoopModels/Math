#pragma once

#include "Math/Array.hpp"
#include "Math/Math.hpp"
#include "Math/MatrixDimensions.hpp"
#include <cstddef>
#include <cstdint>

constexpr auto cstoll(const char *s, size_t &cur) -> int64_t {
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
  -> LinAlg::DenseMatrix<int64_t, 0> {
  invariant(s[0] == '[');
  LinAlg::ManagedArray<int64_t, unsigned, 0> content;
  size_t cur = 1;
  size_t numRows = 1;
  while (s[cur] != ']') {
    switch (s[cur]) {
    case ';':
      ++numRows;
      [[fallthrough]];
    case ' ':
      ++cur;
      break;
    default:
      content.push_back(cstoll(s, cur));
    }
  }
  size_t numCols = content.size() / numRows;
  if (content.size() % numRows != 0) __builtin_trap();
  LinAlg::DenseMatrix<int64_t, 0> A(std::move(content),
                                    DenseDims{Row{numRows}, Col{numCols}});
  return A;
}
