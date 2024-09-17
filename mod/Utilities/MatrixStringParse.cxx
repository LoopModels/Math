#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifndef USE_MODULE
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "Math/StaticArrays.cxx"
#else
export module ArrayParse;

import StaticArray;
import STL;
#endif

// #if !defined(__clang__)
#define CONSTEVAL_LITERAL_ARRAYS
// #endif
template <size_t N> struct String {
  char data_[N];

  static constexpr auto size() -> size_t { return N; }
  constexpr String(char const (&p)[N]) { std::copy_n(p, N, data_); }
  constexpr auto operator[](ptrdiff_t i) const -> char { return data_[i]; }
};

// returns an array {nrows, ncols}
template <String S> consteval auto dims_eltype() -> std::array<ptrdiff_t, 2> {
  ptrdiff_t num_rows = 1, num_cols = 0;
  // count numCols
  const char *s = S.data_ + 1; // skip `[`
  // while (*s != ';'){
  while (true) {
    char c = *s;
    while (c == ' ') c = *(++s);
    if (c == ';') break;
    if (c == ']') return {num_rows, num_cols};
    while (true) {
      c = *(++s);
      if (c == '-') continue;
      if (c >= '0' && c <= '9') continue;
      break;
    }
    ++num_cols;
  }
  ++num_rows;
  while (true) {
    char c = *(++s);
    if (c == ']') return {num_rows, num_cols};
    if (c == ';') ++num_rows;
  }
}

#ifndef CONSTEVAL_LITERAL_ARRAYS
template <String S> constexpr auto matrix_from_string() {
#else
template <String S> consteval auto matrix_from_string() {
#endif
  constexpr std::array<ptrdiff_t, 2> dims = dims_eltype<S>();
  // #if !defined(__clang__)
  //   // we want the array to be dense, so we check if the remainder
  //   // of the number of cols by simd width would allow it to be dense.
  //   // If so, we set `Compess = false`.
  //   constexpr bool compress =
  //     (dims[0] > 1) && (dims[1] % simd::VecLen<dims[1], int64_t>) != 0;
  //   math::StaticArray<int64_t, dims[0], dims[1], compress> A(int64_t(0));
  // #else
  math::StaticArray<int64_t, dims[0], dims[1], true> A{};
  // #endif
  const char *s = S.data_;
  for (ptrdiff_t i = 0; i < dims[0]; ++i) {
    for (ptrdiff_t j = 0; j < dims[1]; ++j) {
      int64_t x = 0;
      char c = *s;
      while (c != '-' && (c < '0' || c > '9')) c = *(++s);
      bool neg = c == '-';
      if (neg) c = *(++s);
      do {
        x = x * 10 + (c - '0');
        c = *(++s);
      } while (c >= '0' && c <= '9');
      if (neg) x = -x;
      A.set(x, i, j);
    }
  }
  return A;
}

#ifdef USE_MODULE
export namespace utils {
#else
namespace utils {
#endif

// constexpr auto cstoll(const char *s, ptrdiff_t &cur) -> int64_t {
//   int64_t res = 0;
//   bool neg = false;
//   while (s[cur] == ' ') ++cur;
//   if (s[cur] == '-') {
//     neg = true;
//     ++cur;
//   }
//   while (s[cur] >= '0' && s[cur] <= '9') {
//     res = res * 10 + (s[cur] - '0');
//     ++cur;
//   }
//   return neg ? -res : res;
// }

// static_assert(__cpp_nontype_template_args >= 201911);

#ifndef CONSTEVAL_LITERAL_ARRAYS
template <String S> [[nodiscard]] constexpr auto operator"" _mat() {
#else
template <String S> [[nodiscard]] consteval auto operator"" _mat() {
#endif
  return matrix_from_string<S>();
}

} // namespace utils
