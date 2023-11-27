#pragma once
#ifndef POLY_SIMD_Transpose_hpp_INCLUDED
#define POLY_SIMD_Transpose_hpp_INCLUDED

#include "SIMD/Unroll.hpp"
namespace poly::simd {

// 4 x 4C -> 4C x 4
template <ptrdiff_t C, typename T>
constexpr auto transpose(Unroll<2, C, 2, T> u) -> Unroll<2 * C, 1, 2, T> {
  Unroll<2 * C, 1, 2, T> z;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < C; ++i) {
    Vec<2, T> a{u[0, i]}, b{u[1, i]};
    z[0, i] = __builtin_shufflevector(a, b, 0, 2);
    z[1, i] = __builtin_shufflevector(a, b, 1, 3);
  }
  return z;
}
template <ptrdiff_t C, typename T>
constexpr auto transpose(Unroll<4, C, 4, T> u) -> Unroll<4 * C, 1, 4, T> {
  Unroll<4 * C, 1, 4, T> z;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < C; ++i) {
    Vec<4, T> a{u[0, i]}, b{u[1, i]}, c{u[2, i]}, d{u[3, i]};
    Vec<4, T> e{__builtin_shufflevector(a, b, 0, 1, 4, 5)};
    Vec<4, T> f{__builtin_shufflevector(a, b, 2, 3, 6, 7)};
    Vec<4, T> g{__builtin_shufflevector(c, d, 0, 1, 4, 5)};
    Vec<4, T> h{__builtin_shufflevector(c, d, 2, 3, 6, 7)};
    z[0, i] = __builtin_shufflevector(e, g, 0, 2, 4, 6);
    z[1, i] = __builtin_shufflevector(e, g, 1, 3, 5, 7);
    z[2, i] = __builtin_shufflevector(f, h, 0, 2, 4, 6);
    z[3, i] = __builtin_shufflevector(f, h, 1, 3, 5, 7);
  }
  return z;
}
template <ptrdiff_t C, typename T>
constexpr auto transpose(Unroll<8, C, 8, T> u) -> Unroll<8 * C, 1, 8, T> {
  Unroll<8 * C, 1, 8, T> z;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < C; ++i) {
    Vec<4, T> a{u[0, i]}, b{u[1, i]}, c{u[2, i]}, d{u[3, i]}, e{u[4, i]},
      f{u[5, i]}, g{u[6, i]}, h{u[7, i]};
    Vec<4, T> j{__builtin_shufflevector(a, b, 0, 8, 2, 10, 4, 12, 6, 14)},
      k{__builtin_shufflevector(a, b, 1, 9, 3, 11, 5, 13, 7, 15)},
      l{__builtin_shufflevector(c, d, 0, 8, 2, 10, 4, 12, 6, 14)},
      m{__builtin_shufflevector(c, d, 1, 9, 3, 11, 5, 13, 7, 15)},
      n{__builtin_shufflevector(e, f, 0, 8, 2, 10, 4, 12, 6, 14)},
      o{__builtin_shufflevector(e, f, 1, 9, 3, 11, 5, 13, 7, 15)},
      p{__builtin_shufflevector(g, h, 0, 8, 2, 10, 4, 12, 6, 14)},
      q{__builtin_shufflevector(g, h, 1, 9, 3, 11, 5, 13, 7, 15)};
    a = __builtin_shufflevector(j, l, 0, 1, 8, 9, 4, 5, 12, 13);
    b = __builtin_shufflevector(j, l, 2, 3, 10, 11, 6, 7, 14, 15);
    c = __builtin_shufflevector(k, m, 0, 1, 8, 9, 4, 5, 12, 13);
    d = __builtin_shufflevector(k, m, 2, 3, 10, 11, 6, 7, 14, 15);
    e = __builtin_shufflevector(n, p, 0, 1, 8, 9, 4, 5, 12, 13);
    f = __builtin_shufflevector(n, p, 2, 3, 10, 11, 6, 7, 14, 15);
    g = __builtin_shufflevector(o, q, 0, 1, 8, 9, 4, 5, 12, 13);
    h = __builtin_shufflevector(o, q, 2, 3, 10, 11, 6, 7, 14, 15);
    z[0, i] = __builtin_shufflevector(a, e, 0, 1, 2, 3, 8, 9, 10, 11);
    z[1, i] = __builtin_shufflevector(a, e, 4, 5, 6, 7, 12, 13, 14, 15);
    z[2, i] = __builtin_shufflevector(b, f, 0, 1, 2, 3, 8, 9, 10, 11);
    z[3, i] = __builtin_shufflevector(b, f, 4, 5, 6, 7, 12, 13, 14, 15);
    z[4, i] = __builtin_shufflevector(c, g, 0, 1, 2, 3, 8, 9, 10, 11);
    z[5, i] = __builtin_shufflevector(c, g, 4, 5, 6, 7, 12, 13, 14, 15);
    z[6, i] = __builtin_shufflevector(d, h, 0, 1, 2, 3, 8, 9, 10, 11);
    z[7, i] = __builtin_shufflevector(d, h, 4, 5, 6, 7, 12, 13, 14, 15);
  }
  return z;
}

} // namespace poly::simd
#endif // Transpose_hpp_INCLUDED
