#pragma once
#ifndef POLY_SIMD_Unroll_hpp_INCLUDED
#define POLY_SIMD_Unroll_hpp_INCLUDED
#include "SIMD/Intrin.hpp"
#include <functional>
namespace poly::simd {
// template <typename T, ptrdiff_t W, typename S>
// [[gnu::always_inline]] constexpr auto vcvt(Vec<W, S> v) {
//   if constexpr (std::same_as<T, S>) return v;
//   else if constexpr (W == 1) return T(v);
//   else return __builtin_convertvector(v, Vec<W, T>);
// }
// Vector goes across cols
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t N, typename T> struct Unroll {
  static constexpr ptrdiff_t W = ptrdiff_t(std::bit_ceil(size_t(N)));
  using VT = Vec<W, T>;
  static_assert(R * C > 0);
  VT data[R * C];
  constexpr auto operator[](ptrdiff_t i) -> VT & { return data[i]; }
  constexpr auto operator[](ptrdiff_t r, ptrdiff_t c) -> VT & {
    return data[r * C + c];
  }
  constexpr auto operator[](ptrdiff_t i) const -> VT { return data[i]; }
  constexpr auto operator[](ptrdiff_t r, ptrdiff_t c) const -> VT {
    return data[r * C + c];
  }
  template <typename U>
  [[gnu::always_inline]] constexpr operator Unroll<R, C, N, U>() const
  requires(!std::same_as<T, U>)
  {
    Unroll<R, C, N, U> x;
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i)
      if constexpr (W == 1) x.data[i] = U(data[i]);
      else x.data[i] = __builtin_convertvector(data[i], Vec<W, U>);
    return x;
  }
  [[gnu::always_inline]] constexpr auto operator-() {
    Unroll a;
    for (ptrdiff_t i = 0; i < R * C; ++i) a.data[i] = -data[i];
    return a;
  }
  [[gnu::always_inline]] constexpr auto operator+=(const Unroll &a)
    -> Unroll & {
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) data[i] += a.data[i];
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator-=(const Unroll &a)
    -> Unroll & {
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) data[i] -= a.data[i];
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator*=(const Unroll &a)
    -> Unroll & {
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) data[i] *= a.data[i];
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator/=(const Unroll &a)
    -> Unroll & {
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) data[i] /= a.data[i];
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator+=(VT a) -> Unroll & {
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) data[i] += a;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator-=(VT a) -> Unroll & {
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) data[i] -= a;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator*=(VT a) -> Unroll & {
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) data[i] *= a;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator/=(VT a) -> Unroll & {
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) data[i] /= a;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto
  operator+=(std::convertible_to<T> auto a) -> Unroll &
  requires(W != 1)
  {
    return (*this) += vbroadcast<W, T>(a);
  }
  [[gnu::always_inline]] constexpr auto
  operator-=(std::convertible_to<T> auto a) -> Unroll &
  requires(W != 1)
  {
    return (*this) -= vbroadcast<W, T>(a);
  }
  [[gnu::always_inline]] constexpr auto
  operator*=(std::convertible_to<T> auto a) -> Unroll &
  requires(W != 1)
  {
    return (*this) *= vbroadcast<W, T>(a);
  }
  [[gnu::always_inline]] constexpr auto
  operator/=(std::convertible_to<T> auto a) -> Unroll &
  requires(W != 1)
  {
    return (*this) /= vbroadcast<W, T>(a);
  }
};
template <ptrdiff_t N, typename T> struct Unroll<1, 1, N, T> {
  static constexpr ptrdiff_t W = ptrdiff_t(std::bit_ceil(size_t(N)));
  using VT = Vec<W, T>;
  VT vec;
  constexpr auto operator[](ptrdiff_t) -> VT & { return vec; }
  constexpr auto operator[](ptrdiff_t, ptrdiff_t) -> VT & { return vec; }
  constexpr auto operator[](ptrdiff_t) const -> VT { return vec; }
  constexpr auto operator[](ptrdiff_t, ptrdiff_t) const -> VT { return vec; }
  template <typename U>
  [[gnu::always_inline]] constexpr operator Unroll<1, 1, N, U>() const
  requires(!std::same_as<T, U>)
  {
    if constexpr (W == 1) return {U(vec)};
    else return {__builtin_convertvector(vec, Vec<W, U>)};
  }
  [[gnu::always_inline]] constexpr auto operator-() { return Unroll{-vec}; }
  [[gnu::always_inline]] constexpr auto operator+=(const Unroll &a)
    -> Unroll & {
    vec += a.vec;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator-=(const Unroll &a)
    -> Unroll & {
    vec -= a.vec;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator*=(const Unroll &a)
    -> Unroll & {
    vec *= a.vec;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator/=(const Unroll &a)
    -> Unroll & {
    vec /= a.vec;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator+=(VT a) -> Unroll & {
    vec += a;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator-=(VT a) -> Unroll & {
    vec -= a;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator*=(VT a) -> Unroll & {
    vec *= a;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator/=(VT a) -> Unroll & {
    vec /= a;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto
  operator+=(std::convertible_to<T> auto a) -> Unroll &
  requires(W != 1)
  {
    vec += vbroadcast<W, T>(a);
    return *this;
  }
  [[gnu::always_inline]] constexpr auto
  operator-=(std::convertible_to<T> auto a) -> Unroll &
  requires(W != 1)
  {
    vec -= vbroadcast<W, T>(a);
    return *this;
  }
  [[gnu::always_inline]] constexpr auto
  operator*=(std::convertible_to<T> auto a) -> Unroll &
  requires(W != 1)
  {
    vec *= vbroadcast<W, T>(a);
    return *this;
  }
  [[gnu::always_inline]] constexpr auto
  operator/=(std::convertible_to<T> auto a) -> Unroll &
  requires(W != 1)
  {
    vec /= vbroadcast<W, T>(a);
    return *this;
  }
};

template <ptrdiff_t R0, ptrdiff_t C0, ptrdiff_t W0, typename T0, ptrdiff_t R1,
          ptrdiff_t C1, ptrdiff_t W1, typename T1, typename Op>
[[gnu::always_inline]] constexpr auto applyop(const Unroll<R0, C0, W0, T0> &a,
                                              const Unroll<R1, C1, W1, T1> &b,
                                              Op op) {
  // Possibilities:
  // 1. All match
  // 2. We had separate unrolls across rows and columns, and some arrays
  // were indexed by one or two of them.
  // In the latter case, we could have arrays indexed by rows, cols, or both.
  if constexpr (!std::same_as<T0, T1>) {
    using T = std::common_type_t<T0, T1>;
    return applyop(Unroll<R0, C0, W0, T>(a), Unroll<R1, C1, W1, T>(b), op);
  } else if constexpr (W0 == W1) {
    // both were indexed by cols, and `C`s should also match
    // or neither were, and they should still match.
    static_assert(C0 == C1);
    if constexpr (R0 == R1) {
      // Both have the same index across rows
      Unroll<R0, C0, W0, T0> c;
      if constexpr ((R0 == 1) && (C0 == 1)) {
        c.vec = op(a.vec, b.vec);
      } else {
        POLYMATHFULLUNROLL
        for (ptrdiff_t i = 0; i < R0 * C0; ++i)
          c.data[i] = op(a.data[i], b.data[i]);
      }
      return c;
    } else if constexpr (R0 == 1) { // R1 > 0
      // `a` was indexed across cols only
      Unroll<R1, C0, W0, T0> z;
      if constexpr (C0 == 1) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t r = 0; r < R1; ++r) z.data[r] = op(a.vec, b.data[r]);
      } else {
        POLYMATHFULLUNROLL
        for (ptrdiff_t r = 0; r < R1; ++r) {
          POLYMATHFULLUNROLL
          for (ptrdiff_t c = 0; c < C0; ++c) z[r, c] = op(a.data[c], b[r, c]);
        }
      }
      return z;
    } else {
      static_assert(R1 == 1); // R0 > 0
      // `b` was indexed across cols only
      Unroll<R0, C0, W0, T0> z;
      if constexpr (C0 == 1) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t r = 0; r < R0; ++r) z.data[r] = op(a.data[r], b.vec);
      } else {
        POLYMATHFULLUNROLL
        for (ptrdiff_t r = 0; r < R0; ++r) {
          POLYMATHFULLUNROLL
          for (ptrdiff_t c = 0; c < C0; ++c) z[r, c] = op(a[r, c], b.data[c]);
        }
      }
      return z;
    }
  } else if constexpr (W0 == 1) {
    static_assert(R0 == 1 || C0 == 1);
    constexpr ptrdiff_t R = R0 == 1 ? C0 : R0;
    // `a` was indexed by row only
    Unroll<R, C1, W1, T0> z;
    static_assert(R1 == R || R1 == 1);
    if constexpr ((R == 1) && (C1 == 1)) z.vec = op(a.vec, b.vec);
    else if constexpr (R == 1) {
      POLYMATHFULLUNROLL
      for (ptrdiff_t c = 0; c < C1; ++c) z.data[c] = op(a.vec, b.data[c]);
    } else if constexpr (C1 == 1) {
      POLYMATHFULLUNROLL
      for (ptrdiff_t r = 0; r < R; ++r)
        if constexpr (R == R1) z.data[r] = op(a.data[r], b.vec[r]);
        else z.data[r] = op(a.data[r], b.vec);
    } else {
      POLYMATHFULLUNROLL
      for (ptrdiff_t r = 0; r < R; ++r) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C1; ++c)
          if constexpr (R == R1) z[r, c] = op(a.data[r], b[r, c]);
          else z[r, c] = op(a.data[r], b.data[c]);
      }
    }
    return z;
  } else {
    static_assert(W1 == 1);
    static_assert(R1 == 1 || C1 == 1);
    constexpr ptrdiff_t R = R1 == 1 ? C1 : R1;
    // `b` was indexed by row only
    Unroll<R, C0, W0, T0> z;
    static_assert(R0 == R || R0 == 1);
    if constexpr ((R == 1) && (C0 == 1)) z.vec = op(a.vec, b.vec);
    else if constexpr (R == 1) {
      POLYMATHFULLUNROLL
      for (ptrdiff_t c = 0; c < C0; ++c) z.data[c] = op(a.data[c], b.vec);
    } else if constexpr (C0 == 1) {
      POLYMATHFULLUNROLL
      for (ptrdiff_t r = 0; r < R; ++r)
        if constexpr (R0 == R) z.data[r] = op(a.data[r], b.data[r]);
        else z.data[r] = op(a.vec, b.data[r]);
    } else {
      POLYMATHFULLUNROLL
      for (ptrdiff_t r = 0; r < R; ++r) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C0; ++c)
          if constexpr (R0 == R) z[r, c] = op(a[r, c], b.data[r]);
          else z[r, c] = op(a.data[c], b.data[r]);
      }
    }
    return z;
  }
}

template <ptrdiff_t R0, ptrdiff_t C0, ptrdiff_t W0, typename T0, ptrdiff_t R1,
          ptrdiff_t C1, ptrdiff_t W1, typename T1>
[[gnu::always_inline]] constexpr auto
operator+(const Unroll<R0, C0, W0, T0> &a, const Unroll<R1, C1, W1, T1> &b) {
  return applyop(a, b, std::plus<>{});
}

template <ptrdiff_t R0, ptrdiff_t C0, ptrdiff_t W0, typename T0, ptrdiff_t R1,
          ptrdiff_t C1, ptrdiff_t W1, typename T1>
[[gnu::always_inline]] constexpr auto
operator-(const Unroll<R0, C0, W0, T0> &a, const Unroll<R1, C1, W1, T1> &b) {
  return applyop(a, b, std::minus<>{});
}

template <ptrdiff_t R0, ptrdiff_t C0, ptrdiff_t W0, typename T0, ptrdiff_t R1,
          ptrdiff_t C1, ptrdiff_t W1, typename T1>
[[gnu::always_inline]] constexpr auto
operator*(const Unroll<R0, C0, W0, T0> &a, const Unroll<R1, C1, W1, T1> &b) {
  return applyop(a, b, std::multiplies<>{});
}

template <ptrdiff_t R0, ptrdiff_t C0, ptrdiff_t W0, typename T0, ptrdiff_t R1,
          ptrdiff_t C1, ptrdiff_t W1, typename T1>
[[gnu::always_inline]] constexpr auto
operator/(const Unroll<R0, C0, W0, T0> &a, const Unroll<R1, C1, W1, T1> &b) {
  return applyop(a, b, std::divides<>{});
}

template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator+(const Unroll<R, C, W, T> &a, typename Unroll<R, C, W, T>::VT b)
  -> Unroll<R, C, W, T> {
  if constexpr (R * C == 1) return {a.vec + b};
  else {
    Unroll<R, C, W, T> c;
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a.data[i] + b;
    return c;
  }
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator-(const Unroll<R, C, W, T> &a, typename Unroll<R, C, W, T>::VT b)
  -> Unroll<R, C, W, T> {
  if constexpr (R * C == 1) return {a.vec - b};
  else {
    Unroll<R, C, W, T> c;
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a.data[i] - b;
    return c;
  }
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator*(const Unroll<R, C, W, T> &a, typename Unroll<R, C, W, T>::VT b)
  -> Unroll<R, C, W, T> {
  if constexpr (R * C == 1) return {a.vec * b};
  else {
    Unroll<R, C, W, T> c;
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a.data[i] * b;
    return c;
  }
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator/(const Unroll<R, C, W, T> &a, typename Unroll<R, C, W, T>::VT b)
  -> Unroll<R, C, W, T> {
  if constexpr (R * C == 1) return {a.vec / b};
  else {
    Unroll<R, C, W, T> c;
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a.data[i] / b;
    return c;
  }
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator+(const Unroll<R, C, W, T> &a,
                                                std::convertible_to<T> auto b)
  -> Unroll<R, C, W, T>
requires(W != 1)
{
  return a + vbroadcast<W, T>(b);
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator-(const Unroll<R, C, W, T> &a,
                                                std::convertible_to<T> auto b)
  -> Unroll<R, C, W, T>
requires(W != 1)
{
  return a - vbroadcast<W, T>(b);
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator*(const Unroll<R, C, W, T> &a,
                                                std::convertible_to<T> auto b)
  -> Unroll<R, C, W, T>
requires(W != 1)
{
  return a * vbroadcast<W, T>(b);
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator/(const Unroll<R, C, W, T> &a,
                                                std::convertible_to<T> auto b)
  -> Unroll<R, C, W, T>
requires(W != 1)
{
  return a / vbroadcast<W, T>(b);
}

template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator+(typename Unroll<R, C, W, T>::VT a, const Unroll<R, C, W, T> &b)
  -> Unroll<R, C, W, T> {
  if constexpr (R * C == 1) return {a + b.vec};
  else {
    Unroll<R, C, W, T> c;
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a + b.data[i];
    return c;
  }
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator-(typename Unroll<R, C, W, T>::VT a, const Unroll<R, C, W, T> &b)
  -> Unroll<R, C, W, T> {
  if constexpr (R * C == 1) return {a - b.vec};
  else {
    Unroll<R, C, W, T> c;
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a - b.data[i];
    return c;
  }
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator*(typename Unroll<R, C, W, T>::VT a, const Unroll<R, C, W, T> &b)
  -> Unroll<R, C, W, T> {
  if constexpr (R * C == 1) return {a * b.vec};
  else {
    Unroll<R, C, W, T> c;
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a * b.data[i];
    return c;
  }
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator/(typename Unroll<R, C, W, T>::VT a, const Unroll<R, C, W, T> &b)
  -> Unroll<R, C, W, T> {
  if constexpr (R * C == 1) return {a / b.vec};
  else {
    Unroll<R, C, W, T> c;
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a / b.data[i];
    return c;
  }
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator+(T b,
                                                const Unroll<R, C, W, T> &a)
  -> Unroll<R, C, W, T>
requires(W != 1)
{
  return vbroadcast<W, T>(b) + a;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator-(T b,
                                                const Unroll<R, C, W, T> &a)
  -> Unroll<R, C, W, T>
requires(W != 1)
{
  return vbroadcast<W, T>(b) - a;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator*(T b,
                                                const Unroll<R, C, W, T> &a)
  -> Unroll<R, C, W, T>
requires(W != 1)
{
  return vbroadcast<W, T>(b) * a;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator/(T b,
                                                const Unroll<R, C, W, T> &a)
  -> Unroll<R, C, W, T>
requires(W != 1)
{
  return vbroadcast<W, T>(b) / a;
}

template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t N, typename T, ptrdiff_t X,
          size_t NM, typename MT = mask::None<N>>
[[gnu::always_inline]] constexpr auto
loadunroll(const T *ptr, math::RowStride<X> rowStride, std::array<MT, NM> masks)
  -> Unroll<R, C, N, T> {
  if constexpr (R * C == 1) return {load(ptr, masks[0])};
  else {
    constexpr auto W = ptrdiff_t(std::bit_ceil(size_t(N)));
    auto rs = ptrdiff_t(rowStride);
    Unroll<R, C, N, T> ret;
    POLYMATHFULLUNROLL
    for (ptrdiff_t r = 0; r < R; ++r, ptr += rs) {
      if constexpr (NM == 0) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C; ++c)
          ret[r, c] = load(ptr + c * W, mask::None<W>{});
      } else if constexpr (NM == C) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C; ++c)
          ret[r, c] = load(ptr + c * W, masks[c]);
      } else {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C - 1; ++c)
          ret[r, c] = load(ptr + c * W, mask::None<W>{});
        ret[r, C - 1] = load(ptr + (C - 1) * W, masks[0]);
      }
    }
    return ret;
  }
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t N, typename T, ptrdiff_t X,
          size_t NM, typename MT = mask::None<N>>
[[gnu::always_inline]] constexpr auto
loadstrideunroll(const T *ptr, math::RowStride<X> rowStride,
                 std::array<MT, NM> masks) -> Unroll<R, C, N, T> {
  auto s = int32_t(ptrdiff_t(rowStride));
  if constexpr (R * C == 1) return {load(ptr, masks[0], s)};
  else {
    constexpr auto W = ptrdiff_t(std::bit_ceil(size_t(N)));
    Unroll<R, C, N, T> ret;
    POLYMATHFULLUNROLL
    for (ptrdiff_t r = 0; r < R; ++r, ++ptr) {
      if constexpr (NM == 0) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C; ++c)
          ret[r, c] = load(ptr + c * W * s, mask::None<W>{}, s);
      } else if constexpr (NM == C) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C; ++c)
          ret[r, c] = load(ptr + c * W * s, masks[c], s);
      } else {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C - 1; ++c)
          ret[r, c] = load(ptr + c * W * s, mask::None<W>{}, s);
        ret[r, C - 1] = load(ptr + (C - 1) * W * s, masks[0], s);
      }
    }
    return ret;
  }
}

// Represents a reference for a SIMD load, in particular so that we can store.
// Needs to support masking
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t N, typename T, ptrdiff_t X,
          ptrdiff_t NM, typename MT = mask::None<N>, bool Transposed = false>
struct UnrollRef {
  static constexpr ptrdiff_t W = ptrdiff_t(std::bit_ceil(size_t(N)));
  static_assert(N == W || C == 1,
                "If N != the next power of `2`, then `C` should be `1`");
  static_assert(
    NM == 0 || NM == 1 || NM == C,
    "Should have no masks, one mask for last `C`, or one mask per `C`");
  T *ptr;
  [[no_unique_address]] math::RowStride<X> rowStride;
  [[no_unique_address]] std::array<MT, NM> masks;
  constexpr operator Unroll<R, C, N, T>() {
    if constexpr (!Transposed)
      return loadunroll<R, C, N, T, X, NM, MT>(ptr, rowStride, masks);
    else return loadstrideunroll<R, C, N, T, X, NM, MT>(ptr, rowStride, masks);
  }
  constexpr auto operator=(Unroll<R, C, N, T> x) -> UnrollRef &
  requires(!Transposed)
  {
    auto rs = ptrdiff_t(rowStride);
    T *p = ptr;
    POLYMATHFULLUNROLL
    for (ptrdiff_t r = 0; r < R; ++r, p += rs) {
      if constexpr (NM == 0) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C; ++c)
          store<T>(p + c * W, mask::None<W>{}, x[r, c]);
      } else if constexpr (NM == C) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C; ++c)
          store<T>(p + c * W, masks[c], x[r, c]);
      } else { // NM == 1
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C - 1; ++c)
          store<T>(p + c * W, mask::None<W>{}, x[r, c]);
        store<T>(p + (C - 1) * W, masks[0], x[r, C - 1]);
      }
    }
    return *this;
  }
  constexpr auto operator=(Vec<W, T> v) -> UnrollRef &
  requires(!Transposed)
  {
    auto rs = ptrdiff_t(rowStride);
    T *p = ptr;
    POLYMATHFULLUNROLL
    for (ptrdiff_t r = 0; r < R; ++r, p += rs) {
      if constexpr (NM == 0) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C; ++c)
          store<T>(p + c * W, mask::None<W>{}, v);
      } else if constexpr (NM == C) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C; ++c) store<T>(p + c * W, masks[c], v);
      } else { // NM == 1
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C - 1; ++c)
          store<T>(p + c * W, mask::None<W>{}, v);
        store<T>(p + (C - 1) * W, masks[0], v);
      }
    }
    return *this;
  }
  constexpr auto operator=(Unroll<R, C, N, T> x) -> UnrollRef &
  requires(Transposed)
  {
    auto s = int32_t(ptrdiff_t(rowStride));
    T *p = ptr;
    for (ptrdiff_t r = 0; r < R; ++r, ++p) {
      if constexpr (NM == 0) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C; ++c)
          store<T>(p + c * W * s, mask::None<W>{}, x[0, c], s);
      } else if constexpr (NM == C) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C; ++c)
          store<T>(p + c * W * s, masks[c], x[0, c], s);
      } else {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C - 1; ++c)
          store<T>(p + c * W * s, mask::None<W>{}, x[0, c], s);
        store<T>(p + (C - 1) * W * s, masks[0], x[0, C - 1], s);
      }
    }
    return *this;
  }
  constexpr auto operator=(Vec<W, T> v) -> UnrollRef &
  requires(Transposed)
  {
    auto s = int32_t(ptrdiff_t(rowStride));
    T *p = ptr;
    for (ptrdiff_t r = 0; r < R; ++r, ++p) {
      if constexpr (NM == 0) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C; ++c)
          store<T>(p + c * W * s, mask::None<W>{}, v, s);
      } else if constexpr (NM == C) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C; ++c)
          store<T>(p + c * W * s, masks[c], v, s);
      } else {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C - 1; ++c)
          store<T>(p + c * W * s, mask::None<W>{}, v, s);
        store<T>(p + (C - 1) * W * s, masks[0], v, s);
      }
    }
    return *this;
  }
  constexpr auto operator=(std::convertible_to<T> auto x) -> UnrollRef & {
    *this = Vec<W, T>{} + T(x);
    return *this;
  }
  constexpr auto operator+=(const auto &x) -> UnrollRef & {
    return (*this) = Unroll<R, C, N, T>(*this) + x;
  }
  constexpr auto operator-=(const auto &x) -> UnrollRef & {
    return (*this) = Unroll<R, C, N, T>(*this) - x;
  }
  constexpr auto operator*=(const auto &x) -> UnrollRef & {
    return (*this) = Unroll<R, C, N, T>(*this) * x;
  }
  constexpr auto operator/=(const auto &x) -> UnrollRef & {
    return (*this) = Unroll<R, C, N, T>(*this) / x;
  }
};
template <typename T, ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename M,
          bool Transposed, ptrdiff_t X>
[[gnu::always_inline]] constexpr auto
ref(const T *p, index::UnrollDims<R, C, W, M, Transposed, X> i)
  -> Unroll<R, C, W, T> {
  if constexpr (Transposed)
    return loadstrideunroll<R, C, W>(p, i.rs, std::array<M, 1>{i.mask});
  else return loadunroll<R, C, W>(p, i.rs, std::array<M, 1>{i.mask});
}
template <typename T, ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename M,
          bool Transposed, ptrdiff_t X>
[[gnu::always_inline]] constexpr auto
ref(T *p, index::UnrollDims<R, C, W, M, Transposed, X> i)
  -> UnrollRef<R, C, W, T, X, 1, M, Transposed> {
  return {p, i.rs, std::array<M, 1>{i.mask}};
}
namespace index {

template <ptrdiff_t U, ptrdiff_t W, typename M>
constexpr auto operator==(Unroll<U, W, M> x, ptrdiff_t y) {
  if constexpr (W == 1) {
    if constexpr (U > 1) {
      poly::simd::Unroll<U, 1, 1, int64_t> ret;
      POLYMATHFULLUNROLL
      for (ptrdiff_t u = 0; u < U; ++u) ret.data[u] = (x.index + u) == y;
      return ret;
    } else return poly::simd::Unroll<1, 1, W, int64_t>{x.index == y};
  } else if constexpr (U > 1) {
    poly::simd::Unroll<1, U, W, int64_t> ret;
    Vec<W, int64_t> v = vbroadcast<W, int64_t>(y - x.index);
    POLYMATHFULLUNROLL
    for (ptrdiff_t u = 0; u < U; ++u)
      ret.data[u] = range<W, int64_t>() == (v - u * W);
    return ret;
  } else
    return poly::simd::Unroll<1, 1, W, int64_t>{
      range<W, int64_t>() == vbroadcast<W, int64_t>(y - x.index)};
}
template <ptrdiff_t U, ptrdiff_t W, typename M>
constexpr auto operator!=(Unroll<U, W, M> x, ptrdiff_t y) {
  if constexpr (W == 1) {
    if constexpr (U > 1) {
      poly::simd::Unroll<U, 1, 1, int64_t> ret;
      POLYMATHFULLUNROLL
      for (ptrdiff_t u = 0; u < U; ++u) ret.data[u] = (x.index + u) != y;
      return ret;
    } else return poly::simd::Unroll<1, 1, W, int64_t>{x.index != y};
  } else if constexpr (U > 1) {
    poly::simd::Unroll<1, U, W, int64_t> ret;
    Vec<W, int64_t> v = vbroadcast<W, int64_t>(y - x.index);
    POLYMATHFULLUNROLL
    for (ptrdiff_t u = 0; u < U; ++u)
      ret.data[u] = range<W, int64_t>() != (v - u * W);
    return ret;
  } else
    return poly::simd::Unroll<1, 1, W, int64_t>{
      range<W, int64_t>() != vbroadcast<W, int64_t>(y - x.index)};
}

template <ptrdiff_t U, ptrdiff_t W, typename M>
constexpr auto operator<(Unroll<U, W, M> x, ptrdiff_t y) {
  if constexpr (W == 1) {
    if constexpr (U > 1) {
      poly::simd::Unroll<U, 1, 1, int64_t> ret;
      POLYMATHFULLUNROLL
      for (ptrdiff_t u = 0; u < U; ++u) ret.data[u] = (x.index + u) < y;
      return ret;
    } else return poly::simd::Unroll<1, 1, W, int64_t>{x.index < y};
  } else if constexpr (U > 1) {
    poly::simd::Unroll<1, U, W, int64_t> ret;
    Vec<W, int64_t> v = vbroadcast<W, int64_t>(y - x.index);
    POLYMATHFULLUNROLL
    for (ptrdiff_t u = 0; u < U; ++u)
      ret.data[u] = range<W, int64_t>() < (v - u * W);
    return ret;
  } else
    return poly::simd::Unroll<1, 1, W, int64_t>{
      range<W, int64_t>() < vbroadcast<W, int64_t>(y - x.index)};
}

template <ptrdiff_t U, ptrdiff_t W, typename M>
constexpr auto operator>(Unroll<U, W, M> x, ptrdiff_t y) {
  if constexpr (W == 1) {
    if constexpr (U > 1) {
      poly::simd::Unroll<U, 1, 1, int64_t> ret;
      POLYMATHFULLUNROLL
      for (ptrdiff_t u = 0; u < U; ++u) ret.data[u] = (x.index + u) > y;
      return ret;
    } else return poly::simd::Unroll<1, 1, W, int64_t>{x.index > y};
  } else if constexpr (U > 1) {
    poly::simd::Unroll<1, U, W, int64_t> ret;
    Vec<W, int64_t> v = vbroadcast<W, int64_t>(y - x.index);
    POLYMATHFULLUNROLL
    for (ptrdiff_t u = 0; u < U; ++u)
      ret.data[u] = range<W, int64_t>() > (v - u * W);
    return ret;
  } else
    return poly::simd::Unroll<1, 1, W, int64_t>{
      range<W, int64_t>() > vbroadcast<W, int64_t>(y - x.index)};
}

template <ptrdiff_t U, ptrdiff_t W, typename M>
constexpr auto operator<=(Unroll<U, W, M> x, ptrdiff_t y) {
  if constexpr (W == 1) {
    if constexpr (U > 1) {
      poly::simd::Unroll<U, 1, 1, int64_t> ret;
      POLYMATHFULLUNROLL
      for (ptrdiff_t u = 0; u < U; ++u) ret.data[u] = (x.index + u) <= y;
      return ret;
    } else return poly::simd::Unroll<1, 1, W, int64_t>{x.index <= y};
  } else if constexpr (U > 1) {
    poly::simd::Unroll<1, U, W, int64_t> ret;
    Vec<W, int64_t> v = vbroadcast<W, int64_t>(y - x.index);
    POLYMATHFULLUNROLL
    for (ptrdiff_t u = 0; u < U; ++u)
      ret.data[u] = range<W, int64_t>() <= (v - u * W);
    return ret;
  } else
    return poly::simd::Unroll<1, 1, W, int64_t>{
      range<W, int64_t>() <= vbroadcast<W, int64_t>(y - x.index)};
}

template <ptrdiff_t U, ptrdiff_t W, typename M>
constexpr auto operator>=(Unroll<U, W, M> x, ptrdiff_t y) {
  if constexpr (W == 1) {
    if constexpr (U > 1) {
      poly::simd::Unroll<U, 1, 1, int64_t> ret;
      POLYMATHFULLUNROLL
      for (ptrdiff_t u = 0; u < U; ++u) ret.data[u] = (x.index + u) >= y;
      return ret;
    } else return poly::simd::Unroll<1, 1, W, int64_t>{x.index >= y};
  } else if constexpr (U > 1) {
    poly::simd::Unroll<1, U, W, int64_t> ret;
    Vec<W, int64_t> v = vbroadcast<W, int64_t>(y - x.index);
    POLYMATHFULLUNROLL
    for (ptrdiff_t u = 0; u < U; ++u)
      ret.data[u] = range<W, int64_t>() >= (v - u * W);
    return ret;
  } else
    return poly::simd::Unroll<1, 1, W, int64_t>{
      range<W, int64_t>() >= vbroadcast<W, int64_t>(y - x.index)};
}

} // namespace index

} // namespace poly::simd
#endif // Unroll_hpp_INCLUDED
