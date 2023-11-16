#pragma once
#ifndef POLY_SIMD_Unroll_hpp_INCLUDED
#define POLY_SIMD_Unroll_hpp_INCLUDED
#include "SIMD/Intrin.hpp"
#include <functional>
namespace poly::simd {
// Vector goes across cols
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t N, typename T> struct Unroll {
  static constexpr ptrdiff_t W = ptrdiff_t(std::bit_ceil(size_t(N)));
  using VT = std::conditional_t<W == 1, T, Vec<W, T>>;
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
  [[gnu::always_inline]] constexpr auto operator+=(T a) -> Unroll &
  requires(W != 1)
  {
    return (*this) += (typename Unroll<R, C, W, T>::VT{} + a);
  }
  [[gnu::always_inline]] constexpr auto operator-=(T a) -> Unroll &
  requires(W != 1)
  {
    return (*this) -= (typename Unroll<R, C, W, T>::VT{} + a);
  }
  [[gnu::always_inline]] constexpr auto operator*=(T a) -> Unroll &
  requires(W != 1)
  {
    return (*this) *= (typename Unroll<R, C, W, T>::VT{} + a);
  }
  [[gnu::always_inline]] constexpr auto operator/=(T a) -> Unroll &
  requires(W != 1)
  {
    return (*this) /= (typename Unroll<R, C, W, T>::VT{} + a);
  }
};

template <ptrdiff_t R0, ptrdiff_t C0, ptrdiff_t W0, ptrdiff_t R1, ptrdiff_t C1,
          ptrdiff_t W1, typename T, typename Op>
[[gnu::always_inline]] constexpr auto
applyop(const Unroll<R0, C0, W0, T> &a, const Unroll<R1, C1, W1, T> &b, Op op) {
  // Possibilities:
  // 1. All match
  // 2. We had separate unrolls across rows and columns, and some arrays
  // were indexed by one or two of them.
  // In the latter case, we could have arrays indexed by rows, cols, or both.
  if constexpr (W0 == W1) {
    // both were indexed by cols, and `C`s should also match
    // or neither were, and they should still match.
    static_assert(C0 == C1);
    if constexpr (R0 == R1) {
      // Both have the same index across rows
      if constexpr (C0 == C1) {
        Unroll<R0, C0, W0, T> c;
        POLYMATHFULLUNROLL
        for (ptrdiff_t i = 0; i < R0 * C0; ++i)
          c.data[i] = op(a.data[i], b.data[i]);
        return c;
      }
    } else if constexpr (R0 == 1) {
      // `a` was indexed across cols only
      Unroll<R1, C0, W0, T> z;
      POLYMATHFULLUNROLL
      for (ptrdiff_t r = 0; r < R1; ++r) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C0; ++c)
          z.data[r, c] = op(a.data[c], b.data[r, c]);
      }
      return z;
    } else {
      static_assert(R1 == 1);
      // `b` was indexed across cols only
      Unroll<R0, C0, W0, T> z;
      POLYMATHFULLUNROLL
      for (ptrdiff_t r = 0; r < R0; ++r) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t c = 0; c < C0; ++c)
          z.data[r, c] = op(a.data[r, c], b.data[c]);
      }
      return z;
    }
  } else if constexpr (W0 == 1) {
    static_assert(C0 == 1);
    // `a` was indexed by row only
    Unroll<R0, C1, W1, T> z;
    static_assert(R1 == R0 || R1 == 1);
    POLYMATHFULLUNROLL
    for (ptrdiff_t r = 0; r < R0; ++r) {
      POLYMATHFULLUNROLL
      for (ptrdiff_t c = 0; c < C1; ++c)
        if constexpr (R0 == R1) z.data[r, c] = op(a.data[r], b.data[r, c]);
        else z.data[r, c] = op(a.data[r], b.data[c]);
    }
    return z;
  } else {
    static_assert(W1 == 1 && C1 == 1);
    // `b` was indexed by row only
    Unroll<R1, C0, W0, T> z;
    static_assert(R0 == R1 || R0 == 1);
    POLYMATHFULLUNROLL
    for (ptrdiff_t r = 0; r < R1; ++r) {
      POLYMATHFULLUNROLL
      for (ptrdiff_t c = 0; c < C0; ++c)
        if constexpr (R0 == R1) z.data[r, c] = op(a.data[r, c], b.data[r]);
        else z.data[r, c] = op(a.data[c], b.data[r]);
    }
    return z;
  }
}

template <ptrdiff_t R0, ptrdiff_t C0, ptrdiff_t W0, ptrdiff_t R1, ptrdiff_t C1,
          ptrdiff_t W1, typename T>
[[gnu::always_inline]] constexpr auto
operator+(const Unroll<R0, C0, W0, T> &a, const Unroll<R1, C1, W1, T> &b) {
  return applyop(a, b, std::plus<>{});
}

template <ptrdiff_t R0, ptrdiff_t C0, ptrdiff_t W0, ptrdiff_t R1, ptrdiff_t C1,
          ptrdiff_t W1, typename T>
[[gnu::always_inline]] constexpr auto
operator-(const Unroll<R0, C0, W0, T> &a, const Unroll<R1, C1, W1, T> &b) {
  return applyop(a, b, std::minus<>{});
}

template <ptrdiff_t R0, ptrdiff_t C0, ptrdiff_t W0, ptrdiff_t R1, ptrdiff_t C1,
          ptrdiff_t W1, typename T>
[[gnu::always_inline]] constexpr auto
operator*(const Unroll<R0, C0, W0, T> &a, const Unroll<R1, C1, W1, T> &b) {
  return applyop(a, b, std::multiplies<>{});
}

template <ptrdiff_t R0, ptrdiff_t C0, ptrdiff_t W0, ptrdiff_t R1, ptrdiff_t C1,
          ptrdiff_t W1, typename T>
[[gnu::always_inline]] constexpr auto
operator/(const Unroll<R0, C0, W0, T> &a, const Unroll<R1, C1, W1, T> &b) {
  return applyop(a, b, std::divides<>{});
}

template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator+(const Unroll<R, C, W, T> &a, typename Unroll<R, C, W, T>::VT b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a.data[i] + b;
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator-(const Unroll<R, C, W, T> &a, typename Unroll<R, C, W, T>::VT b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a.data[i] - b;
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator*(const Unroll<R, C, W, T> &a, typename Unroll<R, C, W, T>::VT b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a.data[i] * b;
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator/(const Unroll<R, C, W, T> &a, typename Unroll<R, C, W, T>::VT b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a.data[i] / b;
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator+(const Unroll<R, C, W, T> &a,
                                                T b) -> Unroll<R, C, W, T>
requires(W != 1)
{
  return a + (typename Unroll<R, C, W, T>::VT{} + b);
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator-(const Unroll<R, C, W, T> &a,
                                                T b) -> Unroll<R, C, W, T>
requires(W != 1)
{
  return a - (typename Unroll<R, C, W, T>::VT{} + b);
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator*(const Unroll<R, C, W, T> &a,
                                                T b) -> Unroll<R, C, W, T>
requires(W != 1)
{
  return a * (typename Unroll<R, C, W, T>::VT{} + b);
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator/(const Unroll<R, C, W, T> &a,
                                                T b) -> Unroll<R, C, W, T>
requires(W != 1)
{
  return a / (typename Unroll<R, C, W, T>::VT{} + b);
}

template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator+(typename Unroll<R, C, W, T>::VT a, const Unroll<R, C, W, T> &b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a + b.data[i];
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator-(typename Unroll<R, C, W, T>::VT a, const Unroll<R, C, W, T> &b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a - b.data[i];
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator*(typename Unroll<R, C, W, T>::VT a, const Unroll<R, C, W, T> &b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a * b.data[i];
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto
operator/(typename Unroll<R, C, W, T>::VT a, const Unroll<R, C, W, T> &b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a / b.data[i];
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator+(T b,
                                                const Unroll<R, C, W, T> &a)
  -> Unroll<R, C, W, T>
requires(W != 1)
{
  return (typename Unroll<R, C, W, T>::VT{} + b) + a;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator-(T b,
                                                const Unroll<R, C, W, T> &a)
  -> Unroll<R, C, W, T>
requires(W != 1)
{
  return (typename Unroll<R, C, W, T>::VT{} + b) - a;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator*(T b,
                                                const Unroll<R, C, W, T> &a)
  -> Unroll<R, C, W, T>
requires(W != 1)
{
  return (typename Unroll<R, C, W, T>::VT{} + b) * a;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator/(T b,
                                                const Unroll<R, C, W, T> &a)
  -> Unroll<R, C, W, T>
requires(W != 1)
{
  return (typename Unroll<R, C, W, T>::VT{} + b) / a;
}

template <typename T>
constexpr auto load(const T *p, mask::None<1>) -> const T & {
  return *p;
}
template <typename T>
constexpr auto load(const T *p, mask::None<1>, int32_t) -> const T & {
  return *p;
}

template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t N, typename T, ptrdiff_t X,
          size_t NM, typename MT = mask::None<N>>
[[gnu::always_inline]] constexpr auto
loadunroll(const T *ptr, math::RowStride<X> rowStride, std::array<MT, NM> masks)
  -> Unroll<R, C, N, T> {
  static constexpr auto W = ptrdiff_t(std::bit_ceil(size_t(N)));
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
      for (ptrdiff_t c = 0; c < C; ++c) ret[r, c] = load(ptr + c * W, masks[c]);
    } else {
      POLYMATHFULLUNROLL
      for (ptrdiff_t c = 0; c < C - 1; ++c)
        ret[r, c] = load(ptr + c * W, mask::None<W>{});
      ret[r, C - 1] = load(ptr + (C - 1) * W, masks[0]);
    }
  }
  return ret;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t N, typename T, ptrdiff_t X,
          size_t NM, typename MT = mask::None<N>>
[[gnu::always_inline]] constexpr auto
loadstrideunroll(const T *ptr, math::RowStride<X> rowStride,
                 std::array<MT, NM> masks) -> Unroll<R, C, N, T> {
  static constexpr auto W = ptrdiff_t(std::bit_ceil(size_t(N)));
  Unroll<R, C, N, T> ret;
  auto s = int32_t(ptrdiff_t(rowStride));
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

} // namespace poly::simd
#endif // Unroll_hpp_INCLUDED