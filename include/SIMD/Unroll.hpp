#pragma once
#ifndef POLY_SIMD_Unroll_hpp_INCLUDED
#define POLY_SIMD_Unroll_hpp_INCLUDED

#include "SIMD/Intrin.hpp"
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

template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator+(const Unroll<R, C, W, T> &a,
                                                const Unroll<R, C, W, T> &b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a.data[i] + b.data[i];
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator-(const Unroll<R, C, W, T> &a,
                                                const Unroll<R, C, W, T> &b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a.data[i] - b.data[i];
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator*(const Unroll<R, C, W, T> &a,
                                                const Unroll<R, C, W, T> &b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a.data[i] * b.data[i];
  return c;
}
template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename T>
[[gnu::always_inline]] constexpr auto operator/(const Unroll<R, C, W, T> &a,
                                                const Unroll<R, C, W, T> &b)
  -> Unroll<R, C, W, T> {
  Unroll<R, C, W, T> c;
  POLYMATHFULLUNROLL
  for (ptrdiff_t i = 0; i < R * C; ++i) c.data[i] = a.data[i] / b.data[i];
  return c;
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
