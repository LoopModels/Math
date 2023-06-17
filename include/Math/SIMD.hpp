#pragma once

#include "Math/AxisTypes.hpp"
#include "Math/Indexing.hpp"
#include "Math/MatrixDimensions.hpp"
#include <array>
#include <concepts>
#include <cstddef>
#include <eve/module/core.hpp>

namespace poly::math {
namespace simd {
struct NoPredicate {
  constexpr explicit operator ptrdiff_t() const { return 0; }
};

template <ptrdiff_t W, ptrdiff_t N, typename P> struct Unroll {
  [[no_unique_address]] ptrdiff_t i;
  [[no_unique_address]] P p;
  constexpr explicit operator ptrdiff_t() const { return i; }
};

template <ptrdiff_t W, ptrdiff_t M, ptrdiff_t N, typename P> struct Tile {
  [[no_unique_address]] ptrdiff_t row, col;
  [[no_unique_address]] P p;
  constexpr explicit operator CartesianIndex<ptrdiff_t, ptrdiff_t>() const {
    return {row, col};
  }
};
template <ptrdiff_t W, ptrdiff_t N, typename P>
struct UnrollOffset : public Unroll<W, N, P> {
  [[nodiscard]] constexpr auto offset() const -> ptrdiff_t { return this->i; }
};
template <ptrdiff_t W, ptrdiff_t N, typename P>
struct StridedUnrollOffset : public Unroll<W, N, P> {
  [[no_unique_address]] ptrdiff_t stride;
};
template <ptrdiff_t W, ptrdiff_t M, ptrdiff_t N, typename P>
struct TileOffset : public Tile<W, M, N, P> {
  [[nodiscard]] constexpr auto offset() const -> ptrdiff_t { return this->row; }
  [[nodiscard]] constexpr auto stride() const -> ptrdiff_t { return this->col; }
};

static_assert(sizeof(Tile<8, 2, 2, NoPredicate>) == sizeof(ptrdiff_t) * 2);
static_assert(sizeof(Tile<8, 2, 2, ptrdiff_t>) == sizeof(ptrdiff_t) * 3);

template <ptrdiff_t W, ptrdiff_t N, typename P>
[[nodiscard]] inline constexpr auto calcOffset(ptrdiff_t len, Unroll<W, N, P> i)
  -> UnrollOffset<W, N, P> {
  invariant((ptrdiff_t(i) + W * N - ptrdiff_t(i.p)) <= len);
  return {i, i.p};
}
template <ptrdiff_t W, ptrdiff_t N, typename P>
constexpr auto calcOffset(StridedRange d, Unroll<W, N, P> i)
  -> StridedUnrollOffset<W, N, P> {
  return {d.stride * calcOffset(d.len, i), i.p, d.stride};
}

template <ptrdiff_t W, ptrdiff_t M, ptrdiff_t N, typename P>
[[nodiscard]] inline constexpr auto calcOffset(StridedDims d,
                                               Tile<W, M, N, P> i)
  -> TileOffset<W, M, N, P> {
  invariant((ptrdiff_t(i.col) + N * W - ptrdiff_t(i.p)) <= ptrdiff_t(Col{d}));
  invariant((ptrdiff_t(i.row) + M) <= ptrdiff_t(Row{d}));
  ptrdiff_t x = ptrdiff_t(RowStride{d});
  ptrdiff_t r = x * calcOffset(ptrdiff_t(Row{d}), i.row);
  ptrdiff_t c = calcOffset(ptrdiff_t(Col{d}), i.col);
  return {r + c, x, i.p};
}

template <ptrdiff_t W, ptrdiff_t N, typename P>
constexpr auto calcNewDim(VectorDimension auto, Unroll<W, N, P>) -> Empty {
  return {};
}
template <ptrdiff_t W, ptrdiff_t M, ptrdiff_t N, typename P>
constexpr auto calcNewDim(MatrixDimension auto, Tile<W, M, N, P>) -> Empty {
  return {};
}
} // namespace simd
template <typename T> constexpr auto load(T *p, ptrdiff_t i) -> T & {
  return p[i];
}
template <typename T>
constexpr auto load(const T *p, ptrdiff_t i) -> const T & {
  return p[i];
}

template <typename T, ptrdiff_t W, ptrdiff_t N, typename P>
constexpr auto load(const T *p, simd::UnrollOffset<W, N, P> i)
  -> std::array<eve::wide<T, W>, N> {
  std::array<eve::wide<T, W>, N> ret;
  constexpr bool pred = !std::same_as<P, simd::NoPredicate>;
  p += i.offset();
  for (ptrdiff_t n = 0; n < N - pred; ++n)
    ret[n] = eve::load(p + W * n, eve::lane<W>);
  if constexpr (pred)
    ret[N - 1] = eve::load[eve::keep(i.p)](p + W * (N - 1), eve::lane<W>);
  return ret;
}

template <typename T, ptrdiff_t W, ptrdiff_t M, ptrdiff_t N, typename P>
constexpr auto load(const T *p, simd::TileOffset<W, M, N, P> i)
  -> std::array<std::array<eve::wide<T, W>, N>, M> {
  std::array<std::array<eve::wide<T, W>, N>, M> ret;
  p += i.offset();
  ptrdiff_t x = i.stride();
  simd::UnrollOffset<W, N, P> j{0, i.p};
  for (ptrdiff_t m = 0; m < M; ++m) ret[m] = load(p + x * m, j);
  return ret;
}

namespace simd {
template <typename T, ptrdiff_t W, ptrdiff_t N, typename P> struct UnrollRef {
  T *p;
  UnrollIndex<W, N, P> i;
  constexpr auto operator=(const std::array<eve::wide<T, W>, N> &v) const
    -> void {
    constexpr bool pred = !std::same_as<P, simd::NoPredicate>;
    T *q = p + i.offset();
    for (ptrdiff_t n = 0; n < N - pred; ++n) eve::store(v[n], q + n * W);
    if constexpr (pred) eve::store[eve::keep_first(i.p)](v[n], q + (N - 1) * W);
  }
  constexpr operator std::array<eve::wide<T, W>, N>() {
    return load(static_cast<const T *>(p), i);
  }
};
template <typename T, ptrdiff_t W, ptrdiff_t M, ptrdiff_t N, typename P>
struct TileRef {
  T *p;
  TileIndex<W, M, N, P> i;
  constexpr auto
  operator=(const std::array<std::array<eve::wide<T, W>, N>, M> &v) const
    -> void {
    constexpr bool pred = !std::same_as<P, simd::NoPredicate>;
    T *q = p + i.offset();
    for (ptrdiff_t m = 0; m < M; ++m) {
      for (ptrdiff_t n = 0; n < N - pred; ++n) eve::store(v[m][n], q + n * W);
      if constexpr (pred)
        eve::store[eve::keep_first(i.p)](v[m][n], q + (N - 1) * W);
      q += i.stride();
    }
  }
  constexpr operator std::array<std::array<eve::wide<T, W>, N>, M>() {
    return load(static_cast<const T *>(p), i);
  }
};
} // namespace simd

} // namespace poly::math
