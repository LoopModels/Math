#pragma once

#include "Math/AxisTypes.hpp"
#include "Math/Indexing.hpp"
#include "Math/MatrixDimensions.hpp"
#include <array>
#include <concepts>
#include <cstddef>
#include <eve/conditional.hpp>
#include <eve/module/core.hpp>

namespace poly::math {
namespace simd {

template <ptrdiff_t W, ptrdiff_t N, typename P> struct Unroll {
  [[no_unique_address]] ptrdiff_t i;
  [[no_unique_address]] P p;
  constexpr explicit operator ptrdiff_t() const { return i; }
};
template <ptrdiff_t W, ptrdiff_t N>
constexpr auto unroll(ptrdiff_t i) -> Unroll<W, N, NoPredicate> {
  return {i, NoPredicate{}};
}
template <ptrdiff_t W, ptrdiff_t N>
constexpr auto unroll(ptrdiff_t i, ptrdiff_t L) -> Unroll<W, N, ptrdiff_t> {
  return {i, L - i};
}
template <ptrdiff_t W, ptrdiff_t M, ptrdiff_t N, typename P> struct Tile {
  [[no_unique_address]] ptrdiff_t row, col;
  [[no_unique_address]] P p;
  constexpr explicit operator CartesianIndex<ptrdiff_t, ptrdiff_t>() const {
    return {row, col};
  }
};
template <ptrdiff_t W, ptrdiff_t M, ptrdiff_t N>
constexpr auto tile(ptrdiff_t i, ptrdiff_t j) -> Tile<W, M, N, NoPredicate> {
  return {i, j, NoPredicate{}};
}
template <ptrdiff_t W, ptrdiff_t M, ptrdiff_t N>
constexpr auto tile(ptrdiff_t i, ptrdiff_t j, ptrdiff_t L)
  -> Tile<W, M, N, ptrdiff_t> {
  return {i, j, L - j};
}

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

} // namespace simd

template <ptrdiff_t W, ptrdiff_t N, typename P>
[[nodiscard]] inline constexpr auto calcOffset(ptrdiff_t len,
                                               simd::Unroll<W, N, P> i)
  -> simd::UnrollOffset<W, N, P> {
  invariant((ptrdiff_t(i) + W * N - ptrdiff_t(i.p)) <= len);
  return {i.i, i.p};
}
template <ptrdiff_t W, ptrdiff_t N, typename P>
constexpr auto calcOffset(StridedRange d, simd::Unroll<W, N, P> i)
  -> simd::StridedUnrollOffset<W, N, P> {
  return {d.stride * calcOffset(d.len, i), i.p, d.stride};
}

template <ptrdiff_t W, ptrdiff_t M, ptrdiff_t N, typename P>
[[nodiscard]] inline constexpr auto calcOffset(StridedDims d,
                                               simd::Tile<W, M, N, P> i)
  -> simd::TileOffset<W, M, N, P> {
  invariant((ptrdiff_t(i.col) + N * W - ptrdiff_t(i.p)) <= ptrdiff_t(Col{d}));
  invariant((ptrdiff_t(i.row) + M) <= ptrdiff_t(Row{d}));
  ptrdiff_t x = ptrdiff_t(RowStride{d});
  ptrdiff_t r = x * calcOffset(ptrdiff_t(Row{d}), i.row);
  ptrdiff_t c = calcOffset(ptrdiff_t(Col{d}), i.col);
  return {r + c, x, i.p};
}

template <ptrdiff_t W, ptrdiff_t N, typename P>
constexpr auto calcNewDim(VectorDimension auto, simd::Unroll<W, N, P>)
  -> Empty {
  return {};
}
template <ptrdiff_t W, ptrdiff_t M, ptrdiff_t N, typename P>
constexpr auto calcNewDim(MatrixDimension auto, simd::Tile<W, M, N, P>)
  -> Empty {
  return {};
}
template <typename T> constexpr auto ref(T *p, ptrdiff_t i) -> T & {
  return p[i];
}
template <typename T> constexpr auto ref(const T *p, ptrdiff_t i) -> const T & {
  return p[i];
}

template <typename T, ptrdiff_t W, ptrdiff_t N, typename P>
constexpr auto ref(const T *p, simd::UnrollOffset<W, N, P> i)
  -> std::array<eve::wide<T, eve::fixed<W>>, N> {
  std::array<eve::wide<T, eve::fixed<W>>, N> ret;
  constexpr bool pred = !std::same_as<P, simd::NoPredicate>;
  constexpr ptrdiff_t NN = N - pred;
  p += i.offset();
  // GCC doesn't support unrolling based on a template param, but clang does
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=102855
  // we can use `#pragma GCC unroll NN` when this is fixed
#pragma GCC unroll 16
  for (ptrdiff_t n = 0; n < NN; ++n)
    ret[n] = eve::load(p + W * n, eve::lane<W>);
  if constexpr (pred)
    ret[NN] = eve::load[eve::keep_first(i.p)](p + W * NN, eve::lane<W>);
  return ret;
}

template <typename T, ptrdiff_t W, ptrdiff_t M, ptrdiff_t N, typename P>
constexpr auto ref(const T *p, simd::TileOffset<W, M, N, P> i)
  -> std::array<std::array<eve::wide<T, eve::fixed<W>>, N>, M> {
  std::array<std::array<eve::wide<T, eve::fixed<W>>, N>, M> ret;
  p += i.offset();
  ptrdiff_t x = i.stride();
  simd::UnrollOffset<W, N, P> j{0, i.p};
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=102855
#pragma GCC unroll 16
  for (ptrdiff_t m = 0; m < M; ++m) ret[m] = ref(p + x * m, j);
  return ret;
}

namespace simd {
template <typename T, ptrdiff_t W, ptrdiff_t N, typename P> struct UnrollRef {
  T *p;
  UnrollOffset<W, N, P> i;
  constexpr auto
  operator=(const std::array<eve::wide<T, eve::fixed<W>>, N> &v) const
    -> UnrollRef & {
    constexpr bool pred = !std::same_as<P, simd::NoPredicate>;
    constexpr ptrdiff_t NN = N - pred;
    T *q = p + i.offset();
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=102855
#pragma GCC unroll 16
    for (ptrdiff_t n = 0; n < NN; ++n) eve::store(v[n], q + n * W);
    if constexpr (pred) eve::store[eve::keep_first(i.p)](v[NN], q + NN * W);
    return *this;
  }
  constexpr operator std::array<eve::wide<T, eve::fixed<W>>, N>() {
    return ref(static_cast<const T *>(p), i);
  }
};
template <typename T, ptrdiff_t W, ptrdiff_t M, ptrdiff_t N, typename P>
struct TileRef {
  T *p;
  TileOffset<W, M, N, P> i;
  constexpr auto operator=(
    const std::array<std::array<eve::wide<T, eve::fixed<W>>, N>, M> &v) const
    -> TileRef & {
    constexpr bool pred = !std::same_as<P, simd::NoPredicate>;
    constexpr ptrdiff_t NN = N - pred;
    T *q = p + i.offset();
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=102855
#pragma GCC unroll 16
    for (ptrdiff_t m = 0; m < M; ++m) {
#pragma GCC unroll 16
      for (ptrdiff_t n = 0; n < NN; ++n) eve::store(v[m][n], q + n * W);
      if constexpr (pred)
        eve::store[eve::keep_first(i.p)](v[m][NN], q + NN * W);
      q += i.stride();
    }
    return *this;
  }
  constexpr
  operator std::array<std::array<eve::wide<T, eve::fixed<W>>, N>, M>() {
    return ref(static_cast<const T *>(p), i);
  }
};
} // namespace simd
template <typename T, ptrdiff_t W, ptrdiff_t N, typename P>
constexpr auto ref(T *p, simd::UnrollOffset<W, N, P> i)
  -> simd::UnrollRef<T, W, N, P> {
  return {p, i};
}

template <typename T, ptrdiff_t W, ptrdiff_t M, ptrdiff_t N, typename P>
constexpr auto ref(const T *p, simd::TileOffset<W, M, N, P> i)
  -> simd::TileRef<T, W, M, N, P> {
  return {p, i};
}
} // namespace poly::math
