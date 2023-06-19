#pragma once

#include "Math/AxisTypes.hpp"
#include "Math/Indexing.hpp"
#include "Math/MatrixDimensions.hpp"
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <eve/conditional.hpp>
#include <eve/module/core.hpp>
#include <eve/module/core/regular/gather.hpp>
#include <type_traits>

namespace poly::math {

template <typename T>
concept DefinesIsScalar = requires(T) {
  { std::remove_reference_t<T>::is_scalar };
};

template <typename T>
concept Scalar =
  std::integral<T> || std::floating_point<T> || DefinesIsScalar<T>;
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
  [[nodiscard]] constexpr auto offset() const -> ptrdiff_t { return this->i; }
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
  return {d.stride * calcOffset(d.len, i.i), i.p, d.stride};
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
template <class T> constexpr auto ref(T *p, ptrdiff_t i) -> T & { return p[i]; }
template <class T> constexpr auto ref(const T *p, ptrdiff_t i) -> const T & {
  return p[i];
}

template <Scalar T, ptrdiff_t W, ptrdiff_t N> struct Unrolled {
  std::array<eve::wide<T, eve::fixed<W>>, N> data;
  constexpr auto operator[](ptrdiff_t i) -> eve::wide<T, eve::fixed<W>> & {
    return data[i];
  }
  constexpr auto operator[](ptrdiff_t i) const
    -> const eve::wide<T, eve::fixed<W>> & {
    return data[i];
  }

  template <Scalar U> constexpr auto operator+=(const Unrolled<U, W, N> &b) {
#pragma GCC unroll 16
    for (ptrdiff_t i = 0; i < N; ++i) data[i] += b[i];
    return *this;
  }
  constexpr auto operator+=(Scalar auto b) {
#pragma GCC unroll 16
    for (ptrdiff_t i = 0; i < N; ++i) data[i] += b;
    return *this;
  }
  template <Scalar U> constexpr auto operator-=(const Unrolled<U, W, N> &b) {
#pragma GCC unroll 16
    for (ptrdiff_t i = 0; i < N; ++i) data[i] -= b[i];
    return *this;
  }
  constexpr auto operator-=(Scalar auto b) {
#pragma GCC unroll 16
    for (ptrdiff_t i = 0; i < N; ++i) data[i] -= b;
    return *this;
  }
  template <Scalar U> constexpr auto operator*=(const Unrolled<U, W, N> &b) {
#pragma GCC unroll 16
    for (ptrdiff_t i = 0; i < N; ++i) data[i] *= b[i];
    return *this;
  }
  constexpr auto operator*=(Scalar auto b) {
#pragma GCC unroll 16
    for (ptrdiff_t i = 0; i < N; ++i) data[i] *= b;
    return *this;
  }
  template <std::floating_point U>
  constexpr auto operator/=(const Unrolled<U, W, N> &b) {
    static_assert(std::floating_point<T>);
#pragma GCC unroll 16
    for (ptrdiff_t i = 0; i < N; ++i) data[i] /= b[i];
    return *this;
  }
  constexpr auto operator/=(std::floating_point auto b) {
    static_assert(std::floating_point<T>);
#pragma GCC unroll 16
    for (ptrdiff_t i = 0; i < N; ++i) data[i] /= b;
    return *this;
  }
};
template <Scalar T, ptrdiff_t W, ptrdiff_t M, ptrdiff_t N> struct Tiled {
  std::array<std::array<eve::wide<T, eve::fixed<W>>, N>, M> data;
  constexpr auto operator[](ptrdiff_t i)
    -> std::array<eve::wide<T, eve::fixed<W>>, N> & {
    return data[i];
  }
  constexpr auto operator[](ptrdiff_t i) const
    -> const std::array<eve::wide<T, eve::fixed<W>>, N> & {
    return data[i];
  }

  template <Scalar U> constexpr auto operator+=(const Tiled<U, W, M, N> &b) {
#pragma GCC unroll 16
    for (ptrdiff_t i = 0; i < M; ++i)
#pragma GCC unroll 16
      for (ptrdiff_t j = 0; j < N; ++j) data[i][j] += b[i][j];
    return *this;
  }
  constexpr auto operator+=(Scalar auto b) {
#pragma GCC unroll 16
    for (ptrdiff_t i = 0; i < M; ++i)
#pragma GCC unroll 16
      for (ptrdiff_t j = 0; j < N; ++j) data[i][j] += b;
    return *this;
  }
  template <Scalar U> constexpr auto operator-=(const Tiled<U, W, M, N> &b) {
#pragma GCC unroll 16
    for (ptrdiff_t i = 0; i < M; ++i)
#pragma GCC unroll 16
      for (ptrdiff_t j = 0; j < N; ++j) data[i][j] -= b[i][j];
    return *this;
  }
  constexpr auto operator-=(Scalar auto b) {
#pragma GCC unroll 16
    for (ptrdiff_t i = 0; i < M; ++i)
#pragma GCC unroll 16
      for (ptrdiff_t j = 0; j < N; ++j) data[i][j] -= b;
    return *this;
  }
  template <Scalar U> constexpr auto operator*=(const Tiled<U, W, M, N> &b) {
#pragma GCC unroll 16
    for (ptrdiff_t i = 0; i < M; ++i)
#pragma GCC unroll 16
      for (ptrdiff_t j = 0; j < N; ++j) data[i][j] *= b[i][j];
    return *this;
  }
  constexpr auto operator*=(Scalar auto b) {
#pragma GCC unroll 16
    for (ptrdiff_t i = 0; i < M; ++i)
#pragma GCC unroll 16
      for (ptrdiff_t j = 0; j < N; ++j) data[i][j] *= b;
    return *this;
  }
  template <std::floating_point U>
  constexpr auto operator/=(const Tiled<U, W, M, N> &b) {
    static_assert(std::floating_point<T>);
#pragma GCC unroll 16
    for (ptrdiff_t i = 0; i < M; ++i)
#pragma GCC unroll 16
      for (ptrdiff_t j = 0; j < N; ++j) data[i][j] /= b[i][j];
    return *this;
  }
  constexpr auto operator/=(std::floating_point auto b) {
    static_assert(std::floating_point<T>);
#pragma GCC unroll 16
    for (ptrdiff_t i = 0; i < M; ++i)
#pragma GCC unroll 16
      for (ptrdiff_t j = 0; j < N; ++j) data[i][j] /= b;
    return *this;
  }
};
template <Scalar T, Scalar U, ptrdiff_t W, ptrdiff_t N>
constexpr auto operator+(const Unrolled<T, W, N> &a,
                         const Unrolled<U, W, N> &b) {
  Unrolled<std::common_type_t<T, U>, W, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < N; ++i) ret[i] = a[i] + b[i];
  return ret;
}
template <Scalar T, ptrdiff_t W, ptrdiff_t N>
constexpr auto operator+(const Unrolled<T, W, N> &a, Scalar auto b) {
  Unrolled<T, W, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < N; ++i) ret[i] = a[i] + b;
  return ret;
}
template <Scalar T, ptrdiff_t W, ptrdiff_t N>
constexpr auto operator+(Scalar auto a, const Unrolled<T, W, N> &b) {
  Unrolled<T, W, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < N; ++i) ret[i] = a + b[i];
  return ret;
}
template <Scalar T, Scalar U, ptrdiff_t W, ptrdiff_t M, ptrdiff_t N>
constexpr auto operator+(const Tiled<T, W, M, N> &a,
                         const Tiled<U, W, M, N> &b) {
  Tiled<std::common_type_t<T, U>, W, M, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < M; ++i)
#pragma GCC unroll 16
    for (ptrdiff_t j = 0; j < N; ++j) ret[i][j] = a[i][j] + b[i][j];
  return ret;
}
template <Scalar T, ptrdiff_t W, ptrdiff_t M, ptrdiff_t N>
constexpr auto operator+(const Tiled<T, W, M, N> &a, Scalar auto b) {
  Tiled<T, W, M, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < M; ++i)
#pragma GCC unroll 16
    for (ptrdiff_t j = 0; j < N; ++j) ret[i][j] = a[i][j] + b;
  return ret;
}
template <Scalar T, ptrdiff_t W, ptrdiff_t M, ptrdiff_t N>
constexpr auto operator+(Scalar auto a, const Tiled<T, W, M, N> &b) {
  Tiled<T, W, M, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < M; ++i)
#pragma GCC unroll 16
    for (ptrdiff_t j = 0; j < N; ++j) ret[i][j] = a + b[i][j];
  return ret;
}
template <Scalar T, Scalar U, ptrdiff_t W, ptrdiff_t N>
constexpr auto operator-(const Unrolled<T, W, N> &a,
                         const Unrolled<U, W, N> &b) {
  Unrolled<std::common_type_t<T, U>, W, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < N; ++i) ret[i] = a[i] - b[i];
  return ret;
}
template <Scalar T, ptrdiff_t W, ptrdiff_t N>
constexpr auto operator-(const Unrolled<T, W, N> &a, Scalar auto b) {
  Unrolled<T, W, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < N; ++i) ret[i] = a[i] - b;
  return ret;
}
template <Scalar T, ptrdiff_t W, ptrdiff_t N>
constexpr auto operator-(Scalar auto a, const Unrolled<T, W, N> &b) {
  Unrolled<T, W, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < N; ++i) ret[i] = a - b[i];
  return ret;
}
template <Scalar T, ptrdiff_t W, ptrdiff_t N>
constexpr auto operator-(const Unrolled<T, W, N> &b) {
  Unrolled<T, W, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < N; ++i) ret[i] = -b[i];
  return ret;
}
template <Scalar T, Scalar U, ptrdiff_t W, ptrdiff_t M, ptrdiff_t N>
constexpr auto operator-(const Tiled<T, W, M, N> &a,
                         const Tiled<U, W, M, N> &b) {
  Tiled<std::common_type_t<T, U>, W, M, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < M; ++i)
#pragma GCC unroll 16
    for (ptrdiff_t j = 0; j < N; ++j) ret[i][j] = a[i][j] - b[i][j];
  return ret;
}
template <Scalar T, ptrdiff_t W, ptrdiff_t M, ptrdiff_t N>
constexpr auto operator-(const Tiled<T, W, M, N> &a, Scalar auto b) {
  Tiled<T, W, M, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < M; ++i)
#pragma GCC unroll 16
    for (ptrdiff_t j = 0; j < N; ++j) ret[i][j] = a[i][j] - b;
  return ret;
}
template <Scalar T, ptrdiff_t W, ptrdiff_t M, ptrdiff_t N>
constexpr auto operator-(Scalar auto a, const Tiled<T, W, M, N> &b) {
  Tiled<T, W, M, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < M; ++i)
#pragma GCC unroll 16
    for (ptrdiff_t j = 0; j < N; ++j) ret[i][j] = a - b[i][j];
  return ret;
}
template <Scalar T, ptrdiff_t W, ptrdiff_t M, ptrdiff_t N>
constexpr auto operator-(const Tiled<T, W, M, N> &b) {
  Tiled<T, W, M, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < M; ++i)
#pragma GCC unroll 16
    for (ptrdiff_t j = 0; j < N; ++j) ret[i][j] = -b[i][j];
  return ret;
}
template <Scalar T, Scalar U, ptrdiff_t W, ptrdiff_t N>
constexpr auto operator*(const Unrolled<T, W, N> &a,
                         const Unrolled<U, W, N> &b) {
  Unrolled<std::common_type_t<T, U>, W, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < N; ++i) ret[i] = a[i] * b[i];
  return ret;
}
template <Scalar T, ptrdiff_t W, ptrdiff_t N>
constexpr auto operator*(const Unrolled<T, W, N> &a, Scalar auto b) {
  Unrolled<T, W, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < N; ++i) ret[i] = a[i] * b;
  return ret;
}
template <Scalar T, ptrdiff_t W, ptrdiff_t N>
constexpr auto operator*(Scalar auto a, const Unrolled<T, W, N> &b) {
  Unrolled<T, W, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < N; ++i) ret[i] = a * b[i];
  return ret;
}
template <Scalar T, Scalar U, ptrdiff_t W, ptrdiff_t M, ptrdiff_t N>
constexpr auto operator*(const Tiled<T, W, M, N> &a,
                         const Tiled<U, W, M, N> &b) {
  Tiled<std::common_type_t<T, U>, W, M, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < M; ++i)
#pragma GCC unroll 16
    for (ptrdiff_t j = 0; j < N; ++j) ret[i][j] = a[i][j] * b[i][j];
  return ret;
}
template <Scalar T, ptrdiff_t W, ptrdiff_t M, ptrdiff_t N>
constexpr auto operator*(const Tiled<T, W, M, N> &a, Scalar auto b) {
  Tiled<T, W, M, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < M; ++i)
#pragma GCC unroll 16
    for (ptrdiff_t j = 0; j < N; ++j) ret[i][j] = a[i][j] * b;
  return ret;
}
template <Scalar T, ptrdiff_t W, ptrdiff_t M, ptrdiff_t N>
constexpr auto operator*(Scalar auto a, const Tiled<T, W, M, N> &b) {
  Tiled<T, W, M, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < M; ++i)
#pragma GCC unroll 16
    for (ptrdiff_t j = 0; j < N; ++j) ret[i][j] = a * b[i][j];
  return ret;
}
template <std::floating_point T, std::floating_point U, ptrdiff_t W,
          ptrdiff_t N>
constexpr auto operator/(const Unrolled<T, W, N> &a,
                         const Unrolled<U, W, N> &b) {
  Unrolled<std::common_type_t<T, U>, W, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < N; ++i) ret[i] = a[i] / b[i];
  return ret;
}
template <std::floating_point T, ptrdiff_t W, ptrdiff_t N>
constexpr auto operator/(const Unrolled<T, W, N> &a,
                         std::floating_point auto b) {
  Unrolled<T, W, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < N; ++i) ret[i] = a[i] / b;
  return ret;
}
template <std::floating_point T, ptrdiff_t W, ptrdiff_t N>
constexpr auto operator/(std::floating_point auto a,
                         const Unrolled<T, W, N> &b) {
  Unrolled<T, W, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < N; ++i) ret[i] = a / b[i];
  return ret;
}
template <std::floating_point T, std::floating_point U, ptrdiff_t W,
          ptrdiff_t M, ptrdiff_t N>
constexpr auto operator/(const Tiled<T, W, M, N> &a,
                         const Tiled<U, W, M, N> &b) {
  Tiled<std::common_type_t<T, U>, W, M, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < M; ++i)
#pragma GCC unroll 16
    for (ptrdiff_t j = 0; j < N; ++j) ret[i][j] = a[i][j] / b[i][j];
  return ret;
}
template <std::floating_point T, ptrdiff_t W, ptrdiff_t M, ptrdiff_t N>
constexpr auto operator/(const Tiled<T, W, M, N> &a,
                         std::floating_point auto b) {
  Tiled<T, W, M, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < M; ++i)
#pragma GCC unroll 16
    for (ptrdiff_t j = 0; j < N; ++j) ret[i][j] = a[i][j] / b;
  return ret;
}
template <std::floating_point T, ptrdiff_t W, ptrdiff_t M, ptrdiff_t N>
constexpr auto operator/(std::floating_point auto a,
                         const Tiled<T, W, M, N> &b) {
  Tiled<T, W, M, N> ret;
#pragma GCC unroll 16
  for (ptrdiff_t i = 0; i < M; ++i)
#pragma GCC unroll 16
    for (ptrdiff_t j = 0; j < N; ++j) ret[i][j] = a / b[i][j];
  return ret;
}

template <Scalar T, ptrdiff_t W, ptrdiff_t N, typename P>
constexpr auto ref(const T *p, simd::UnrollOffset<W, N, P> i)
  -> Unrolled<T, W, N> {
  Unrolled<T, W, N> ret;
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
template <Scalar T, ptrdiff_t W, ptrdiff_t N, typename P>
constexpr auto ref(const T *p, simd::StridedUnrollOffset<W, N, P> i)
  -> Unrolled<T, W, N> {
  Unrolled<T, W, N> ret;
  constexpr bool pred = !std::same_as<P, simd::NoPredicate>;
  constexpr ptrdiff_t NN = N - pred;
  p += i.offset();
  using I = std::conditional_t<sizeof(T) >= 8, ptrdiff_t, int32_t>;
  I stride = i.stride;
  eve::wide<std::int32_t, eve::fixed<W>> s{[](auto j, auto) { return j; }};
  s *= stride;
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=102855
#pragma GCC unroll 16
  for (ptrdiff_t n = 0; n < NN; ++n) ret[n] = eve::gather(p + W * n, s);
  if constexpr (pred)
    ret[NN] = eve::gather[eve::keep_first(i.p)](p + W * NN, s);
  return ret;
}

template <Scalar T, ptrdiff_t W, ptrdiff_t M, ptrdiff_t N, typename P>
constexpr auto ref(const T *p, simd::TileOffset<W, M, N, P> i)
  -> Tiled<T, W, M, N> {
  Tiled<T, W, M, N> ret;
  p += i.offset();
  ptrdiff_t x = i.stride();
  simd::UnrollOffset<W, N, P> j{0, i.p};
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=102855
#pragma GCC unroll 16
  for (ptrdiff_t m = 0; m < M; ++m) ret[m] = ref(p + x * m, j);
  return ret;
}

namespace simd {
template <Scalar T, ptrdiff_t W, ptrdiff_t N, typename P> struct UnrollRef {
  T *p;
  UnrollOffset<W, N, P> i;
  constexpr auto operator=(const Unrolled<T, W, N> &v) -> UnrollRef & {
    constexpr bool pred = !std::same_as<P, simd::NoPredicate>;
    constexpr ptrdiff_t NN = N - pred;
    T *q = p + i.offset();
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=102855
#pragma GCC unroll 16
    for (ptrdiff_t n = 0; n < NN; ++n) eve::store(v[n], q + n * W);
    if constexpr (pred) eve::store[eve::keep_first(i.p)](v[NN], q + NN * W);
    return *this;
  }
  constexpr operator Unrolled<T, W, N>() {
    return ref(static_cast<const T *>(p), i);
  }
};
template <Scalar T, ptrdiff_t W, ptrdiff_t N, typename P>
struct StridedUnrollRef {
  T *p;
  StridedUnrollOffset<W, N, P> i;
  constexpr auto operator=(const Unrolled<T, W, N> &v) -> StridedUnrollRef & {
    std::array<eve::wide<T, eve::fixed<W>>, N> ret;
    constexpr bool pred = !std::same_as<P, simd::NoPredicate>;
    constexpr ptrdiff_t NN = N - pred;
    p += i.offset();
    // using I = std::conditional_t<sizeof(T) >= 8, ptrdiff_t, int32_t>;
    // stride = i.stride;
    // eve::wide<std::int32_t, eve::fixed<W>> s{[](auto i, auto) { return i; }};
    // s *= stride;

    for (ptrdiff_t n = 0; n < NN; ++n) {
      eve::wide<T, eve::fixed<W>> vn = v[n];
      for (ptrdiff_t w = 0; w < W; ++w)
        p[n * W * i.stride + w * i.stride] = vn.get(w);
    }
    if constexpr (pred) {
      eve::wide<T, eve::fixed<W>> vn = v[NN];
      for (ptrdiff_t w = 0; w < i.p; ++w)
        p[NN * W * i.stride + w * i.stride] = vn.get(w);
    }
    return *this;
  }
  constexpr operator Unrolled<T, W, N>() {
    return ref(static_cast<const T *>(p), i);
  }
};
template <Scalar T, ptrdiff_t W, ptrdiff_t M, ptrdiff_t N, typename P>
struct TileRef {
  T *p;
  TileOffset<W, M, N, P> i;
  constexpr auto operator=(const Tiled<T, W, M, N> &v) -> TileRef & {
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
  constexpr operator Tiled<T, W, M, N>() {
    return ref(static_cast<const T *>(p), i);
  }
};
} // namespace simd
template <Scalar T, ptrdiff_t W, ptrdiff_t N, typename P>
constexpr auto ref(T *p, simd::UnrollOffset<W, N, P> i)
  -> simd::UnrollRef<T, W, N, P> {
  return {p, i};
}
template <Scalar T, ptrdiff_t W, ptrdiff_t N, typename P>
constexpr auto ref(T *p, simd::StridedUnrollOffset<W, N, P> i)
  -> simd::StridedUnrollRef<T, W, N, P> {
  return {p, i};
}

template <Scalar T, ptrdiff_t W, ptrdiff_t M, ptrdiff_t N, typename P>
constexpr auto ref(const T *p, simd::TileOffset<W, M, N, P> i)
  -> simd::TileRef<T, W, M, N, P> {
  return {p, i};
}
} // namespace poly::math
