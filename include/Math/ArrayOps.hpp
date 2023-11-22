#pragma once

#include "Containers/Tuple.hpp"
#include "Math/Indexing.hpp"
#include "Math/Matrix.hpp"
#include "Math/UniformScaling.hpp"
#include "SIMD/Unroll.hpp"
#include "Utilities/Assign.hpp"
#include "Utilities/LoopMacros.hpp"
#include <algorithm>
#include <cstring>
#include <type_traits>

namespace poly::math {
// scalars broadcast
template <typename S>
[[gnu::always_inline]] constexpr auto get(const auto &s, auto) {
  return s;
}
template <typename S>
[[gnu::always_inline]] constexpr auto get(const auto &s, auto, auto) {
  return s;
}
template <typename S, LinearlyIndexable<S> V>
[[gnu::always_inline]] constexpr auto get(const V &v, auto i) {
  return v[i];
}
template <typename S, CartesianIndexable<S> V>
[[gnu::always_inline]] constexpr auto get(const V &v, auto i, auto j) {
  return v[i, j];
}
template <typename T, typename S>
concept OnlyLinearlyIndexable = LinearlyIndexable<S> && !CartesianIndexable<S>;
template <typename S, OnlyLinearlyIndexable<S> V>
[[gnu::always_inline]] constexpr auto get(const V &v, auto i, auto j) {
  static_assert(AbstractVector<V>);
  if constexpr (RowVector<V>) return v[j];
  else return v[i];
}

// returns Unroll, Iters, Remainder
template <ptrdiff_t R> consteval auto unrollf() -> std::array<ptrdiff_t, 2> {
  if (R <= 5) return {0, R};
  if (R == 7) return {0, 7};
  if ((R % 4) == 0) return {4, 0};
  if ((R % 3) == 0) return {3, 0};
  if ((R % 5) == 0) return {5, 0};
  if ((R % 7) == 0) return {7, 0};
  return {4, R % 4};
}

template <typename D, typename S>
constexpr void fastCopy(D *d, const S *s, size_t N) {
  if (!N) return;
  if constexpr (std::same_as<D, S> && std::is_trivially_copyable_v<D>)
    std::memcpy(d, s, N * sizeof(D));
  else std::copy_n(s, N, d);
}

template <typename T>
concept IsOne =
  std::same_as<std::remove_cvref_t<T>, std::integral_constant<ptrdiff_t, 1>>;
// inputs must be `ptrdiff_t` or `std::integral_constant<ptrdiff_t,value>`
template <typename X, typename Y>
[[gnu::always_inline]] constexpr auto check_sizes(X x, Y y) {
  if constexpr (std::same_as<ptrdiff_t, X>) {
    if constexpr (std::same_as<ptrdiff_t, Y>) {
      invariant(x, y);
      return x;
    } else {
      constexpr ptrdiff_t L = y;
      invariant(x, L);
      return std::integral_constant<ptrdiff_t, L>{};
    }
  } else if constexpr (std::same_as<ptrdiff_t, Y>) {
    constexpr ptrdiff_t L = x;
    invariant(L, y);
    return std::integral_constant<ptrdiff_t, L>{};
  } else {
    static_assert(x == y);
    return std::integral_constant<ptrdiff_t, ptrdiff_t(x)>{};
  }
}
template <typename A, typename B>
[[gnu::always_inline]] constexpr auto promote_shape(const A &a, const B &b) {
  auto sa = shape(a);
  if constexpr (!std::convertible_to<B, utils::eltype_t<A>>) {
    auto M = unwrapRow(numRows(b));
    auto N = unwrapCol(numCols(b));
    if constexpr (IsOne<decltype(M)>)
      if constexpr (IsOne<decltype(N)>) return sa;
      else return CartesianIndex(sa.row, check_sizes(sa.col, N));
    else if constexpr (IsOne<decltype(N)>)
      return CartesianIndex(check_sizes(sa.row, M), sa.col);
    else return CartesianIndex(check_sizes(sa.row, M), check_sizes(sa.col, N));
  } else return sa;
}

template <typename T> class SmallSparseMatrix;
template <class T, class S, class P> class ArrayOps {

  static_assert(std::is_copy_assignable_v<T> ||
                (std::is_trivially_copyable_v<T> &&
                 std::is_trivially_move_assignable_v<T>));
  [[gnu::returns_nonnull]] constexpr auto data_() -> T * {
    return static_cast<P *>(this)->data();
  }
  [[gnu::returns_nonnull]] constexpr auto data_() const -> const T * {
    return static_cast<const P *>(this)->data();
  }
  constexpr auto size_() const { return static_cast<const P *>(this)->size(); }
  constexpr auto Self() -> P & { return *static_cast<P *>(this); }
  [[nodiscard]] constexpr auto nr() const -> ptrdiff_t {
    return ptrdiff_t(static_cast<const P *>(this)->numRow());
  }
  [[nodiscard]] constexpr auto nc() const {
    return unwrapCol(static_cast<const P *>(this)->numCol());
  }
  [[nodiscard]] constexpr auto rs() const {
    return unwrapRow(static_cast<const P *>(this)->rowStride());
  }
  template <typename I, typename R, typename Op>
  [[gnu::always_inline]] static void vcopyToSIMD(P &self, const auto &B, I L,
                                                 R row, Op op) {
    // TODO: if `R` is a row index, maybe don't fully unroll static `L`
    // We're going for very short SIMD vectors to focus on small sizes
    if constexpr (StaticInt<I>) {
      constexpr ptrdiff_t SL = ptrdiff_t(L);
      constexpr std::array<ptrdiff_t, 3> vdr = simd::VectorDivRem<SL, T>();
      constexpr ptrdiff_t W = vdr[0];
      constexpr ptrdiff_t fulliter = vdr[1];
      constexpr ptrdiff_t remainder = vdr[2];
      if constexpr (remainder > 0) {
        auto u{simd::index::unrollmask<fulliter + 1, W>(L, 0)};
        utils::assign(self, B, row, u, op);
      } else {
        simd::index::Unroll<fulliter, W> u{0};
        utils::assign(self, B, row, u, op);
      }
    } else {
      constexpr ptrdiff_t W = simd::Width<T>;
      for (ptrdiff_t i = 0;; i += W) {
        auto u{simd::index::unrollmask<1, W>(L, i)};
        if (!u) break;
        utils::assign(self, B, row, u, op);
      }
    }
  }

  template <typename Op> void vcopyTo(const auto &B, Op op) {
    static_assert(sizeof(utils::eltype_t<decltype(B)>) <= 8);
    P &self{Self()};
    auto [M, N] = promote_shape(self, B);
    if constexpr (std::same_as<Op, utils::CopyAssign> && DenseLayout<S> &&
                  DenseTensor<std::remove_cvref_t<decltype(B)>>) {
      if constexpr (std::is_trivially_copyable_v<T>)
        std::memcpy(data_(), B.begin(), M * N * sizeof(T));
      else std::copy_n(B.begin(), M * N, data_());
    } else if constexpr (simd::SIMDSupported<T>) {
      if constexpr (IsOne<decltype(M)>)
        vcopyToSIMD(self, B, N, utils::NoRowIndex{}, op);
      else if constexpr (IsOne<decltype(N)>)
        vcopyToSIMD(self, B, M, utils::NoRowIndex{}, op);
      else if constexpr (StaticInt<decltype(M)>) {
        constexpr std::array<ptrdiff_t, 2> UIR = unrollf<ptrdiff_t(M)>();
        constexpr ptrdiff_t U = UIR[0];
        if constexpr (U != 0)
          for (ptrdiff_t r = 0; r < (M - U + 1); r += U)
            vcopyToSIMD(self, B, N, simd::index::Unroll<U>{r}, op);
        constexpr ptrdiff_t R = UIR[1];
        if constexpr (R != 0)
          vcopyToSIMD(self, B, N, simd::index::Unroll<R>{M - R}, op);
      } else {
        ptrdiff_t r = 0;
        for (; r < (M - 3); r += 4)
          vcopyToSIMD(self, B, N, simd::index::Unroll<4>{r}, op);
        switch (M & 3) {
        case 0: return;
        case 1: return vcopyToSIMD(self, B, N, simd::index::Unroll<1>{r}, op);
        case 2: return vcopyToSIMD(self, B, N, simd::index::Unroll<2>{r}, op);
        default: return vcopyToSIMD(self, B, N, simd::index::Unroll<3>{r}, op);
        }
      }
    } else if constexpr (AbstractVector<P>) {
      ptrdiff_t L = IsOne<decltype(N)> ? M : N;
      if constexpr (!std::is_copy_assignable_v<T> &&
                    std::same_as<Op, utils::CopyAssign>) {
        POLYMATHVECTORIZE
        for (ptrdiff_t j = 0; j < L; ++j)
          if constexpr (std::convertible_to<decltype(B), T>) self[j] = auto{B};
          else self[j] = auto{B[j]};
      } else {
        POLYMATHVECTORIZE
        for (ptrdiff_t j = 0; j < L; ++j)
          utils::assign(self, B, utils::NoRowIndex{}, j, op);
      }
    } else {
      ptrdiff_t R = ptrdiff_t(M), C = ptrdiff_t(N);
      POLYMATHNOVECTORIZE
      for (ptrdiff_t i = 0; i < R; ++i) {
        if constexpr (!std::is_copy_assignable_v<T> &&
                      std::same_as<Op, utils::CopyAssign>) {
          POLYMATHVECTORIZE
          for (ptrdiff_t j = 0; j < C; ++j)
            if constexpr (std::convertible_to<decltype(B), T>)
              self[i, j] = auto{B};
            else if constexpr (RowVector<decltype(B)>) self[i, j] = auto{B[j]};
            else if constexpr (ColVector<decltype(B)>) self[i, j] = auto{B[i]};
            else self[i, j] = auto{B[i, j]};
        } else {
          POLYMATHVECTORIZE
          for (ptrdiff_t j = 0; j < C; ++j) utils::assign(self, B, i, j, op);
        }
      }
    }
  }
  template <typename Op> constexpr void scopyTo(const auto &B, Op op) {
    P &self{Self()};
    auto [M, N] = promote_shape(self, B);
    if constexpr (std::same_as<Op, utils::CopyAssign> && DenseLayout<S> &&
                  DenseTensor<std::remove_cvref_t<decltype(B)>>) {
      fastCopy(data_(), B.begin(), M * N);
    } else if constexpr (AbstractVector<P>) {
      ptrdiff_t L = IsOne<decltype(N)> ? M : N;
      if constexpr (!std::is_copy_assignable_v<T> &&
                    std::same_as<Op, utils::CopyAssign>) {
        POLYMATHIVDEP
        for (ptrdiff_t j = 0; j < L; ++j)
          if constexpr (std::convertible_to<decltype(B), T>) self[j] = auto{B};
          else self[j] = auto{B[j]};
      } else {
        // POLYMATHIVDEP
        for (ptrdiff_t j = 0; j < L; ++j)
          utils::assign(self, B, utils::NoRowIndex{}, j, op);
      }
    } else {
      POLYMATHNOVECTORIZE
      for (ptrdiff_t i = 0; i < M; ++i) {
        if constexpr (!std::is_copy_assignable_v<T> &&
                      std::same_as<Op, utils::CopyAssign>) {
          POLYMATHIVDEP
          for (ptrdiff_t j = 0; j < N; ++j)
            if constexpr (std::convertible_to<decltype(B), T>)
              self[i, j] = auto{B};
            else if constexpr (RowVector<decltype(B)>) self[i, j] = auto{B[j]};
            else if constexpr (ColVector<decltype(B)>) self[i, j] = auto{B[i]};
            else self[i, j] = auto{B[i, j]};
        } else {
          POLYMATHIVDEP
          for (ptrdiff_t j = 0; j < N; ++j) utils::assign(self, B, i, j, op);
        }
      }
    }
  }

  template <typename Op> constexpr void copyTo(const auto &B, Op op) {
    if consteval {
      scopyTo(B, op);
    } else {
      if constexpr (sizeof(T) <= sizeof(double)) vcopyTo(B, op);
      else scopyTo(B, op);
    }
  }

public:
  template <std::convertible_to<T> Y>
  [[gnu::flatten]] constexpr auto operator<<(const UniformScaling<Y> &B)
    -> P & {
    static_assert(MatrixDimension<S>);
    std::fill_n(data_(), ptrdiff_t(this->dim()), T{});
    this->diag() << B.value;
    return *static_cast<P *>(this);
  }
  [[gnu::flatten]] constexpr auto operator<<(const SmallSparseMatrix<T> &B)
    -> P &;

  [[gnu::flatten]] constexpr auto operator<<(const auto &B) -> P & {
    copyTo(B, utils::CopyAssign{});
    return Self();
  }

  [[gnu::flatten]] constexpr auto operator+=(const auto &B) -> P & {
    copyTo(B, std::plus<>{});
    return Self();
  }
  [[gnu::flatten]] constexpr auto operator-=(const auto &B) -> P & {
    copyTo(B, std::minus<>{});
    return Self();
  }
  [[gnu::flatten]] constexpr auto operator*=(const auto &B) -> P & {
    copyTo(B, std::multiplies<>{});
    return Self();
  }
  [[gnu::flatten]] constexpr auto operator/=(const auto &B) -> P & {
    copyTo(B, std::divides<>{});
    return Self();
  }
};

} // namespace poly::math

namespace poly::containers {

namespace tupletensorops {
// FIXME:
// Need to do all loads before all stores!!!
// This is because we want to support fusing loops where we may be overwriting
// inputs, e.g. `tie(x,y) << Tuple(x - y, x + y);`
template <typename A, typename... As, typename B, typename... Bs, typename I,
          typename R>
[[gnu::always_inline]] static void
vcopyToSIMD(Tuple<A, As...> &dst, const Tuple<B, Bs...> &src, I L, R row) {
  // TODO: if `R` is a row index, maybe don't fully unroll static `L`
  // We're going for very short SIMD vectors to focus on small sizes
  using T = std::common_type<utils::eltype_t<A>, utils::eltype_t<As>...,
                             utils::eltype_t<B>, utils::eltype_t<Bs>...>;
  if constexpr (math::StaticInt<I>) {
    constexpr ptrdiff_t SL = ptrdiff_t(L);
    constexpr std::array<ptrdiff_t, 3> vdr = simd::VectorDivRem<SL, T>();
    constexpr ptrdiff_t W = vdr[0];
    constexpr ptrdiff_t fulliter = vdr[1];
    constexpr ptrdiff_t remainder = vdr[2];
    if constexpr (remainder > 0) {
      auto u{simd::index::unrollmask<fulliter + 1, W>(L, 0)};
      if constexpr (std::same_as<R, utils::NoRowIndex>)
        dst.apply(src.map([=](const auto &s) { return get<T>(s, u); }),
                  [=](auto &d, const auto &s) { d[u] = s; });
      else
        dst.apply(src.map([=](const auto &s) { return get<T>(s, row, u); }),
                  [=](auto &d, const auto &s) { d[row, u] = s; });
    } else if constexpr (std::same_as<R, utils::NoRowIndex>) {
      simd::index::Unroll<fulliter, W> u{0};
      dst.apply(src.map([=](const auto &s) { return get<T>(s, u); }),
                [=](auto &d, const auto &s) { d[u] = s; });
    } else {
      simd::index::Unroll<fulliter, W> u{0};
      dst.apply(src.map([=](const auto &s) { return get<T>(s, row, u); }),
                [=](auto &d, const auto &s) { d[row, u] = s; });
    }
  } else {
    constexpr ptrdiff_t W = simd::Width<T>;
    for (ptrdiff_t i = 0;; i += W) {
      auto u{simd::index::unrollmask<1, W>(L, i)};
      if (!u) break;
      if constexpr (std::same_as<R, utils::NoRowIndex>)
        dst.apply(src.map([=](const auto &s) { return get<T>(s, u); }),
                  [=](auto &d, const auto &s) { d[u] = s; });
      else
        dst.apply(src.map([=](const auto &s) { return get<T>(s, row, u); }),
                  [=](auto &d, const auto &s) { d[row, u] = s; });
    }
  }
}

template <typename A, typename B>
[[gnu::always_inline]] constexpr auto promote_shape(const Tuple<A> &a,
                                                    const Tuple<B> &b) {
  return math::promote_shape(a.head, b.head);
}
template <typename A, typename... As, typename B, typename... Bs>
[[gnu::always_inline]] constexpr auto promote_shape(const Tuple<A, As...> &a,
                                                    const Tuple<B, Bs...> &b) {
  auto [Mh, Nh] = math::promote_shape(a.head, b.head);
  auto [Mt, Nt] = promote_shape(a.tail, b.tail);
  return math::CartesianIndex(math::check_sizes(Mh, Mt),
                              math::check_sizes(Nh, Nt));
}
template <typename A, typename... As, typename B, typename... Bs>
void vcopyTo(Tuple<A, As...> &dst, const Tuple<B, Bs...> &src) {
  using T = std::common_type<utils::eltype_t<A>, utils::eltype_t<As>...,
                             utils::eltype_t<B>, utils::eltype_t<Bs>...>;
  static_assert(sizeof(T) <= 8);
  auto [M, N] = promote_shape(dst, src);
  if constexpr (simd::SIMDSupported<T>) {
    if constexpr (math::IsOne<decltype(M)>)
      vcopyToSIMD(dst, src, N, utils::NoRowIndex{});
    else if constexpr (math::IsOne<decltype(N)>)
      vcopyToSIMD(dst, src, M, utils::NoRowIndex{});
    else if constexpr (math::StaticInt<decltype(M)>) {
      constexpr std::array<ptrdiff_t, 2> UIR = math::unrollf<ptrdiff_t(M)>();
      constexpr ptrdiff_t U = UIR[0];
      if constexpr (U != 0)
        for (ptrdiff_t r = 0; r < (M - U + 1); r += U)
          vcopyToSIMD(dst, src, N, simd::index::Unroll<U>{r});
      constexpr ptrdiff_t R = UIR[1];
      if constexpr (R != 0)
        vcopyToSIMD(dst, src, N, simd::index::Unroll<R>{M - R});
    } else {
      ptrdiff_t r = 0;
      for (; r < (M - 3); r += 4)
        vcopyToSIMD(dst, src, N, simd::index::Unroll<4>{r});
      switch (M & 3) {
      case 0: return;
      case 1: return vcopyToSIMD(dst, src, N, simd::index::Unroll<1>{r});
      case 2: return vcopyToSIMD(dst, src, N, simd::index::Unroll<2>{r});
      default: return vcopyToSIMD(dst, src, N, simd::index::Unroll<3>{r});
      }
    }
  } else if constexpr (math::AbstractVector<A>) {
    ptrdiff_t L = math::IsOne<decltype(N)> ? M : N;
    if constexpr (!std::is_copy_assignable_v<T>) {
      POLYMATHVECTORIZE
      for (ptrdiff_t j = 0; j < L; ++j)
        dst.apply(
          src,
          [=](const auto &s) {
            if constexpr (std::convertible_to<decltype(s), T>) return s;
            else return s[j];
          },
          [=](auto &d, auto s) { d[j] = s; });
    } else {
      POLYMATHVECTORIZE
      for (ptrdiff_t j = 0; j < L; ++j)
        dst.apply(src.map([=](const auto &s) { return s[j]; }),
                  [=](auto &d, const auto &s) { d[j] = s; });
    }
  } else {
    ptrdiff_t R = ptrdiff_t(M), C = ptrdiff_t(N);
    POLYMATHNOVECTORIZE
    for (ptrdiff_t i = 0; i < R; ++i) {
      if constexpr (!std::is_copy_assignable_v<T>) {
        POLYMATHVECTORIZE
        for (ptrdiff_t j = 0; j < C; ++j)
          dst.apply(src.map([=](const auto &s) {
            if constexpr (std::convertible_to<decltype(s), T>) return auto{s};
            else if constexpr (math::RowVector<decltype(s)>) return auto{s[j]};
            else if constexpr (math::ColVector<decltype(s)>) return auto{s[i]};
            else return auto{s[i, j]};
          }),
                    [=](auto &d, const auto &s) { d[i, j] = s; });
      } else {
        POLYMATHVECTORIZE
        for (ptrdiff_t j = 0; j < C; ++j)
          dst.apply(src.map([=](const auto &s) { return s[i, j]; }),
                    [=](auto &d, const auto &s) { d[i, j] = s; });
      }
    }
  }
}
template <typename A, typename... As, typename B, typename... Bs>
void scopyTo(Tuple<A, As...> &dst, const Tuple<B, Bs...> &src) {
  using T = std::common_type<utils::eltype_t<A>, utils::eltype_t<As>...,
                             utils::eltype_t<B>, utils::eltype_t<Bs>...>;
  static_assert(sizeof(T) <= 8);
  auto [M, N] = promote_shape(dst, src);
  if constexpr (math::AbstractVector<A>) {
    ptrdiff_t L = math::IsOne<decltype(N)> ? M : N;
    if constexpr (!std::is_copy_assignable_v<T>) {
      POLYMATHVECTORIZE
      for (ptrdiff_t j = 0; j < L; ++j)
        dst.apply(
          src,
          [=](const auto &s) {
            if constexpr (std::convertible_to<decltype(s), T>) return s;
            else return s[j];
          },
          [=](auto &d, auto s) { d[j] = s; });
    } else {
      POLYMATHVECTORIZE
      for (ptrdiff_t j = 0; j < L; ++j)
        dst.apply(src.map([=](const auto &s) { return s[j]; }),
                  [=](auto &d, const auto &s) { d[j] = s; });
    }
  } else {
    ptrdiff_t R = ptrdiff_t(M), C = ptrdiff_t(N);
    POLYMATHNOVECTORIZE
    for (ptrdiff_t i = 0; i < R; ++i) {
      if constexpr (!std::is_copy_assignable_v<T>) {
        POLYMATHVECTORIZE
        for (ptrdiff_t j = 0; j < C; ++j)
          dst.apply(src.map([=](const auto &s) {
            if constexpr (std::convertible_to<decltype(s), T>) return s;
            else if constexpr (math::RowVector<decltype(s)>) return s[j];
            else if constexpr (math::ColVector<decltype(s)>) return s[i];
            else return s[i, j];
          }),
                    [=](auto &d, auto s) { d[i, j] = s; });
      } else {
        POLYMATHVECTORIZE
        for (ptrdiff_t j = 0; j < C; ++j)
          dst.apply(src.map([=](const auto &s) { return s[i, j]; }),
                    [=](auto &d, const auto &s) { d[i, j] = s; });
      }
    }
  }
}
}; // namespace tupletensorops

template <typename A, typename... As>
template <typename B, typename... Bs>
inline constexpr auto Tuple<A, As...>::operator<<(const Tuple<B, Bs...> &src)
requires(sizeof...(As) == sizeof...(Bs))
{
  using T = std::common_type<utils::eltype_t<A>, utils::eltype_t<As>...,
                             utils::eltype_t<B>, utils::eltype_t<Bs>...>;
  if consteval {
    tupletensorops::scopyTo(*this, src);
  } else {
    if constexpr (sizeof(T) <= sizeof(double))
      tupletensorops::vcopyTo(*this, src);
    else tupletensorops::scopyTo(*this, src);
  }
}

} // namespace poly::containers

