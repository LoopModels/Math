#ifdef USE_MODULE
module;
#else
#pragma once
#endif
#include "LoopMacros.hxx"
#ifndef USE_MODULE
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <functional>
#include <type_traits>

#include "Containers/Tuple.cxx"
#include "Math/ArrayConcepts.cxx"
#include "Math/CheckSizes.cxx"
#include "Math/Indexing.cxx"
#include "Math/MatrixDimensions.cxx"
#include "Math/ScalarizeViaCastArrayOps.cxx"
#include "Math/UniformScaling.cxx"
#include "SIMD/Intrin.cxx"
#include "SIMD/UnrollIndex.cxx"
#include "Utilities/Invariant.cxx"
#else
export module AssignExprTemplates;

import ArrayConcepts;
import CheckSizes;
import Indexing;
import Invariant;
import ScalarizeViaCast;
import SIMD;
import STL;
import Tuple;
import TypeCompression;
import UniformScaling;
#endif

#define CASTTOSCALARIZE

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif
using utils::invariant;
// scalars broadcast

namespace detail {
struct NoRowIndex {};
struct CopyAssign {};

template <typename D, typename S, typename Op>
[[gnu::artificial, gnu::always_inline]] inline constexpr void
assign(D &&d, const S &s, Op op) {
  if constexpr (std::same_as<Op, CopyAssign>) d = s;
  else if constexpr (std::same_as<Op, std::plus<>>) d += s;
  else if constexpr (std::same_as<Op, std::minus<>>) d -= s;
  else if constexpr (std::same_as<Op, std::multiplies<>>) d *= s;
  else if constexpr (std::same_as<Op, std::divides<>>) d /= s;
  else d = op(d, s);
}

template <typename S, typename D, typename Op>
concept Assignable = requires(D &dst, S src, Op op) {
  { assign(dst, src, op) };
};

template <typename D, typename S, typename R, typename C, typename Op>
[[gnu::artificial, gnu::always_inline]] inline constexpr void
assign(D d, const S &s, R r, C c, Op op) {
  constexpr bool no_row_ind = std::same_as<R, NoRowIndex>;
  if constexpr (std::is_assignable_v<utils::eltype_t<D> &, S>)
    // if constexpr (Assignable<S, utils::eltype_t<D>, Op>)
    // if constexpr (std::convertible_to<S, utils::eltype_t<D>>)
    if constexpr (no_row_ind) assign(d[c], s, op);
    else assign(d[r, c], s, op);
  else if constexpr (math::RowVector<S>)
    if constexpr (no_row_ind) assign(d[c], s[c], op);
    else assign(d[r, c], s[c], op);
  else if constexpr (math::ColVector<S>)
    if constexpr (no_row_ind) assign(d[c], s[c], op);
    else assign(d[r, c], s[r], op);
  else if constexpr (std::same_as<Op, CopyAssign>)
    if constexpr (no_row_ind) d[c] = s[c];
    else d[r, c] = s[r, c];
  else if constexpr (std::same_as<Op, std::plus<>>)
    if constexpr (no_row_ind) d[c] += s[c];
    else d[r, c] += s[r, c];
  else if constexpr (std::same_as<Op, std::minus<>>)
    if constexpr (no_row_ind) d[c] -= s[c];
    else d[r, c] -= s[r, c];
  else if constexpr (std::same_as<Op, std::multiplies<>>)
    if constexpr (no_row_ind) d[c] *= s[c];
    else d[r, c] *= s[r, c];
  else if constexpr (std::same_as<Op, std::divides<>>)
    if constexpr (no_row_ind) d[c] /= s[c];
    else d[r, c] /= s[r, c];
  else if constexpr (no_row_ind) d[c] = op(const_cast<const D &>(d)[c], s[c]);
  else d[r, c] = op(const_cast<const D &>(d)[r, c], s[r, c]);
}

template <typename S, typename T>
[[gnu::always_inline]] constexpr auto get(T &&s, auto) -> decltype(auto) {
  return std::forward<T>(s);
}
template <typename S, typename T>
[[gnu::always_inline]] constexpr auto get(T &&s, auto, auto) -> decltype(auto) {
  return std::forward<T>(s);
}
template <typename S, math::LinearlyIndexable<S> V>
[[gnu::always_inline]] constexpr auto get(const V &v,
                                          auto i) -> decltype(auto) {
  return v[i];
}
template <typename S, math::CartesianIndexable<S> V>
[[gnu::always_inline]] constexpr auto get(const V &v, auto i,
                                          auto j) -> decltype(auto) {
  return v[i, j];
}
template <typename T, typename S>
concept OnlyLinearlyIndexable =
  math::LinearlyIndexable<S> && !math::CartesianIndexable<S>;
template <typename S, OnlyLinearlyIndexable<S> V>
[[gnu::always_inline]] constexpr auto get(const V &v, auto i,
                                          auto j) -> decltype(auto) {
  static_assert(math::AbstractVector<V>);
  if constexpr (math::RowVector<V>) return v[j];
  else return v[i];
}

template <typename To, typename From>
constexpr bool ScalarizeViaCastToImpl =
  std::same_as<To, math::scalarize_via_cast_t<std::remove_cvref_t<From>>>;

template <typename To, typename... U>
consteval auto ScalarizeViaCastTo() -> bool {
  return (... && ScalarizeViaCastToImpl<To, U>);
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

// template <typename D, typename S>
// constexpr void fastCopy(D *d, const S *s, size_t N) {
//   if (!N) return;
//   // if constexpr (std::same_as<D, S> && std::is_trivially_copyable_v<D>)
//   //   std::memcpy(d, s, N * sizeof(D));
//   // else
//   std::copy_n(s, N, d);
// }
} // namespace detail

template <typename A, typename B>
[[gnu::always_inline]] constexpr auto promote_shape(const A &a, const B &b) {
  if constexpr (AbstractVector<A> && AbstractVector<B>) {
    return CartesianIndex(std::integral_constant<ptrdiff_t, 1>{},
                          check_sizes(a.size(), b.size()));
  } else {
    auto sa = shape(a);
    // broadcasting static sizes is awkward, as it can prevent propogating
    // static size information for copying an `StaticArray` to an `Array` of
    // the same size, when the `StaticArray` has static size of `1`.
    if constexpr (!std::convertible_to<B, utils::eltype_t<A>>) {
      auto M = unwrapRow(numRows(b));
      auto N = unwrapCol(numCols(b));
      if constexpr (IsOne<decltype(M)>)
        if constexpr (IsOne<decltype(N)>) return sa;
        else return CartesianIndex(sa.row, check_sizes(sa.col, N));
      else if constexpr (IsOne<decltype(N)>)
        return CartesianIndex(check_sizes(sa.row_idx_, M), sa.col);
      else
        return CartesianIndex(check_sizes(sa.row_idx_, M),
                              check_sizes(sa.col_idx_, N));
    } else return sa;
  }
}

#ifndef POLYMATHNOEXPLICITSIMDARRAY
template <typename LHS, typename RHS, typename I, typename R, typename Op>
[[gnu::always_inline]] inline void vcopyToSIMD(LHS A, RHS B, I L, R row,
                                               Op op) {
  // TODO: if `R` is a row index, maybe don't fully unroll static `L`
  // We're going for very short SIMD vectors to focus on small sizes
  // using PT = std::common_type_t<utils::eltype_t<LHS>, utils::eltype_t<RHS>>;
  using PT = std::common_type_t<utils::eltype_t<LHS>, utils::eltype_t<RHS>>;
  invariant(L >= 0);
  if constexpr (StaticInt<I>) {
    constexpr std::array<ptrdiff_t, 3> vdr =
      simd::VectorDivRem<ptrdiff_t(L), PT>();
    constexpr ptrdiff_t W = vdr[0];
    constexpr ptrdiff_t fulliter = vdr[1];
    constexpr ptrdiff_t remainder = vdr[2];
    if constexpr (remainder > 0) {
      auto u{simd::index::unrollmask<fulliter + 1, W>(L, 0)};
      detail::assign(A, B, row, u, op);
    } else {
      simd::index::Unroll<fulliter, W> u{0};
      detail::assign(A, B, row, u, op);
    }
  } else {
    constexpr ptrdiff_t W = simd::Width<PT>;
#ifdef __AVX512VL__
    // ptrdiff_t i = 0;
    // for (ptrdiff_t j = W; j <= L; j += W) {
    //   simd::index::Unroll<1, W> u{i};
    //   detail::assign(A, B, row, u, op);
    //   i = j;
    // }
    // if (ptrdiff_t M = L % W) {
    //   auto u{simd::index::tailmask<W>(i, M)};
    //   detail::assign(A, B, row, u, op);
    // }
    ptrdiff_t i = 0;
    static constexpr ptrdiff_t vbody = std::min(4 * W, ptrdiff_t(64));
    POLYMATHNOUNROLL
    for (; i <= L - vbody; i += vbody) {
      simd::index::Unroll<vbody / W, W> u{i};
      detail::assign(A, B, row, u, op);
    }
    if (i < L) {
      auto ufull{simd::index::tailmask<W>(i, L - i)};
      // auto ufull{simd::index::unrollmask<1, 64>(L, i)};
      for (;;) {
        auto u{ufull.template sub<W>()};
        detail::assign(A, B, row, u, op);
        if (!ufull) break;
        // if (L <= ufull.index_) return;
        // if (!ufull) ufull = simd::index::unrollmask<1, 64>(L,
        // ufull.index_);
      }
    }
    // for (ptrdiff_t i = 0;;) {
    //   auto ufull{simd::index::unrollmask<1, 64>(L, i)};
    //   POLYMATHIVDEP
    //   for (ptrdiff_t j = 0; (j < (64 / W)); ++j) {
    //     if (!ufull) return;
    //     auto u{ufull.template sub<W>()};
    //     detail::assign(A, B, row, u, op);
    //   }
    //   i = ufull.index_;
    // }
#else
    ptrdiff_t i = 0;
    for (; i <= L - W; i += W) {
      simd::index::Unroll<1, W> u{i};
      detail::assign(A, B, row, u, op);
    }
    if (ptrdiff_t M = L - i) {
      auto u{simd::index::tailmask<W>(i, M)};
      detail::assign(A, B, row, u, op);
    }
    // for (ptrdiff_t i = 0;; i += W) {
    //   auto u{simd::index::unrollmask<1, W>(L, i)};
    //   if (!u) break;
    //   detail::assign(A, B, row, u, op);
    // }
#endif
  }
}
#endif

template <class T, class S, class P> class ArrayOps {
  static_assert(std::is_copy_assignable_v<T> ||
                (std::is_trivially_copyable_v<T> &&
                 std::is_trivially_move_assignable_v<T>));
  constexpr auto data_() { return static_cast<P *>(this)->data(); }
  // [[gnu::returns_nonnull]] constexpr auto data_() const -> const T * {
  //   return static_cast<const P *>(this)->data();
  // }
  [[nodiscard]] constexpr auto size_() const -> ptrdiff_t {
    return static_cast<const P *>(this)->size();
  }
  [[nodiscard]] constexpr auto dim_() const -> S {
    return static_cast<const P *>(this)->dim();
  }
  // returns a mutable view of self
  constexpr auto Self() -> P & { return *static_cast<P *>(this); }
  constexpr auto SelfView() { return static_cast<P *>(this)->mview(); }
  [[nodiscard]] constexpr auto nr() const -> ptrdiff_t {
    return ptrdiff_t(static_cast<const P *>(this)->numRow());
  }
  [[nodiscard]] constexpr auto nc() const {
    return unwrapCol(static_cast<const P *>(this)->numCol());
  }
  [[nodiscard]] constexpr auto rs() const {
    return unwrapRow(static_cast<const P *>(this)->rowStride());
  }

protected:
  template <typename Op, typename RHS> void vcopyTo(RHS B, Op op) {
    auto self{SelfView()};
    auto [M, N] = promote_shape(self, B);
    constexpr bool assign = std::same_as<Op, detail::CopyAssign>;
    using PT = std::common_type_t<utils::eltype_t<P>, utils::eltype_t<RHS>>;
#ifdef CASTTOSCALARIZE
    using E = math::scalarize_via_cast_t<
      std::remove_cvref_t<decltype(std::declval<P>().view())>>;
    if constexpr (!std::same_as<E, void> &&
                  ((detail::ScalarizeViaCastTo<E, decltype(B)>()) ||
                   (std::same_as<std::remove_cvref_t<decltype(B)>, double> &&
                    std::same_as<Op, std::multiplies<>>))) {
      auto d{reinterpret<E>(self)};
      if constexpr (assign) d << reinterpret<E>(B);
      else d << op(d, reinterpret<E>(B));
#ifndef POLYMATHNOEXPLICITSIMDARRAY
    } else if constexpr (simd::SIMDSupported<PT>) {
#else
    } else if constexpr (AbstractVector<P>) {
#endif
#elifndef POLYMATHNOEXPLICITSIMDARRAY
    if constexpr (simd::SIMDSupported<PT>) {
#else
    if constexpr (AbstractVector<P>) {
#endif
#ifndef POLYMATHNOEXPLICITSIMDARRAY
      if constexpr (IsOne<decltype(M)>)
        vcopyToSIMD(self, view(B), N, detail::NoRowIndex{}, op);
      else if constexpr (IsOne<decltype(N)>)
        vcopyToSIMD(self, view(B), M, detail::NoRowIndex{}, op);
      else if constexpr (StaticInt<decltype(M)>) {
        constexpr std::array<ptrdiff_t, 2> UIR =
          detail::unrollf<ptrdiff_t(M)>();
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
#endif
      ptrdiff_t L = IsOne<decltype(N)> ? M : N;
      constexpr bool isstatic =
        IsOne<decltype(N)> ? StaticInt<decltype(M)> : StaticInt<decltype(N)>;
      if constexpr (!std::is_copy_assignable_v<PT> && assign) {
        POLYMATHIVDEP
        for (ptrdiff_t j = 0; j < L; ++j)
          if constexpr (std::convertible_to<decltype(B), PT>) self[j] = auto{B};
          else self[j] = auto{B[j]};
      } else if constexpr (isstatic) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t j = 0; j < L; ++j)
          detail::assign(self, B, detail::NoRowIndex{}, j, op);
      } else {
        POLYMATHIVDEP
        for (ptrdiff_t j = 0; j < L; ++j)
          detail::assign(self, B, detail::NoRowIndex{}, j, op);
      }
    } else {
      ptrdiff_t R = ptrdiff_t(M), C = ptrdiff_t(N);
      POLYMATHNOVECTORIZE
      for (ptrdiff_t i = 0; i < R; ++i) {
        if constexpr (!std::is_copy_assignable_v<PT> && assign) {
          POLYMATHIVDEP
          for (ptrdiff_t j = 0; j < C; ++j)
            if constexpr (std::convertible_to<decltype(B), PT>)
              self[i, j] = auto{B};
            else if constexpr (RowVector<decltype(B)>) self[i, j] = auto{B[j]};
            else if constexpr (ColVector<decltype(B)>) self[i, j] = auto{B[i]};
            else self[i, j] = auto{B[i, j]};
        } else {
          POLYMATHIVDEP
          for (ptrdiff_t j = 0; j < C; ++j) detail::assign(self, B, i, j, op);
        }
      }
    }
  }

public:
  using array_op_parent_type = P;

  template <std::convertible_to<T> Y>
  [[gnu::always_inline, gnu::flatten]] constexpr auto
  operator<<(const UniformScaling<Y> &B) -> P & {
    static_assert(MatrixDimension<S>);
    std::fill_n(data_(), ptrdiff_t(this->dim_()), T{});
    static_cast<P *>(this)->diag() << B.value;
    return *static_cast<P *>(this);
  }

  [[gnu::always_inline, gnu::flatten]] constexpr auto
  operator<<(const auto &B) -> P & {
    vcopyTo(view(B), detail::CopyAssign{});
    return Self();
  }
  [[gnu::always_inline, gnu::flatten]] constexpr auto
  operator+=(const auto &B) -> P & {
    vcopyTo(view(B), std::plus<>{});
    return Self();
  }
  [[gnu::always_inline, gnu::flatten]] constexpr auto
  operator-=(const auto &B) -> P & {
    vcopyTo(view(B), std::minus<>{});
    return Self();
  }
  [[gnu::always_inline, gnu::flatten]] constexpr auto
  operator*=(const auto &B) -> P & {
    vcopyTo(view(B), std::multiplies<>{});
    return Self();
  }
  [[gnu::always_inline, gnu::flatten]] constexpr auto
  operator/=(const auto &B) -> P & {
    vcopyTo(view(B), std::divides<>{});
    return Self();
  }
};

} // namespace math

#ifdef USE_MODULE
export namespace containers {
#else
namespace containers {
#endif

namespace detail {
template <typename T> struct IsArrayOpImpl : std::false_type {};
template <typename T, typename S, typename P>
struct IsArrayOpImpl<math::ArrayOps<T, S, P>> : std::true_type {};

template <typename T>
concept IsArrayOp =
  requires(T) { typename std::remove_cvref_t<T>::array_op_parent_type; };
} // namespace detail

namespace tupletensorops {
#ifndef POLYMATHNOEXPLICITSIMDARRAY
// FIXME:
// Need to do all loads before all stores!!!
// This is because we want to support fusing loops where we may be overwriting
// inputs, e.g. `tie(x,y) << Tuple(x - y, x + y);`
template <typename A, typename... As, typename B, typename... Bs, typename I,
          typename R>
[[gnu::always_inline]] inline void
vcopyToSIMD(Tuple<A, As...> &dref, const Tuple<B, Bs...> &sref, I L, R row) {
  // we want to avoid the pointer reloading on every iter, so we help alias
  // analysis out.
  auto dst{dref.mutmap([](auto &d) { return d.mview(); })};
  auto src{sref.map([](auto &s) { return view(s); })};
  // TODO: if `R` is a row index, maybe don't fully unroll static `L`
  // We're going for very short SIMD vectors to focus on small sizes
  using T = std::common_type_t<utils::eltype_t<A>, utils::eltype_t<As>...,
                               utils::eltype_t<B>, utils::eltype_t<Bs>...>;
  if constexpr (math::StaticInt<I>) {
    constexpr ptrdiff_t SL = ptrdiff_t(L);
    constexpr std::array<ptrdiff_t, 3> vdr = simd::VectorDivRem<SL, T>();
    constexpr ptrdiff_t W = vdr[0];
    constexpr ptrdiff_t fulliter = vdr[1];
    constexpr ptrdiff_t remainder = vdr[2];
    if constexpr (remainder > 0) {
      auto u{simd::index::unrollmask<fulliter + 1, W>(L, 0)};
      if constexpr (std::same_as<R, math::detail::NoRowIndex>)
        dst.apply(
          src.map([=](const auto &s) { return math::detail::get<T>(s, u); }),
          [=](auto &d, const auto &s) { d[u] = s; });
      else
        dst.apply(src.map([=](const auto &s) {
          return math::detail::get<T>(s, row, u);
        }),
                  [=](auto &d, const auto &s) { d[row, u] = s; });
    } else if constexpr (std::same_as<R, math::detail::NoRowIndex>) {
      simd::index::Unroll<fulliter, W> u{0};
      dst.apply(
        src.map([=](const auto &s) { return math::detail::get<T>(s, u); }),
        [=](auto &d, const auto &s) { d[u] = s; });
    } else {
      simd::index::Unroll<fulliter, W> u{0};
      dst.apply(
        src.map([=](const auto &s) { return math::detail::get<T>(s, row, u); }),
        [=](auto &d, const auto &s) { d[row, u] = s; });
    }
  } else {
    constexpr ptrdiff_t W = simd::Width<T>;
    for (ptrdiff_t i = 0;; i += W) {
      auto u{simd::index::unrollmask<1, W>(L, i)};
      if (!u) break;
      if constexpr (std::same_as<R, math::detail::NoRowIndex>)
        dst.apply(
          src.map([=](const auto &s) { return math::detail::get<T>(s, u); }),
          [=](auto &d, const auto &s) { d[u] = s; });
      else
        dst.apply(src.map([=](const auto &s) {
          return math::detail::get<T>(s, row, u);
        }),
                  [=](auto &d, const auto &s) { d[row, u] = s; });
    }
  }
}
#endif

// template <typename A, typename B>
// [[gnu::always_inline]] constexpr auto promote_shape(const Tuple<A> &a,
//                                                     const Tuple<B> &b) {
//   return math::promote_shape(a.head, b.head);
// }
template <typename A, typename... As, typename B, typename... Bs>
[[gnu::always_inline]] constexpr auto promote_shape(const Tuple<A, As...> &a,
                                                    const Tuple<B, Bs...> &b)
requires(sizeof...(As) == sizeof...(Bs))
{
  auto h = math::promote_shape(a.head_, b.head_);
  if constexpr (sizeof...(As) == 0) return h;
  else {
    auto [Mh, Nh] = h;
    auto [Mt, Nt] = promote_shape(a.tail_, b.tail_);
    return math::CartesianIndex(math::check_sizes(Mh, Mt),
                                math::check_sizes(Nh, Nt));
  }
}
template <typename A, typename... As, typename B, typename... Bs>
[[gnu::always_inline]] constexpr void vcopyTo(Tuple<A, As...> &dst,
                                              const Tuple<B, Bs...> &src) {
  using T = std::common_type_t<utils::eltype_t<A>, utils::eltype_t<As>...,
                               utils::eltype_t<B>, utils::eltype_t<Bs>...>;
  // static_assert(sizeof(T) <= 8);
  auto [M, N] = promote_shape(dst, src);
#ifndef POLYMATHNOEXPLICITSIMDARRAY
  if constexpr (simd::SIMDSupported<std::remove_cvref_t<T>>) {
    if constexpr (math::IsOne<decltype(M)>)
      vcopyToSIMD(dst, src, N, math::detail::NoRowIndex{});
    else if constexpr (math::IsOne<decltype(N)>)
      vcopyToSIMD(dst, src, M, math::detail::NoRowIndex{});
    else if constexpr (math::StaticInt<decltype(M)>) {
      constexpr std::array<ptrdiff_t, 2> UIR =
        math::detail::unrollf<ptrdiff_t(M)>();
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
#else
  if constexpr (math::AbstractVector<A>) {
#endif
    ptrdiff_t L = math::IsOne<decltype(N)> ? M : N;
    if constexpr (!std::is_copy_assignable_v<T>) {
      POLYMATHIVDEP
      for (ptrdiff_t j = 0; j < L; ++j)
        dst.apply(src.map([=](const auto &s) {
          if constexpr (std::convertible_to<decltype(s), T>) return s;
          else return s[j];
        }),
                  [=](auto &d, auto s) { d[j] = s; });
    } else {
      POLYMATHIVDEP
      for (ptrdiff_t j = 0; j < L; ++j)
        dst.apply(src.map([=](const auto &s) { return s[j]; }),
                  [=](auto &d, const auto &s) { d[j] = s; });
    }
  } else {
    ptrdiff_t R = ptrdiff_t(M), C = ptrdiff_t(N);
    POLYMATHNOVECTORIZE
    for (ptrdiff_t i = 0; i < R; ++i) {
      if constexpr (!std::is_copy_assignable_v<T>) {
        POLYMATHIVDEP
        for (ptrdiff_t j = 0; j < C; ++j)
          dst.apply(src.map([=](const auto &s) {
            if constexpr (std::convertible_to<decltype(s), T>) return auto{s};
            else if constexpr (math::RowVector<decltype(s)>) return auto{s[j]};
            else if constexpr (math::ColVector<decltype(s)>) return auto{s[i]};
            else return auto{s[i, j]};
          }),
                    [=](auto &d, const auto &s) { d[i, j] = s; });
      } else {
        POLYMATHIVDEP
        for (ptrdiff_t j = 0; j < C; ++j)
          dst.apply(src.map([=](const auto &s) { return s[i, j]; }),
                    [=](auto &d, const auto &s) { d[i, j] = s; });
      }
    }
  }
}
}; // namespace tupletensorops

// Note: these are inlined, because we want
// the compiler to be exposed to which arrays
// are identical, in case of re-use, so that
// they can share pointers.
template <typename A, typename... As, typename B, typename... Bs>
[[gnu::always_inline, gnu::flatten]] constexpr void
copyFrom(Tuple<A, As...> &dst, const Tuple<B, Bs...> &src)
requires(sizeof...(As) == sizeof...(Bs) && detail::IsArrayOp<A> &&
         (... && detail::IsArrayOp<As>))
{
#ifndef CASTTOSCALARIZE
  tupletensorops::vcopyTo(dst, src);
#else
  using C = math::scalarize_via_cast_t<
    std::remove_cvref_t<decltype(std::declval<A>().view())>>;
  if constexpr ((!std::same_as<C, void>) &&
                math::detail::ScalarizeViaCastTo<
                  C, As..., decltype(std::declval<B>().view()), Bs...>()) {
    using T = std::common_type_t<utils::eltype_t<A>, utils::eltype_t<As>...,
                                 utils::eltype_t<B>, utils::eltype_t<Bs>...>;
    if constexpr ((sizeof(T) % (sizeof(C) * simd::Width<C>)) != 0) {
      auto lval{dst.mutmap([](auto &d) { return reinterpret<C>(d); })};
      tupletensorops::vcopyTo(
        lval, src.map([](const auto &s) { return reinterpret<C>(s); }));
    } else tupletensorops::vcopyTo(dst, src);
  } else tupletensorops::vcopyTo(dst, src);
#endif
}
} // namespace containers
