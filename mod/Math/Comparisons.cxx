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
#include <type_traits>

#include "Math/ArrayConcepts.cxx"
#include "SIMD/SIMD.cxx"
#else
export module Comparisons;
import ArrayConcepts;
import SIMD;
import STL;
#endif

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif
// constexpr auto allZero(const auto &x) -> bool {
//   return std::all_of(x.begin(), x.end(), [](auto a) { return a == 0; });
//   // return std::ranges::all_of(x, [](auto x) { return x == 0; });
// }
// constexpr auto allGEZero(const auto &x) -> bool {
//   return std::all_of(x.begin(), x.end(), [](auto a) { return a >= 0; });
//   // return std::ranges::all_of(x, [](auto x) { return x >= 0; });
// }
// constexpr auto allLEZero(const auto &x) -> bool {
//   return std::all_of(x.begin(), x.end(), [](auto a) { return a <= 0; });
//   // return std::ranges::all_of(x, [](auto x) { return x <= 0; });
// }

// constexpr auto anyNEZero(const auto &x) -> bool {
//   return std::any_of(x.begin(), x.end(), [](int64_t y) { return y != 0; });
// }
[[gnu::always_inline, gnu::flatten]] constexpr auto
any(const AbstractTensor auto &A, const auto &f) -> bool {
  auto [M, N] = shape(A);
  using TA = std::remove_cvref_t<decltype(A)>;
  using T = utils::eltype_t<TA>;
  if constexpr (simd::SIMDSupported<T>) {
    if constexpr (AbstractMatrix<TA>) {
      if constexpr (StaticInt<decltype(N)>) {
        constexpr std::array<ptrdiff_t, 3> vdr =
          simd::VectorDivRem<ptrdiff_t(N), T>();
        constexpr ptrdiff_t W = vdr[0];
        constexpr ptrdiff_t fulliter = vdr[1];
        constexpr ptrdiff_t remainder = vdr[2];
        for (ptrdiff_t r = 0; r < M; ++r) {
          ptrdiff_t L = W * fulliter;
          for (ptrdiff_t i = 0; i < L; i += W)
            if (f(A[r, simd::index::Unroll<1, W>{i}])) return true;
          if constexpr (remainder > 0)
            if (f(A[r, simd::index::unrollmask<1, W>(N, L)])) return true;
        }
      } else {
        constexpr ptrdiff_t W = simd::Width<T>;
        for (ptrdiff_t r = 0; r < M; ++r) {
          for (ptrdiff_t i = 0;; i += W) {
            auto u{simd::index::unrollmask<1, W>(N, i)};
            if (!u) break;
            if (f(A[r, u])) return true;
          }
        }
      }
    } else if constexpr (StaticInt<decltype(M)> && StaticInt<decltype(N)>) {
      ptrdiff_t L = RowVector<TA> ? N : M;
      using SL = std::conditional_t<RowVector<TA>, decltype(N), decltype(M)>;
      constexpr std::array<ptrdiff_t, 3> vdr =
        simd::VectorDivRem<ptrdiff_t(SL{}), T>();
      constexpr ptrdiff_t W = vdr[0];
      constexpr ptrdiff_t fulliter = vdr[1];
      constexpr ptrdiff_t remainder = vdr[2];
      ptrdiff_t K = W * fulliter;
      for (ptrdiff_t i = 0; i < K; i += W)
        if (f(A[simd::index::Unroll<1, W>{i}])) return true;
      if constexpr (remainder > 0)
        if (f(A[simd::index::unrollmask<1, W>(L, K)])) return true;
    } else {
      constexpr ptrdiff_t W = simd::Width<T>;
      ptrdiff_t L = RowVector<TA> ? N : M;
      for (ptrdiff_t i = 0;; i += W) {
        auto u{simd::index::unrollmask<1, W>(L, i)};
        if (!u) break;
        if (f(A[u])) return true;
      }
    }
  } else if constexpr (AbstractMatrix<TA>) {
    for (ptrdiff_t r = 0; r < M; ++r)
      for (ptrdiff_t i = 0; i < N; ++i)
        if (f(A[r, i])) return true;
  } else {
    ptrdiff_t L = RowVector<TA> ? N : M;
    for (ptrdiff_t i = 0; i < L; ++i)
      if (f(A[i])) return true;
  }
  return false;
}
constexpr auto anyNEZero(const AbstractTensor auto &A) -> bool {
  using T = utils::eltype_t<decltype(A)>;
  constexpr ptrdiff_t W = simd::VecWidth<T, decltype(numRows(A))::comptime(),
                                         decltype(numCols(A))::comptime()>();
  if constexpr (simd::SIMDSupported<T>)
    return any(A, [](simd::Unroll<1, 1, W, T> v) -> bool {
      return bool(v != simd::Vec<W, T>{});
    });
  else return any(A, [](T x) -> bool { return x != T{}; });
}
constexpr auto anyLTZero(const AbstractTensor auto &A) -> bool {
  using T = utils::eltype_t<decltype(A)>;
  constexpr ptrdiff_t W = simd::VecWidth<T, decltype(numRows(A))::comptime(),
                                         decltype(numCols(A))::comptime()>();
  if constexpr (simd::SIMDSupported<T>)
    return any(A, [](simd::Unroll<1, 1, W, T> v) -> bool {
      return bool(v < simd::Vec<W, T>{});
    });
  else return any(A, [](T x) -> bool { return x < T{}; });
}
constexpr auto anyGTZero(const AbstractTensor auto &A) -> bool {
  using T = utils::eltype_t<decltype(A)>;
  constexpr ptrdiff_t W = simd::VecWidth<T, decltype(numRows(A))::comptime(),
                                         decltype(numCols(A))::comptime()>();
  if constexpr (simd::SIMDSupported<T>)
    return any(A, [](simd::Unroll<1, 1, W, T> v) -> bool {
      return bool(v > simd::Vec<W, T>{});
    });
  else return any(A, [](T x) -> bool { return x > T{}; });
}
constexpr auto countNonZero(const auto &x) -> ptrdiff_t {
  return std::count_if(x.begin(), x.end(), [](auto a) { return a != 0; });
  // return std::ranges::count_if(x, [](auto x) { return x != 0; });
}

constexpr auto allZero(const AbstractTensor auto &A) -> bool {
  return !anyNEZero(A);
}
constexpr auto allLEZero(const AbstractTensor auto &A) -> bool {
  return !anyGTZero(A);
}
constexpr auto allGEZero(const AbstractTensor auto &A) -> bool {
  return !anyLTZero(A);
}

} // namespace math
