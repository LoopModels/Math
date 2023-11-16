#pragma once

#include "Math/Indexing.hpp"
#include "Math/Matrix.hpp"
#include "Math/UniformScaling.hpp"
#include "SIMD/Unroll.hpp"
#include "Utilities/LoopMacros.hpp"
#include <algorithm>
#include <cstring>
#include <type_traits>

namespace poly::math {

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
  struct NoRowIndex {};
  template <typename I, typename R>
  [[gnu::always_inline]] static void vcopyToSIMD(P &self, const auto &B, I L,
                                                 R row) {
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
        if constexpr (std::same_as<R, NoRowIndex>) self[u] = B[u];
        else self[row, u] = B[row, u];
      } else {
        simd::index::Unroll<fulliter, W> u{0};
        if constexpr (std::same_as<R, NoRowIndex>) self[u] = B[u];
        else self[row, u] = B[row, u];
      }
    } else {
      constexpr ptrdiff_t W = simd::Width<T>;
      for (ptrdiff_t i = 0;; i += W) {
        auto u{simd::index::unrollmask<1, W>(L, i)};
        if (!u) break;
        if constexpr (std::same_as<R, NoRowIndex>) self[u] = B[u];
        else self[row, u] = B[row, u];
      }
    }
  }

  void vcopyTo(const AbstractVector auto &B) {
    static_assert(sizeof(utils::eltype_t<decltype(B)>) <= 8);
    P &self{Self()};
    if constexpr (MatrixDimension<S>) {
      ptrdiff_t M = nr(), N = nc();
      invariant(M, ptrdiff_t(B.size()));
      POLYMATHNOVECTORIZE
      for (ptrdiff_t i = 0; i < M; ++i) {
        T Bi = B[i];
        POLYMATHVECTORIZE
        for (ptrdiff_t j = 0; j < N; ++j) self[i, j] = Bi;
      }
    } else {
      auto L = size_();
      invariant(ptrdiff_t(L), ptrdiff_t(B.size()));
      if constexpr (simd::SIMDSupported<T>) {
        vcopyToSIMD(self, B, L, NoRowIndex{});
      } else if constexpr (std::is_copy_assignable_v<T>) {
        POLYMATHVECTORIZE
        for (ptrdiff_t i = 0; i < L; ++i) self[i] = B[i];
      } else {
        // MutArray is trivially copyable and move-assignable
        // we require triviality to avoid silently being slow.
        // we should fix it if hitting another case.
        POLYMATHVECTORIZE
        for (ptrdiff_t i = 0; i < L; ++i) self[i] = auto{B[i]};
      }
    }
  }
  constexpr void scopyTo(const AbstractVector auto &B) {
    P &self{Self()};
    if constexpr (MatrixDimension<S>) {
      ptrdiff_t M = nr(), N = nc();
      invariant(M, ptrdiff_t(B.size()));
      POLYMATHNOVECTORIZE
      for (ptrdiff_t i = 0; i < M; ++i) {
        T Bi = B[i];
        POLYMATHIVDEP
        for (ptrdiff_t j = 0; j < N; ++j) self[i, j] = Bi;
      }
    } else {
      ptrdiff_t L = size_();
      invariant(L, ptrdiff_t(B.size()));
      if constexpr (std::is_copy_assignable_v<T>) {
        POLYMATHIVDEP
        for (ptrdiff_t i = 0; i < L; ++i) self[i] = B[i];
      } else {
        // MutArray is trivially copyable and move-assignable
        // we require triviality to avoid silently being slow.
        // we should fix it if hitting another case.
        POLYMATHIVDEP
        for (ptrdiff_t i = 0; i < L; ++i) self[i] = auto{B[i]};
      }
    }
  }
  void vcopyTo(const AbstractMatrix auto &B) {
    static_assert(sizeof(utils::eltype_t<decltype(B)>) <= 8);
    static_assert(MatrixDimension<S>);
    auto M = nr();
    auto N = nc();
    invariant(ptrdiff_t(M), ptrdiff_t(B.numRow()));
    invariant(N, ptrdiff_t(B.numCol()));
    P &self{Self()};
    if constexpr (DenseLayout<S> &&
                  DenseTensor<std::remove_cvref_t<decltype(B)>>) {
      if constexpr (std::is_copy_assignable_v<T>)
        std::copy_n(B.begin(), M * N, data_());
      else std::memcpy(data_(), M * N * sizeof(T), B.begin());
    } else if constexpr (simd::SIMDSupported<T>) {
      if constexpr (StaticInt<decltype(M)>) {
        constexpr std::array<ptrdiff_t, 2> UIR = unrollf<ptrdiff_t(M)>();
        constexpr ptrdiff_t U = UIR[0];
        if constexpr (U != 0)
          for (ptrdiff_t r = 0; r < (M - U + 1); r += U)
            vcopyToSIMD(self, B, N, simd::index::Unroll<U>{r});
        constexpr ptrdiff_t R = UIR[1];
        if constexpr (R != 0)
          vcopyToSIMD(self, B, N, simd::index::Unroll<R>{M - R});
      } else {
        ptrdiff_t r = 0;
        for (; r < (M - 3); r += 4)
          vcopyToSIMD(self, B, N, simd::index::Unroll<4>{r});
        switch (M & 3) {
        case 0: return;
        case 1: return vcopyToSIMD(self, B, N, simd::index::Unroll<1>{r});
        case 2: return vcopyToSIMD(self, B, N, simd::index::Unroll<2>{r});
        default: return vcopyToSIMD(self, B, N, simd::index::Unroll<3>{r});
        }
      }
    } else {
      POLYMATHNOVECTORIZE
      for (ptrdiff_t i = 0; i < M; ++i) {
        if constexpr (std::is_copy_assignable_v<T>) {
          POLYMATHVECTORIZE
          for (ptrdiff_t j = 0; j < N; ++j) self[i, j] = B[i, j];
        } else {
          POLYMATHVECTORIZE
          for (ptrdiff_t j = 0; j < N; ++j) self[i, j] = auto{B[i, j]};
        }
      }
    }
  }
  constexpr void scopyTo(const AbstractMatrix auto &B) {
    static_assert(MatrixDimension<S>);
    ptrdiff_t M = ptrdiff_t(nr()), N = ptrdiff_t(nc());
    invariant(M, ptrdiff_t(B.numRow()));
    invariant(N, ptrdiff_t(B.numCol()));
    P &self{Self()};
    if constexpr (DenseLayout<S> &&
                  DenseTensor<std::remove_cvref_t<decltype(B)>>) {
      if constexpr (std::is_copy_assignable_v<T>)
        std::copy_n(B.begin(), M * N, data_());
      else std::memcpy(data_(), M * N * sizeof(T), B.begin());
    } else {
      POLYMATHNOVECTORIZE
      for (ptrdiff_t i = 0; i < M; ++i) {
        if constexpr (std::is_copy_assignable_v<T>) {
          POLYMATHIVDEP
          for (ptrdiff_t j = 0; j < N; ++j) self[i, j] = B[i, j];
        } else {
          POLYMATHIVDEP
          for (ptrdiff_t j = 0; j < N; ++j) self[i, j] = auto{B[i, j]};
        }
      }
    }
  }

  template <std::convertible_to<T> Y> void vcopyTo(const Y &b) {
    static_assert(sizeof(utils::eltype_t<decltype(b)>) <= 8);
    P &self{Self()};
    if constexpr (DenseLayout<S>) {
      std::fill_n(data_(), ptrdiff_t(size_()), T(b));
    } else if constexpr (std::is_same_v<S, StridedRange>) {
      ptrdiff_t L = size_();
      POLYMATHVECTORIZE
      for (ptrdiff_t c = 0; c < L; ++c) self[c] = b;
    } else {
      ptrdiff_t M = ptrdiff_t(nr()), N = ptrdiff_t(nc()), X = ptrdiff_t(rs());
      T *p = data_();
      POLYMATHNOVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r, p += X) std::fill_n(p, N, T(b));
    }
  }
  template <std::convertible_to<T> Y> constexpr void scopyTo(const Y &b) {
    P &self{Self()};
    if constexpr (DenseLayout<S>) {
      std::fill_n(data_(), ptrdiff_t(size_()), T(b));
    } else if constexpr (std::is_same_v<S, StridedRange>) {
      POLYMATHIVDEP
      for (ptrdiff_t c = 0, L = size_(); c < L; ++c) self[c] = b;
    } else {
      ptrdiff_t M = ptrdiff_t(nr()), N = ptrdiff_t(nc()), X = ptrdiff_t(rs());
      T *p = data_();
      POLYMATHNOVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r, p += X) std::fill_n(p, N, T(b));
    }
  }

  void vadd(const AbstractMatrix auto &B) {
    static_assert(sizeof(utils::eltype_t<decltype(B)>) <= 8);
    static_assert(MatrixDimension<S>);
    ptrdiff_t M = ptrdiff_t(nr()), N = ptrdiff_t(nc());
    invariant(M, ptrdiff_t(B.numRow()));
    invariant(N, ptrdiff_t(B.numCol()));
    P &self{Self()};
    POLYMATHNOVECTORIZE
    for (ptrdiff_t r = 0; r < M; ++r) {
      POLYMATHVECTORIZE
      for (ptrdiff_t c = 0; c < N; ++c) self[r, c] += B[r, c];
    }
  }
  constexpr void sadd(const AbstractMatrix auto &B) {
    static_assert(MatrixDimension<S>);
    ptrdiff_t M = ptrdiff_t(nr()), N = ptrdiff_t(nc());
    invariant(M, ptrdiff_t(B.numRow()));
    invariant(N, ptrdiff_t(B.numCol()));
    P &self{Self()};
    POLYMATHNOVECTORIZE
    for (ptrdiff_t r = 0; r < M; ++r) {
      POLYMATHIVDEP
      for (ptrdiff_t c = 0; c < N; ++c) self[r, c] += B[r, c];
    }
  }

  void vsub(const AbstractMatrix auto &B) {
    static_assert(sizeof(utils::eltype_t<decltype(B)>) <= 8);
    static_assert(MatrixDimension<S>);
    ptrdiff_t M = ptrdiff_t(nr()), N = ptrdiff_t(nc());
    invariant(M, ptrdiff_t(B.numRow()));
    invariant(N, ptrdiff_t(B.numCol()));
    P &self{Self()};
    POLYMATHNOVECTORIZE
    for (ptrdiff_t r = 0; r < M; ++r) {
      POLYMATHVECTORIZE
      for (ptrdiff_t c = 0; c < N; ++c) self[r, c] -= B[r, c];
    }
  }
  constexpr void ssub(const AbstractMatrix auto &B) {
    static_assert(MatrixDimension<S>);
    ptrdiff_t M = ptrdiff_t(nr()), N = ptrdiff_t(nc());
    invariant(M, ptrdiff_t(B.numRow()));
    invariant(N, ptrdiff_t(B.numCol()));
    P &self{Self()};
    POLYMATHNOVECTORIZE
    for (ptrdiff_t r = 0; r < M; ++r) {
      POLYMATHIVDEP
      for (ptrdiff_t c = 0; c < N; ++c) self[r, c] -= B[r, c];
    }
  }
  void vadd(const AbstractVector auto &B) {
    static_assert(sizeof(utils::eltype_t<decltype(B)>) <= 8);
    P &self{Self()};
    if constexpr (MatrixDimension<S>) {
      ptrdiff_t M = nr(), N = nc();
      invariant(M, ptrdiff_t(B.size()));
      POLYMATHNOVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r) {
        auto Br = B[r];
        POLYMATHVECTORIZE
        for (ptrdiff_t c = 0; c < N; ++c) self[r, c] += Br;
      }
    } else {
      ptrdiff_t L = size_();
      invariant(L, ptrdiff_t(B.size()));
      POLYMATHVECTORIZE
      for (ptrdiff_t i = 0; i < L; ++i) self[i] += B[i];
    }
  }
  constexpr void sadd(const AbstractVector auto &B) {
    P &self{Self()};
    if constexpr (MatrixDimension<S>) {
      ptrdiff_t M = ptrdiff_t(nr()), N = ptrdiff_t(nc());
      invariant(M, ptrdiff_t(B.size()));
      POLYMATHNOVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r) {
        auto Br = B[r];
        POLYMATHIVDEP
        for (ptrdiff_t c = 0; c < N; ++c) self[r, c] += Br;
      }
    } else {
      ptrdiff_t L = size_();
      invariant(L, ptrdiff_t(B.size()));
      POLYMATHIVDEP
      for (ptrdiff_t i = 0; i < L; ++i) self[i] += B[i];
    }
  }
  template <std::convertible_to<T> Y> void vadd(Y b) {
    static_assert(sizeof(utils::eltype_t<decltype(b)>) <= 8);
    P &self{Self()};
    if constexpr (MatrixDimension<S> && !DenseLayout<S>) {
      ptrdiff_t M = ptrdiff_t(nr()), N = ptrdiff_t(nc());
      POLYMATHNOVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r) {
        POLYMATHVECTORIZE
        for (ptrdiff_t c = 0; c < N; ++c) self[r, c] += b;
      }
    } else {
      POLYMATHVECTORIZE
      for (ptrdiff_t i = 0, L = size_(); i < L; ++i) self[i] += b;
    }
  }
  template <std::convertible_to<T> Y> constexpr void sadd(Y b) {
    P &self{Self()};
    if constexpr (MatrixDimension<S> && !DenseLayout<S>) {
      ptrdiff_t M = ptrdiff_t(nr()), N = ptrdiff_t(nc());
      POLYMATHNOVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r) {
        POLYMATHIVDEP
        for (ptrdiff_t c = 0; c < N; ++c) self[r, c] += b;
      }
    } else {
      POLYMATHIVDEP
      for (ptrdiff_t i = 0, L = size_(); i < L; ++i) self[i] += b;
    }
  }
  void vsub(const AbstractVector auto &B) {
    static_assert(sizeof(utils::eltype_t<decltype(B)>) <= 8);
    P &self{Self()};
    if constexpr (MatrixDimension<S>) {
      ptrdiff_t M = ptrdiff_t(nr()), N = ptrdiff_t(nc());
      invariant(M, ptrdiff_t(B.size()));
      POLYMATHNOVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r) {
        auto Br = B[r];
        POLYMATHVECTORIZE
        for (ptrdiff_t c = 0; c < N; ++c) self[r, c] -= Br;
      }
    } else {
      ptrdiff_t L = size_();
      invariant(L, ptrdiff_t(B.size()));
      POLYMATHVECTORIZE
      for (ptrdiff_t i = 0; i < L; ++i) self[i] -= B[i];
    }
  }
  constexpr void ssub(const AbstractVector auto &B) {
    P &self{Self()};
    if constexpr (MatrixDimension<S>) {
      ptrdiff_t M = ptrdiff_t(nr()), N = ptrdiff_t(nc());
      invariant(M, ptrdiff_t(B.size()));
      POLYMATHNOVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r) {
        auto Br = B[r];
        POLYMATHIVDEP
        for (ptrdiff_t c = 0; c < N; ++c) self[r, c] -= Br;
      }
    } else {
      ptrdiff_t L = size_();
      invariant(L, ptrdiff_t(B.size()));
      POLYMATHIVDEP
      for (ptrdiff_t i = 0; i < L; ++i) self[i] -= B[i];
    }
  }
  template <std::convertible_to<T> Y> void vmul(Y b) {
    static_assert(sizeof(utils::eltype_t<decltype(b)>) <= 8);
    P &self{Self()};
    if constexpr (MatrixDimension<S> && !DenseLayout<S>) {
      ptrdiff_t M = ptrdiff_t(nr()), N = ptrdiff_t(nc());
      POLYMATHNOVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r) {
        POLYMATHVECTORIZE
        for (ptrdiff_t c = 0; c < N; ++c) self[r, c] *= b;
      }
    } else {
      ptrdiff_t L = ptrdiff_t(size_());
      POLYMATHVECTORIZE
      for (ptrdiff_t c = 0; c < L; ++c) self[c] *= b;
    }
  }
  template <std::convertible_to<T> Y> constexpr void smul(Y b) {
    P &self{Self()};
    if constexpr (MatrixDimension<S> && !DenseLayout<S>) {
      ptrdiff_t M = ptrdiff_t(nr()), N = ptrdiff_t(nc());
      POLYMATHNOVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r) {
        POLYMATHIVDEP
        for (ptrdiff_t c = 0; c < N; ++c) self[r, c] *= b;
      }
    } else {
      POLYMATHIVDEP
      for (ptrdiff_t c = 0, L = ptrdiff_t(size_()); c < L; ++c) self[c] *= b;
    }
  }
  template <std::convertible_to<T> Y> void vdiv(Y b) {
    static_assert(sizeof(utils::eltype_t<decltype(b)>) <= 8);
    P &self{Self()};
    if constexpr (MatrixDimension<S> && !DenseLayout<S>) {
      ptrdiff_t M = ptrdiff_t(nr()), N = ptrdiff_t(nc());
      POLYMATHNOVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r) {
        POLYMATHVECTORIZE
        for (ptrdiff_t c = 0; c < N; ++c) self[r, c] /= b;
      }
    } else {
      ptrdiff_t L = ptrdiff_t(size_());
      POLYMATHVECTORIZE
      for (ptrdiff_t c = 0; c < L; ++c) self[c] /= b;
    }
  }
  template <std::convertible_to<T> Y> constexpr void sdiv(Y b) {
    P &self{Self()};
    if constexpr (MatrixDimension<S> && !DenseLayout<S>) {
      ptrdiff_t M = ptrdiff_t(nr()), N = ptrdiff_t(nc());
      POLYMATHNOVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r) {
        POLYMATHIVDEP
        for (ptrdiff_t c = 0; c < N; ++c) self[r, c] /= b;
      }
    } else {
      POLYMATHIVDEP
      for (ptrdiff_t c = 0, L = ptrdiff_t(size_()); c < L; ++c) self[c] /= b;
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
    if consteval {
      scopyTo(B);
    } else {
      if constexpr ((sizeof(T) <= sizeof(double)) &&
                    (!HasInnerReduction<std::remove_cvref_t<decltype(B)>>))
        vcopyTo(B);
      else scopyTo(B);
    }
    return Self();
  }

  [[gnu::flatten]] constexpr auto operator+=(const auto &B) -> P & {
    if consteval {
      sadd(B);
    } else {
      if constexpr ((sizeof(T) <= sizeof(double)) &&
                    (!HasInnerReduction<std::remove_cvref_t<decltype(B)>>))
        vadd(B);
      else sadd(B);
    }
    return Self();
  }
  [[gnu::flatten]] constexpr auto operator-=(const auto &B) -> P & {
    if consteval {
      ssub(B);
    } else {
      if constexpr ((sizeof(T) <= sizeof(double)) &&
                    (!HasInnerReduction<std::remove_cvref_t<decltype(B)>>))
        vsub(B);
      else ssub(B);
    }
    return Self();
  }
  [[gnu::flatten]] constexpr auto operator*=(const auto &B) -> P & {
    if consteval {
      smul(B);
    } else {
      if constexpr ((sizeof(T) <= sizeof(double)) &&
                    (!HasInnerReduction<std::remove_cvref_t<decltype(B)>>))
        vmul(B);
      else smul(B);
    }
    return Self();
  }
  [[gnu::flatten]] constexpr auto operator/=(const auto &B) -> P & {
    if consteval {
      sdiv(B);
    } else {
      if constexpr ((sizeof(T) <= sizeof(double)) &&
                    (!HasInnerReduction<std::remove_cvref_t<decltype(B)>>))
        vdiv(B);
      else sdiv(B);
    }
    return Self();
  }
};
} // namespace poly::math
