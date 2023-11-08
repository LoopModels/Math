#pragma once

#include "Math/Indexing.hpp"
#include "Math/Matrix.hpp"
#include "Math/UniformScaling.hpp"
#include "Math/Vector.hpp"
#include "Utilities/LoopMacros.hpp"
#include <algorithm>
#include <cstring>
#include <type_traits>

namespace poly::math {

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
  constexpr auto dim_() const -> S {
    return static_cast<const P *>(this)->dim();
  }
  constexpr auto Self() -> P & { return *static_cast<P *>(this); }
  [[nodiscard]] constexpr auto nr() const -> ptrdiff_t {
    return ptrdiff_t(static_cast<const P *>(this)->numRow());
  }
  [[nodiscard]] constexpr auto nc() const -> ptrdiff_t {
    return ptrdiff_t(static_cast<const P *>(this)->numCol());
  }
  [[nodiscard]] constexpr auto rs() const -> ptrdiff_t {
    return ptrdiff_t(static_cast<const P *>(this)->rowStride());
  }

  void vcopyTo(const AbstractVector auto &B) {
    P &self{Self()};
    if constexpr (MatrixDimension<S>) {
      ptrdiff_t M = nr(), N = nc();
      invariant(M, B.size());
      POLYMATHNOVECTORIZE
      for (ptrdiff_t i = 0; i < M; ++i) {
        T Bi = B[i];
        POLYMATHVECTORIZE
        for (ptrdiff_t j = 0; j < N; ++j) self[i, j] = Bi;
      }
    } else {
      ptrdiff_t L = size_();
      invariant(L, ptrdiff_t(B.size()));
      if constexpr (std::is_copy_assignable_v<T>) {
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
      invariant(M, B.size());
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
    static_assert(MatrixDimension<S>);
    ptrdiff_t M = nr(), N = nc();
    invariant(M, ptrdiff_t(B.numRow()));
    invariant(N, ptrdiff_t(B.numCol()));
    P &self{Self()};
    if constexpr (DenseLayout<S> &&
                  DataMatrix<std::remove_cvref_t<decltype(B)>> &&
                  DenseLayout<std::remove_cvref_t<decltype(B.dim())>>) {
      if constexpr (std::is_copy_assignable_v<T>)
        std::copy_n(B.data(), M * N, data_());
      else std::memcpy(data_(), M * N * sizeof(T), B.data());
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
    ptrdiff_t M = nr(), N = nc();
    invariant(M, ptrdiff_t(B.numRow()));
    invariant(N, ptrdiff_t(B.numCol()));
    P &self{Self()};
    if constexpr (DenseLayout<S> &&
                  DataMatrix<std::remove_cvref_t<decltype(B)>> &&
                  DenseLayout<std::remove_cvref_t<decltype(B.dim())>>) {
      if constexpr (std::is_copy_assignable_v<T>)
        std::copy_n(B.data(), M * N, data_());
      else std::memcpy(data_(), M * N * sizeof(T), B.data());
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
    P &self{Self()};
    if constexpr (DenseLayout<S>) {
      std::fill_n(data_(), ptrdiff_t(dim_()), T(b));
    } else if constexpr (std::is_same_v<S, StridedRange>) {
      ptrdiff_t L = size_();
      POLYMATHVECTORIZE
      for (ptrdiff_t c = 0; c < L; ++c) self[c] = b;
    } else {
      ptrdiff_t M = nr(), N = nc(), X = rs();
      T *p = data_();
      POLYMATHNOVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r, p += X) std::fill_n(p, N, T(b));
    }
  }
  template <std::convertible_to<T> Y> constexpr void scopyTo(const Y &b) {
    P &self{Self()};
    if constexpr (DenseLayout<S>) {
      std::fill_n(data_(), ptrdiff_t(dim_()), T(b));
    } else if constexpr (std::is_same_v<S, StridedRange>) {
      POLYMATHIVDEP
      for (ptrdiff_t c = 0, L = size_(); c < L; ++c) self[c] = b;
    } else {
      ptrdiff_t M = nr(), N = nc(), X = rs();
      T *p = data_();
      POLYMATHNOVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r, p += X) std::fill_n(p, N, T(b));
    }
  }

  void vadd(const AbstractMatrix auto &B) {
    static_assert(MatrixDimension<S>);
    ptrdiff_t M = nr(), N = nc();
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
    ptrdiff_t M = nr(), N = nc();
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
    static_assert(MatrixDimension<S>);
    ptrdiff_t M = nr(), N = nc();
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
    ptrdiff_t M = nr(), N = nc();
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
    P &self{Self()};
    if constexpr (MatrixDimension<S>) {
      ptrdiff_t M = nr(), N = nc();
      invariant(M, B.size());
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
      ptrdiff_t M = nr(), N = nc();
      invariant(M, B.size());
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
    P &self{Self()};
    if constexpr (MatrixDimension<S> && !DenseLayout<S>) {
      ptrdiff_t M = nr(), N = nc();
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
      ptrdiff_t M = nr(), N = nc();
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
    P &self{Self()};
    if constexpr (MatrixDimension<S>) {
      ptrdiff_t M = nr(), N = nc();
      invariant(M == B.size());
      POLYMATHNOVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r) {
        auto Br = B[r];
        POLYMATHVECTORIZE
        for (ptrdiff_t c = 0; c < N; ++c) self[r, c] -= Br;
      }
    } else {
      ptrdiff_t L = size_();
      invariant(L == B.size());
      POLYMATHVECTORIZE
      for (ptrdiff_t i = 0; i < L; ++i) self[i] -= B[i];
    }
  }
  constexpr void ssub(const AbstractVector auto &B) {
    P &self{Self()};
    if constexpr (MatrixDimension<S>) {
      ptrdiff_t M = nr(), N = nc();
      invariant(M == B.size());
      POLYMATHNOVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r) {
        auto Br = B[r];
        POLYMATHIVDEP
        for (ptrdiff_t c = 0; c < N; ++c) self[r, c] -= Br;
      }
    } else {
      ptrdiff_t L = size_();
      invariant(L == B.size());
      POLYMATHIVDEP
      for (ptrdiff_t i = 0; i < L; ++i) self[i] -= B[i];
    }
  }
  template <std::convertible_to<T> Y> void vmul(Y b) {
    P &self{Self()};
    if constexpr (MatrixDimension<S> && !DenseLayout<S>) {
      ptrdiff_t M = nr(), N = nc();
      POLYMATHNOVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r) {
        POLYMATHVECTORIZE
        for (ptrdiff_t c = 0; c < N; ++c) self[r, c] *= b;
      }
    } else {
      ptrdiff_t L = ptrdiff_t(dim_());
      POLYMATHVECTORIZE
      for (ptrdiff_t c = 0; c < L; ++c) self[c] *= b;
    }
  }
  template <std::convertible_to<T> Y> constexpr void smul(Y b) {
    P &self{Self()};
    if constexpr (MatrixDimension<S> && !DenseLayout<S>) {
      ptrdiff_t M = nr(), N = nc();
      POLYMATHNOVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r) {
        POLYMATHIVDEP
        for (ptrdiff_t c = 0; c < N; ++c) self[r, c] *= b;
      }
    } else {
      POLYMATHIVDEP
      for (ptrdiff_t c = 0, L = ptrdiff_t(dim_()); c < L; ++c) self[c] *= b;
    }
  }
  template <std::convertible_to<T> Y> void vdiv(Y b) {
    P &self{Self()};
    if constexpr (MatrixDimension<S> && !DenseLayout<S>) {
      ptrdiff_t M = nr(), N = nc();
      POLYMATHNOVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r) {
        POLYMATHVECTORIZE
        for (ptrdiff_t c = 0; c < N; ++c) self[r, c] /= b;
      }
    } else {
      ptrdiff_t L = ptrdiff_t(dim_());
      POLYMATHVECTORIZE
      for (ptrdiff_t c = 0; c < L; ++c) self[c] /= b;
    }
  }
  template <std::convertible_to<T> Y> constexpr void sdiv(Y b) {
    P &self{Self()};
    if constexpr (MatrixDimension<S> && !DenseLayout<S>) {
      ptrdiff_t M = nr(), N = nc();
      POLYMATHNOVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r) {
        POLYMATHIVDEP
        for (ptrdiff_t c = 0; c < N; ++c) self[r, c] /= b;
      }
    } else {
      POLYMATHIVDEP
      for (ptrdiff_t c = 0, L = ptrdiff_t(dim_()); c < L; ++c) self[c] /= b;
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
