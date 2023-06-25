#pragma once

#include "Math/Indexing.hpp"
#include "Math/Matrix.hpp"
#include "Math/MatrixDimensions.hpp"
#include "Math/UniformScaling.hpp"
#include "Math/Vector.hpp"
#include <algorithm>
#include <type_traits>

#if defined __GNUC__ && __GNUC__ >= 8
#define POLYMATHVECTORIZE _Pragma("GCC ivdep")
// #define POLYMATHVECTORIZE _Pragma("GCC unroll 2") _Pragma("GCC ivdep")
#elif defined __clang__
#define POLYMATHVECTORIZE _Pragma("clang loop vectorize(enable)")
// _Pragma("clang loop vectorize(enable) interleave_count(2)")
#else
#define POLYMATHVECTORIZE
#endif

namespace poly::math {
template <typename T> class SmallSparseMatrix;
template <class T, class S, class P> class ArrayOps {
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
  constexpr auto index(ptrdiff_t i) -> T & {
    return (*static_cast<P *>(this))[i];
  }
  constexpr auto index(ptrdiff_t i, ptrdiff_t j) -> T & {
    return (*static_cast<P *>(this))(i, j);
  }
  [[nodiscard]] constexpr auto nr() const -> ptrdiff_t {
    return ptrdiff_t(static_cast<const P *>(this)->numRow());
  }
  [[nodiscard]] constexpr auto nc() const -> ptrdiff_t {
    return ptrdiff_t(static_cast<const P *>(this)->numCol());
  }
  [[nodiscard]] constexpr auto rs() const -> ptrdiff_t {
    return ptrdiff_t(static_cast<const P *>(this)->rowStride());
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
  [[gnu::flatten]] constexpr auto operator<<(const AbstractVector auto &B)
    -> P & {
    if constexpr (MatrixDimension<S>) {
      ptrdiff_t M = nr(), N = nc();
      invariant(M, B.size());
      for (ptrdiff_t i = 0; i < M; ++i) {
        T Bi = B[i];
        POLYMATHVECTORIZE
        for (ptrdiff_t j = 0; j < N; ++j) index(i, j) = Bi;
      }
    } else {
      ptrdiff_t L = size_();
      invariant(L, ptrdiff_t(B.size()));
      POLYMATHVECTORIZE
      for (ptrdiff_t i = 0; i < L; ++i) index(i) = B[i];
    }
    return *static_cast<P *>(this);
  }

  [[gnu::flatten]] constexpr auto operator<<(const AbstractMatrix auto &B)
    -> P & {
    static_assert(MatrixDimension<S>);
    ptrdiff_t M = nr(), N = nc();
    invariant(M, ptrdiff_t(B.numRow()));
    invariant(N, ptrdiff_t(B.numCol()));
    if constexpr (DenseLayout<S> &&
                  DataMatrix<std::remove_cvref_t<decltype(B)>> &&
                  DenseLayout<std::remove_cvref_t<decltype(B.dim())>>) {
      std::copy_n(B.data(), M * N, data_());
    } else {
      for (ptrdiff_t i = 0; i < M; ++i) {
        POLYMATHVECTORIZE
        for (ptrdiff_t j = 0; j < N; ++j) index(i, j) = B(i, j);
      }
    }
    return *static_cast<P *>(this);
  }
  template <std::convertible_to<T> Y>
  [[gnu::flatten]] constexpr auto operator<<(const Y b) -> P & {
    if constexpr (DenseLayout<S>) {
      std::fill_n(data_(), ptrdiff_t(dim_()), T(b));
    } else if constexpr (std::is_same_v<S, StridedRange>) {
      POLYMATHVECTORIZE
      for (ptrdiff_t c = 0, L = size_(); c < L; ++c) index(c) = b;
    } else {
      ptrdiff_t M = nr(), N = nc(), X = rs();
      T *p = data_();
      for (ptrdiff_t r = 0; r < M; ++r, p += X) std::fill_n(p, N, T(b));
    }
    return *static_cast<P *>(this);
  }
  [[gnu::flatten]] constexpr auto operator+=(const AbstractMatrix auto &B)
    -> P & {
    static_assert(MatrixDimension<S>);
    ptrdiff_t M = nr(), N = nc();
    invariant(M, ptrdiff_t(B.numRow()));
    invariant(N, ptrdiff_t(B.numCol()));
    for (ptrdiff_t r = 0; r < M; ++r) {
      POLYMATHVECTORIZE
      for (ptrdiff_t c = 0; c < N; ++c) index(r, c) += B(r, c);
    }
    return *static_cast<P *>(this);
  }
  [[gnu::flatten]] constexpr auto operator-=(const AbstractMatrix auto &B)
    -> P & {
    static_assert(MatrixDimension<S>);
    ptrdiff_t M = nr(), N = nc();
    invariant(M, ptrdiff_t(B.numRow()));
    invariant(N, ptrdiff_t(B.numCol()));
    for (ptrdiff_t r = 0; r < M; ++r) {
      POLYMATHVECTORIZE
      for (ptrdiff_t c = 0; c < N; ++c) index(r, c) -= B(r, c);
    }
    return *static_cast<P *>(this);
  }
  [[gnu::flatten]] constexpr auto operator+=(const AbstractVector auto &B)
    -> P & {
    if constexpr (MatrixDimension<S>) {
      ptrdiff_t M = nr(), N = nc();
      invariant(M, B.size());
      for (ptrdiff_t r = 0; r < M; ++r) {
        auto Br = B[r];
        POLYMATHVECTORIZE
        for (ptrdiff_t c = 0; c < N; ++c) index(r, c) += Br;
      }
    } else {
      ptrdiff_t L = size_();
      invariant(L, ptrdiff_t(B.size()));
      POLYMATHVECTORIZE
      for (ptrdiff_t i = 0; i < L; ++i) index(i) += B[i];
    }
    return *static_cast<P *>(this);
  }
  template <std::convertible_to<T> Y>
  [[gnu::flatten]] constexpr auto operator+=(Y b) -> P & {
    if constexpr (MatrixDimension<S> && !DenseLayout<S>) {
      ptrdiff_t M = nr(), N = nc();
      for (ptrdiff_t r = 0; r < M; ++r) {
        POLYMATHVECTORIZE
        for (ptrdiff_t c = 0; c < N; ++c) index(r, c) += b;
      }
    } else {
      POLYMATHVECTORIZE
      for (ptrdiff_t i = 0, L = size_(); i < L; ++i) index(i) += b;
    }
    return *static_cast<P *>(this);
  }
  [[gnu::flatten]] constexpr auto operator-=(const AbstractVector auto &B)
    -> P & {
    if constexpr (MatrixDimension<S>) {
      ptrdiff_t M = nr(), N = nc();
      invariant(M == B.size());
      for (ptrdiff_t r = 0; r < M; ++r) {
        auto Br = B[r];
        POLYMATHVECTORIZE
        for (ptrdiff_t c = 0; c < N; ++c) index(r, c) -= Br;
      }
    } else {
      ptrdiff_t L = size_();
      invariant(L == B.size());
      POLYMATHVECTORIZE
      for (ptrdiff_t i = 0; i < L; ++i) index(i) -= B[i];
    }
    return *static_cast<P *>(this);
  }
  template <std::convertible_to<T> Y>
  [[gnu::flatten]] constexpr auto operator*=(Y b) -> P & {
    if constexpr (MatrixDimension<S> && !DenseLayout<S>) {
      ptrdiff_t M = nr(), N = nc();
      for (ptrdiff_t r = 0; r < M; ++r) {
        POLYMATHVECTORIZE
        for (ptrdiff_t c = 0; c < N; ++c) index(r, c) *= b;
      }
    } else {
      POLYMATHVECTORIZE
      for (ptrdiff_t c = 0, L = ptrdiff_t(dim_()); c < L; ++c) index(c) *= b;
    }
    return *static_cast<P *>(this);
  }
  template <std::convertible_to<T> Y>
  [[gnu::flatten]] constexpr auto operator/=(Y b) -> P & {
    if constexpr (MatrixDimension<S> && !DenseLayout<S>) {
      ptrdiff_t M = nr(), N = nc();
      for (ptrdiff_t r = 0; r < M; ++r) {
        POLYMATHVECTORIZE
        for (ptrdiff_t c = 0; c < N; ++c) index(r, c) /= b;
      }
    } else {
      POLYMATHVECTORIZE
      for (ptrdiff_t c = 0, L = ptrdiff_t(dim_()); c < L; ++c) index(c) /= b;
    }
    return *static_cast<P *>(this);
  }
};
} // namespace poly::math
