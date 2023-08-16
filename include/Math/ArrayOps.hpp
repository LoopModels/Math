#pragma once

#include "Math/Indexing.hpp"
#include "Math/Matrix.hpp"
#include "Math/MatrixDimensions.hpp"
#include "Math/SIMD.hpp"
#include "Math/SIMDWidth.hpp"
#include "Math/UniformScaling.hpp"
#include "Math/Vector.hpp"
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <eve/module/core.hpp>
#include <type_traits>
// #include <eve/module/algo.hpp>

#if defined __GNUC__ && __GNUC__ >= 8
#define POLYMATHVECTORIZE _Pragma("GCC ivdep")
// #define POLYMATHVECTORIZE _Pragma("GCC unroll 2") _Pragma("GCC ivdep")
// #elif defined __clang__
// #define POLYMATHVECTORIZE _Pragma("clang loop vectorize(enable)")
// _Pragma("clang loop vectorize(enable) interleave_count(2)")
#else
#define POLYMATHVECTORIZE
#endif

namespace poly::math {

template <typename T>
concept HasLength = requires(T x) {
  { ptrdiff_t(x.dim()) } -> std::same_as<ptrdiff_t>;
};
template <typename T>
concept HasPaddedLength = HasLength<T> && requires(T x) {
  { x.paddedlength() } -> std::convertible_to<ptrdiff_t>;
};

constexpr auto paddedlen(const HasPaddedLength auto &x) -> ptrdiff_t {
  return ptrdiff_t(x.paddedlength());
}
constexpr auto paddedlen(const HasLength auto &x) -> ptrdiff_t {
  return ptrdiff_t(x.dim());
}

template <typename T> class SmallSparseMatrix;
template <class T, class S, class P> class ArrayOps {
  [[gnu::returns_nonnull]] constexpr auto data_() -> T * {
    return static_cast<P *>(this)->data();
  }
  [[gnu::returns_nonnull]] constexpr auto data_() const -> const T * {
    return static_cast<const P *>(this)->data();
  }
  constexpr auto size_() const { return static_cast<const P *>(this)->size(); }
  constexpr auto paddedlen_() const {
    return paddedlen(*static_cast<const P *>(this));
  }
  constexpr auto dim_() const -> S {
    return static_cast<const P *>(this)->dim();
  }
  constexpr auto index(ptrdiff_t i) -> decltype(auto) {
    return (*static_cast<P *>(this))[i];
  }
  constexpr auto index(ptrdiff_t i, ptrdiff_t j) -> decltype(auto) {
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
  constexpr auto self() -> P & { return *static_cast<P *>(this); }
  template <std::convertible_to<T> Y>
  [[gnu::flatten]] constexpr auto operator<<(const UniformScaling<Y> &B)
    -> P & {
    static_assert(MatrixDimension<S>);
    std::fill_n(data_(), ptrdiff_t(this->dim()), T{});
    this->diag() << B.value;
    return self();
  }
  [[gnu::flatten]] constexpr auto operator<<(const SmallSparseMatrix<T> &B)
    -> P &;
  [[gnu::flatten]] constexpr auto operator<<(const AbstractVector auto &B)
    -> P & {
    if constexpr (MatrixDimension<S>) {
      ptrdiff_t M = nr();
      invariant(M, B.size());
      POLYMATHVECTORIZE
      for (ptrdiff_t i = 0; i < M; ++i) static_cast<P *>(this)(i, _) << B[i];
    } else {
      constexpr ptrdiff_t W = simd::vecWidth<T, S>();
      if constexpr (PrimitiveScalar<T> && (W > 1)) {
        invariant(ptrdiff_t(size_()), ptrdiff_t(B.size()));
        ptrdiff_t L = std::min<ptrdiff_t>(paddedlen_(), paddedlen(B));
        ptrdiff_t i = 0, n = W;
        auto &A{self()};
        POLYMATHVECTORIZE
        for (; n <= L; i = n, n += W) {
          auto j = simd::unroll<W, 1>(i);
          A[j] = B[j];
        }
        if (i < L) {
          auto j = simd::unroll<W, 1>(i, L);
          A[j] = B[j];
        }
      } else {
        ptrdiff_t L = size_();
        invariant(L, ptrdiff_t(B.size()));
        using BType = std::remove_cvref_t<decltype(B)>;
        if constexpr (HasDataPtr<BType> && Trivial<BType>) {
          std::memcpy(data_(), B.data(), L * sizeof(T));
        } else {
          POLYMATHVECTORIZE
          for (ptrdiff_t i = 0; i < L; ++i) index(i) = B[i];
        }
      }
    }
    return self();
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
      POLYMATHVECTORIZE
      for (ptrdiff_t i = 0; i < M; ++i) {
        POLYMATHVECTORIZE
        for (ptrdiff_t j = 0; j < N; ++j) index(i, j) = B(i, j);
      }
    }
    return self();
  }
  [[gnu::flatten]] constexpr auto
  operator<<(const std::convertible_to<T> auto &b) -> P & {
    if constexpr (DenseLayout<S>) {
      constexpr ptrdiff_t W = simd::vecWidth<T, S>();
      if constexpr (PrimitiveScalar<T> && (W > 1)) {
        // eve::algo::fill[eve::algo::allow_frequency_scaling](
        //                                                     data_(), data_()
        //                                                     +
        //                                                     ptrdiff_t(dim_()),
        //                                                     T(b));
        ptrdiff_t L = paddedlen_();
        ptrdiff_t i = 0, n = W;
        auto &A{self()};
        POLYMATHVECTORIZE
        for (; n <= L; i = n, n += W) {
          auto j = simd::unroll<W, 1>(i);
          A[j] = b;
        }
        if (i < L) {
          auto j = simd::unroll<W, 1>(i, L);
          A[j] = b;
        }
      } else {
        std::fill_n(data_(), ptrdiff_t(dim_()), T(b));
      }
    } else if constexpr (std::is_same_v<S, StridedRange>) {
      POLYMATHVECTORIZE
      for (ptrdiff_t c = 0, L = size_(); c < L; ++c) index(c) = b;
    } else {
      ptrdiff_t M = nr(), N = nc(), X = rs();
      T *p = data_();
      // for (ptrdiff_t r = 0; r < M; ++r, p += X) std::fill_n(p, N, b);
      POLYMATHVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r, p += X) {
        POLYMATHVECTORIZE
        for (ptrdiff_t c = 0; c < N; ++c) p[c] = b;
      }
    }
    return self();
  }
  [[gnu::flatten]] constexpr auto operator+=(const AbstractMatrix auto &B)
    -> P & {
    static_assert(MatrixDimension<S>);
    ptrdiff_t M = nr(), N = nc();
    invariant(M, ptrdiff_t(B.numRow()));
    invariant(N, ptrdiff_t(B.numCol()));
    POLYMATHVECTORIZE
    for (ptrdiff_t r = 0; r < M; ++r) {
      POLYMATHVECTORIZE
      for (ptrdiff_t c = 0; c < N; ++c) index(r, c) += B(r, c);
    }
    return self();
  }
  [[gnu::flatten]] constexpr auto operator-=(const AbstractMatrix auto &B)
    -> P & {
    static_assert(MatrixDimension<S>);
    ptrdiff_t M = nr(), N = nc();
    invariant(M, ptrdiff_t(B.numRow()));
    invariant(N, ptrdiff_t(B.numCol()));
    POLYMATHVECTORIZE
    for (ptrdiff_t r = 0; r < M; ++r) {
      POLYMATHVECTORIZE
      for (ptrdiff_t c = 0; c < N; ++c) index(r, c) -= B(r, c);
    }
    return self();
  }
  [[gnu::flatten]] constexpr auto operator+=(const AbstractVector auto &B)
    -> P & {
    if constexpr (MatrixDimension<S>) {
      ptrdiff_t M = nr(), N = nc();
      invariant(M, B.size());
      POLYMATHVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r) {
        auto Br = B[r];
        POLYMATHVECTORIZE
        for (ptrdiff_t c = 0; c < N; ++c) index(r, c) += Br;
      }
    } else {
      constexpr ptrdiff_t W = simd::vecWidth<T, S>();
      if constexpr (PrimitiveScalar<T> && (W > 1)) {
        invariant(ptrdiff_t(size_()), ptrdiff_t(B.size()));
        ptrdiff_t L = std::min<ptrdiff_t>(paddedlen_(), paddedlen(B));
        ptrdiff_t i = 0, n = W;
        auto &A{self()};
        const auto &cA{self()};
        POLYMATHVECTORIZE
        for (; n <= L; i = n, n += W) {
          auto j = simd::unroll<W, 1>(i);
          A[j] = cA[j] + B[j];
        }
        if (i < L) {
          auto j = simd::unroll<W, 1>(i, L);
          A[j] = cA[j] + B[j];
        }
      } else {
        ptrdiff_t L = size_();
        invariant(L, ptrdiff_t(B.size()));
        POLYMATHVECTORIZE
        for (ptrdiff_t i = 0; i < L; ++i) index(i) += B[i];
      }
    }
    return self();
  }
  [[gnu::flatten]] constexpr auto
  operator+=(const std::convertible_to<T> auto &b) -> P & {
    if constexpr (MatrixDimension<S> && !DenseLayout<S>) {
      ptrdiff_t M = nr(), N = nc();
      POLYMATHVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r) {
        POLYMATHVECTORIZE
        for (ptrdiff_t c = 0; c < N; ++c) index(r, c) += b;
      }
    } else {
      POLYMATHVECTORIZE
      for (ptrdiff_t i = 0, L = size_(); i < L; ++i) index(i) += b;
    }
    return self();
  }
  [[gnu::flatten]] constexpr auto operator-=(const AbstractVector auto &B)
    -> P & {
    if constexpr (MatrixDimension<S>) {
      ptrdiff_t M = nr(), N = nc();
      invariant(M == B.size());
      POLYMATHVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r) {
        auto Br = B[r];
        POLYMATHVECTORIZE
        for (ptrdiff_t c = 0; c < N; ++c) index(r, c) -= Br;
      }
    } else {
      constexpr ptrdiff_t W = simd::vecWidth<T, S>();
      if constexpr (PrimitiveScalar<T> && (W > 1)) {
        invariant(ptrdiff_t(size_()), ptrdiff_t(B.size()));
        ptrdiff_t L = std::min<ptrdiff_t>(paddedlen_(), paddedlen(B)), i = 0,
                  n = W;
        auto &A{self()};
        const auto &cA{self()};
        POLYMATHVECTORIZE
        for (; n <= L; i = n, n += W) {
          auto j = simd::unroll<W, 1>(i);
          A[j] = cA[j] - B[j];
        }
        if (i < L) {
          auto j = simd::unroll<W, 1>(i, L);
          A[j] = cA[j] - B[j];
        }
      } else {
        ptrdiff_t L = size_();
        invariant(L == B.size());
        POLYMATHVECTORIZE
        for (ptrdiff_t i = 0; i < L; ++i) index(i) -= B[i];
      }
    }
    return self();
  }
  [[gnu::flatten]] constexpr auto
  operator*=(const std::convertible_to<T> auto &b) -> P & {
    if constexpr (MatrixDimension<S> && !DenseLayout<S>) {
      ptrdiff_t M = nr(), N = nc();
      POLYMATHVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r) {
        POLYMATHVECTORIZE
        for (ptrdiff_t c = 0; c < N; ++c) index(r, c) *= b;
      }
    } else {
      constexpr ptrdiff_t W = simd::vecWidth<T, S>();
      if constexpr (PrimitiveScalar<T> && (W > 1)) {
        ptrdiff_t L = paddedlen_(), i = 0, n = W;
        auto &A{self()};
        const auto &cA{self()};
        POLYMATHVECTORIZE
        for (; n <= L; i = n, n += W) {
          auto j = simd::unroll<W, 1>(i);
          A[j] = cA[j] * b;
        }
        if (i < L) {
          auto j = simd::unroll<W, 1>(i, L);
          A[j] = cA[j] * b;
        }
      } else {
        POLYMATHVECTORIZE
        for (ptrdiff_t c = 0, L = ptrdiff_t(dim_()); c < L; ++c) index(c) *= b;
      }
    }
    return self();
  }
  [[gnu::flatten]] constexpr auto
  operator/=(const std::convertible_to<T> auto &b) -> P & {
    if constexpr (MatrixDimension<S> && !DenseLayout<S>) {
      ptrdiff_t M = nr(), N = nc();
      POLYMATHVECTORIZE
      for (ptrdiff_t r = 0; r < M; ++r) {
        POLYMATHVECTORIZE
        for (ptrdiff_t c = 0; c < N; ++c) index(r, c) /= b;
      }
    } else {
      constexpr ptrdiff_t W = simd::vecWidth<T, S>();
      if constexpr (std::floating_point<T> && (W > 1)) {
        ptrdiff_t L = paddedlen_();
        ptrdiff_t i = 0, n = W;
        auto &A{self()};
        const auto &cA{self()};
        POLYMATHVECTORIZE
        for (; n <= L; i = n, n += W) {
          auto j = simd::unroll<W, 1>(i);
          A[j] = cA[j] / b;
        }
        if (i < L) {
          auto j = simd::unroll<W, 1>(i, L);
          A[j] = cA[j] / b;
        }
      } else {
        POLYMATHVECTORIZE
        for (ptrdiff_t c = 0, L = ptrdiff_t(dim_()); c < L; ++c) index(c) /= b;
      }
    }
    return self();
  }
};
} // namespace poly::math
