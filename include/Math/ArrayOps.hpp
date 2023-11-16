#pragma once

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

template <typename T>
concept IsOne =
  std::same_as<std::remove_cvref_t<T>, std::integral_constant<ptrdiff_t, 1>>;

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
    static_assert(MatrixDimension<S>);
    P &self{Self()};
    auto [M, N] = shape(self);
    invariant(ptrdiff_t(M), ptrdiff_t(B.numRow()));
    invariant(N, ptrdiff_t(B.numCol()));
    if constexpr (std::same_as<Op, utils::CopyAssign> && DenseLayout<S> &&
                  DenseTensor<std::remove_cvref_t<decltype(B)>>) {
      if constexpr (std::is_trivially_copyable_v<T>)
        std::memcpy(data_(), M * N * sizeof(T), B.begin());
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
      if constexpr (!std::is_copy_assignable_v<T> &&
                    std::same_as<Op, utils::CopyAssign>) {
        POLYMATHVECTORIZE
        for (ptrdiff_t j = 0; j < N; ++j)
          if constexpr (std::convertible_to<decltype(B), T>) self[j] = auto{B};
          else self[j] = auto{B[j]};
      } else {
        POLYMATHVECTORIZE
        for (ptrdiff_t j = 0; j < N; ++j)
          utils::assign(self, B, utils::NoRowIndex{}, j, op);
      }
    } else {
      POLYMATHNOVECTORIZE
      for (ptrdiff_t i = 0; i < M; ++i) {
        if constexpr (!std::is_copy_assignable_v<T> &&
                      std::same_as<Op, utils::CopyAssign>) {
          POLYMATHVECTORIZE
          for (ptrdiff_t j = 0; j < N; ++j)
            if constexpr (std::convertible_to<decltype(B), T>)
              self[i, j] = auto{B};
            else if constexpr (RowVector<decltype(B)>) self[i, j] = auto{B[j]};
            else if constexpr (ColVector<decltype(B)>) self[i, j] = auto{B[i]};
            else self[i, j] = auto{B[i, j]};
        } else {
          POLYMATHVECTORIZE
          for (ptrdiff_t j = 0; j < N; ++j) utils::assign(self, B, i, j, op);
        }
      }
    }
  }
  template <typename Op> constexpr void scopyTo(const auto &B, Op op) {
    static_assert(sizeof(utils::eltype_t<decltype(B)>) <= 8);
    static_assert(MatrixDimension<S>);
    P &self{Self()};
    auto [M, N] = shape(self);
    invariant(ptrdiff_t(M), ptrdiff_t(B.numRow()));
    invariant(N, ptrdiff_t(B.numCol()));
    if constexpr (std::same_as<Op, utils::CopyAssign> && DenseLayout<S> &&
                  DenseTensor<std::remove_cvref_t<decltype(B)>>) {
      if constexpr (std::is_trivially_copyable_v<T>)
        std::memcpy(data_(), M * N * sizeof(T), B.begin());
      else std::copy_n(B.begin(), M * N, data_());
    } else if constexpr (AbstractVector<P>) {
      if constexpr (!std::is_copy_assignable_v<T> &&
                    std::same_as<Op, utils::CopyAssign>) {
        POLYMATHIVDEP
        for (ptrdiff_t j = 0; j < N; ++j)
          if constexpr (std::convertible_to<decltype(B), T>) self[j] = auto{B};
          else self[j] = auto{B[j]};
      } else {
        POLYMATHIVDEP
        for (ptrdiff_t j = 0; j < N; ++j)
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
      if constexpr ((sizeof(T) <= sizeof(double)) &&
                    (!HasInnerReduction<std::remove_cvref_t<decltype(B)>>))
        vcopyTo(B, op);
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
