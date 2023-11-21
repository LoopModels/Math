
#include "include/randdual.hpp"
#include <Math/Array.hpp>
#include <Math/Dual.hpp>
#include <Math/LinearAlgebra.hpp>
#include <Math/Matrix.hpp>
#include <Math/StaticArrays.hpp>
#include <Utilities/Invariant.hpp>
#include <algorithm>
#include <array>
#include <benchmark/benchmark.h>
#include <concepts>
#include <cstdint>
#include <random>
#include <ranges>

using poly::math::Dual, poly::math::SquareMatrix, poly::math::SquareDims,
  poly::math::I, poly::math::URand;

static void BM_dual8x2dApI(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<Dual<double, 8>, 2>;
  ptrdiff_t dim = state.range(0);
  SquareMatrix<D> A{SquareDims{{dim}}};
  SquareMatrix<D> B{SquareDims{{dim}}};
  for (auto &&a : A) a = URand<D>{}(rng0);
  for (auto b : state) benchmark::DoNotOptimize(B << 12.0 * A + 120.0 * I);
}
BENCHMARK(BM_dual8x2dApI)->DenseRange(2, 10, 1);

static void BM_dual8x2BmApI(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<Dual<double, 8>, 2>;
  ptrdiff_t dim = state.range(0);
  SquareMatrix<D> A{SquareDims{{dim}}};
  SquareMatrix<D> B{SquareDims{{dim}}};
  SquareMatrix<D> C{SquareDims{{dim}}};
  for (auto &&a : A) a = URand<D>{}(rng0);
  for (auto &&b : B) b = URand<D>{}(rng0);
  for (auto b : state) benchmark::DoNotOptimize(C << B * (A + 60.0 * I));
}
BENCHMARK(BM_dual8x2BmApI)->DenseRange(2, 10, 1);

template <size_t M, size_t N>
void BtimesAplusdI(
  poly::math::MutSquarePtrMatrix<std::array<std::array<double, M>, N>> C,
  poly::math::MutSquarePtrMatrix<std::array<std::array<double, M>, N>> A,
  poly::math::MutSquarePtrMatrix<std::array<std::array<double, M>, N>> B,
  double doffset) {
  using T = std::array<std::array<double, M>, N>;
  poly::utils::invariant(C.numRow() == A.numRow());
  poly::utils::invariant(C.numRow() == B.numRow());
  ptrdiff_t D = ptrdiff_t(C.numRow());
  POLYMATHNOVECTORIZE
  for (ptrdiff_t r = 0; r < D; ++r) {
    POLYMATHNOVECTORIZE
    for (ptrdiff_t c = 0; c < D; ++c) {
      T x{};
      POLYMATHNOVECTORIZE
      for (ptrdiff_t k = 0; k < D; ++k) {
        // x += B[r, k] * A[k, c];
        T &Brk = B[r, k];
        T &Akc = A[k, c];
        // x[0] += Brk[0] * Akc[0];
        x[0][0] += Brk[0][0] * (Akc[0][0] + doffset * (r == c));
        POLYMATHVECTORIZE
        for (size_t i = 1; i < M; ++i)
          x[0][i] += Brk[0][0] * Akc[0][i] + Brk[0][i] * Akc[0][0];
        POLYMATHNOVECTORIZE
        for (size_t o = 1; o < N; ++o) {
          // x[o] += Brk[0]*Akc[o];
          x[o][0] += Brk[0][0] * Akc[o][0];
          POLYMATHVECTORIZE
          for (size_t i = 1; i < M; ++i)
            x[o][i] += Brk[0][0] * Akc[o][i] + Brk[0][i] * Akc[o][0];
          // x[o] += Brk[o]*Akc[0];
          x[o][0] += Brk[o][0] * Akc[0][0];
          POLYMATHVECTORIZE
          for (size_t i = 1; i < M; ++i)
            x[o][i] += Brk[o][0] * Akc[0][i] + Brk[o][i] * Akc[0][0];
        }
      }
      C[r, c] = x;
    }
  }
}

static void BM_dual8x2BmApI_manual(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = std::array<std::array<double, 9>, 3>;
  ptrdiff_t dim = state.range(0);
  SquareMatrix<D> A{SquareDims{{dim}}};
  SquareMatrix<D> B{SquareDims{{dim}}};
  SquareMatrix<D> C{SquareDims{{dim}}};
  for (ptrdiff_t i = 0, L = dim * dim; i < L; ++i) {
    for (ptrdiff_t j = 0; j < 3; ++j) {
      for (ptrdiff_t k = 0; k < 9; ++k) {
        A[i][j][k] = URand<double>{}(rng0);
        B[i][j][k] = URand<double>{}(rng0);
      }
    }
  }
  for (auto b : state) BtimesAplusdI(C, A, B, 60.0);
}
BENCHMARK(BM_dual8x2BmApI_manual)->DenseRange(2, 10, 1);

static void BM_dual7x2dApI(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<Dual<double, 7>, 2>;
  ptrdiff_t dim = state.range(0);
  SquareMatrix<D> A{SquareDims{{dim}}};
  SquareMatrix<D> B{SquareDims{{dim}}};
  for (auto &&a : A) a = URand<D>{}(rng0);
  for (auto b : state) benchmark::DoNotOptimize(B << 12.0 * A + 120.0 * I);
}
BENCHMARK(BM_dual7x2dApI)->DenseRange(2, 10, 1);

static void BM_dual7x2BmApI(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<Dual<double, 7>, 2>;
  ptrdiff_t dim = state.range(0);
  SquareMatrix<D> A{SquareDims{{dim}}};
  SquareMatrix<D> B{SquareDims{{dim}}};
  SquareMatrix<D> C{SquareDims{{dim}}};
  for (auto &&a : A) a = URand<D>{}(rng0);
  for (auto &&b : B) b = URand<D>{}(rng0);
  for (auto b : state) benchmark::DoNotOptimize(C << B * (A + 60.0 * I));
}
BENCHMARK(BM_dual7x2BmApI)->DenseRange(2, 10, 1);

static void BM_dual7x2BmApI_manual(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = std::array<std::array<double, 8>, 3>;
  ptrdiff_t dim = state.range(0);
  SquareMatrix<D> A{SquareDims{{dim}}};
  SquareMatrix<D> B{SquareDims{{dim}}};
  SquareMatrix<D> C{SquareDims{{dim}}};
  for (ptrdiff_t i = 0, L = dim * dim; i < L; ++i) {
    for (ptrdiff_t j = 0; j < 3; ++j) {
      for (ptrdiff_t k = 0; k < 8; ++k) {
        A[i][j][k] = URand<double>{}(rng0);
        B[i][j][k] = URand<double>{}(rng0);
      }
    }
  }
  for (auto b : state) BtimesAplusdI(C, A, B, 60.0);
}
BENCHMARK(BM_dual7x2BmApI_manual)->DenseRange(2, 10, 1);

