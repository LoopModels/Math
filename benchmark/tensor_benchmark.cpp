
#include "LoopMacros.hxx"
#include "Math/Array.cxx"
#include "Math/Dual.cxx"
#include "Math/LinearAlgebra.cxx"
#include "Math/MatrixDimensions.cxx"
#include "Math/Ranges.cxx"
#include "Math/StaticArrays.cxx"
#include "Math/UniformScaling.cxx"
#include "SIMD/Intrin.cxx"
#include "Utilities/Invariant.cxx"
#include "Utilities/TypeCompression.cxx"
#include "include/randdual.hpp"
#include <array>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <random>

using math::Dual, math::SquareMatrix, math::SquareDims, math::I, math::URand;

[[gnu::noinline]] static void A12pI120(auto &B, const auto &A) {
  B << 12.0 * A + 120.0 * I;
}
[[gnu::noinline]] static void BApI60(auto &C, const auto &A, const auto &B) {
  C << B * (A + 60.0 * I);
}

static void BM_dual8x2dApI(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<Dual<double, 8>, 2>;
  ptrdiff_t dim = state.range(0);
  SquareMatrix<D> A{SquareDims{math::row(dim)}};
  SquareMatrix<D> B{SquareDims{math::row(dim)}};
  for (auto &&a : A) a = URand<D>{}(rng0);
  for (auto b : state) {
    A12pI120(B, A);
    benchmark::DoNotOptimize(B);
  }
}
BENCHMARK(BM_dual8x2dApI)->DenseRange(2, 10, 1);

static void BM_dual8x2BmApI(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<Dual<double, 8>, 2>;
  ptrdiff_t dim = state.range(0);
  SquareMatrix<D> A{SquareDims{math::row(dim)}};
  SquareMatrix<D> B{SquareDims{math::row(dim)}};
  SquareMatrix<D> C{SquareDims{math::row(dim)}};
  for (auto &&a : A) a = URand<D>{}(rng0);
  for (auto &&b : B) b = URand<D>{}(rng0);
  for (auto b : state) {
    BApI60(C, A, B);
    benchmark::DoNotOptimize(C);
  }
}
BENCHMARK(BM_dual8x2BmApI)->DenseRange(2, 10, 1);

template <size_t M, size_t N>
void BtimesAplusdI(
  math::MutSquarePtrMatrix<std::array<std::array<double, M>, N>> C,
  math::MutSquarePtrMatrix<std::array<std::array<double, M>, N>> A,
  math::MutSquarePtrMatrix<std::array<std::array<double, M>, N>> B,
  double doffset) {
  using T = std::array<std::array<double, M>, N>;
  utils::invariant(C.numRow() == A.numRow());
  utils::invariant(C.numRow() == B.numRow());
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
  SquareMatrix<D> A{SquareDims{math::row(dim)}};
  SquareMatrix<D> B{SquareDims{math::row(dim)}};
  SquareMatrix<D> C{SquareDims{math::row(dim)}};
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
  static_assert(utils::Compressible<Dual<double, 7>>);
  static_assert(utils::Compressible<D>);
  static_assert(sizeof(utils::compressed_t<D>) == (24 * sizeof(double)));
  static_assert(sizeof(D) == (24 * sizeof(double)));
  ptrdiff_t dim = state.range(0);
  SquareMatrix<D> A{SquareDims{math::row(dim)}};
  SquareMatrix<D> B{SquareDims{math::row(dim)}};
  for (auto &&a : A) a = URand<D>{}(rng0);
  for (auto b : state) {
    A12pI120(B, A);
    benchmark::DoNotOptimize(B);
  }
}
BENCHMARK(BM_dual7x2dApI)->DenseRange(2, 10, 1);

static void BM_dual7x2BmApI(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<Dual<double, 7>, 2>;
  ptrdiff_t dim = state.range(0);
  SquareMatrix<D> A{SquareDims{math::row(dim)}};
  SquareMatrix<D> B{SquareDims{math::row(dim)}};
  SquareMatrix<D> C{SquareDims{math::row(dim)}};
  for (auto &&a : A) a = URand<D>{}(rng0);
  for (auto &&b : B) b = URand<D>{}(rng0);
  for (auto b : state) {
    BApI60(C, A, B);
    benchmark::DoNotOptimize(C);
  }
}
BENCHMARK(BM_dual7x2BmApI)->DenseRange(2, 10, 1);

static void BM_dual7x2BmApI_manual(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = std::array<std::array<double, 8>, 3>;
  ptrdiff_t dim = state.range(0);
  SquareMatrix<D> A{SquareDims{math::row(dim)}};
  SquareMatrix<D> B{SquareDims{math::row(dim)}};
  SquareMatrix<D> C{SquareDims{math::row(dim)}};
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

static void BM_dual6x2dApI(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<Dual<double, 6>, 2>;
  static_assert(utils::Compressible<Dual<double, 6>>);
  static_assert(utils::Compressible<D>);
  static_assert(sizeof(utils::compressed_t<D>) == (21 * sizeof(double)));
  static_assert(simd::SIMDSupported<double> ==
                (sizeof(math::SVector<double, 7>) == (8 * sizeof(double))));
  static_assert(simd::SIMDSupported<double> ==
                (sizeof(Dual<double, 6>) == (8 * sizeof(double))));
  static_assert(simd::SIMDSupported<double> ==
                (sizeof(Dual<double, 6>) == (8 * sizeof(double))));
  static_assert(simd::SIMDSupported<double> ==
                (sizeof(D) == (24 * sizeof(double))));
  // static_assert(sizeof(D) == sizeof(Dual<Dual<double, 8>, 2>));
  ptrdiff_t dim = state.range(0);
  SquareMatrix<D> A{SquareDims{math::row(dim)}};
  SquareMatrix<D> B{SquareDims{math::row(dim)}};
  for (auto &&a : A) a = URand<D>{}(rng0);
  for (auto b : state) {
    A12pI120(B, A);
    benchmark::DoNotOptimize(B);
  }
}
BENCHMARK(BM_dual6x2dApI)->DenseRange(2, 10, 1);

static void BM_dual6x2BmApI(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<Dual<double, 6>, 2>;
  ptrdiff_t dim = state.range(0);
  SquareMatrix<D> A{SquareDims{math::row(dim)}};
  SquareMatrix<D> B{SquareDims{math::row(dim)}};
  SquareMatrix<D> C{SquareDims{math::row(dim)}};
  for (auto &&a : A) a = URand<D>{}(rng0);
  for (auto &&b : B) b = URand<D>{}(rng0);
  for (auto b : state) {
    BApI60(C, A, B);
    benchmark::DoNotOptimize(C);
  }
}
BENCHMARK(BM_dual6x2BmApI)->DenseRange(2, 10, 1);

static void BM_dual6x2BmApI_manual(benchmark::State &state) {
  std::mt19937_64 rng0;
  constexpr size_t Dcount = 6;
  constexpr size_t N = Dcount + 1;
  using D = std::array<std::array<double, N>, 3>;
  ptrdiff_t dim = state.range(0);
  SquareMatrix<D> A{SquareDims{math::row(dim)}};
  SquareMatrix<D> B{SquareDims{math::row(dim)}};
  SquareMatrix<D> C{SquareDims{math::row(dim)}};
  for (ptrdiff_t i = 0, L = dim * dim; i < L; ++i) {
    for (ptrdiff_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < N; ++k) {
        A[i][j][k] = URand<double>{}(rng0);
        B[i][j][k] = URand<double>{}(rng0);
      }
    }
  }
  for (auto b : state) BtimesAplusdI(C, A, B, 60.0);
}
BENCHMARK(BM_dual6x2BmApI_manual)->DenseRange(2, 10, 1);
