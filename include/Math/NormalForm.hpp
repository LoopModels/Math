#pragma once

#include "Containers/Tuple.hpp"
#include "Math/Array.hpp"
#include "Math/Constructors.hpp"
#include "Math/EmptyArrays.hpp"
#include "Math/GreatestCommonDivisor.hpp"
#include "Math/Math.hpp"
#include "Math/MatrixDimensions.hpp"
#include "Math/VectorGreatestCommonDivisor.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace poly::math::NormalForm {
using containers::Tuple, containers::tie;

constexpr auto gcdxScale(int64_t a, int64_t b) -> std::array<int64_t, 4> {
  if (constexpr_abs(a) == 1) return {a, 0, a, b};
  auto [g, p, q, adg, bdg] = dgcdx(a, b);
  return {p, q, adg, bdg};
}
// zero out below diagonal
constexpr void zeroSupDiagonal(MutPtrMatrix<int64_t> A,
                               MutSquarePtrMatrix<int64_t> K, ptrdiff_t i,
                               Row<> M, Col<> N) {
  ptrdiff_t minMN = std::min(ptrdiff_t(M), ptrdiff_t(N));
  for (ptrdiff_t j = i + 1; j < M; ++j) {
    int64_t Aii = A[i, i];
    if (int64_t Aji = A[j, i]) {
      const auto [p, q, Aiir, Aijr] = gcdxScale(Aii, Aji);

      {
        MutPtrVector<int64_t> Ai{A[i, _(0, minMN)]}, Aj{A[j, _(0, minMN)]},
          Ki{K[i, _(0, minMN)]}, Kj{K[j, _(0, minMN)]};
        tie(Ai, Aj, Ki, Kj) << Tuple(p * Ai + q * Aj, Aiir * Aj - Aijr * Ai,
                                     p * Ki + q * Kj, Aiir * Kj - Aijr * Ki);
      }
      if (ptrdiff_t(M) > ptrdiff_t(N)) {
        MutPtrVector<int64_t> Ki{K[i, _(N, M)]}, Kj{K[j, _(N, M)]};
        tie(Ki, Kj) << Tuple(p * Ki + q * Kj, Aiir * Kj - Aijr * Ki);
      } else if (ptrdiff_t(N) > ptrdiff_t(M)) {
        MutPtrVector<int64_t> Ai{A[i, _(M, N)]}, Aj{A[j, _(M, N)]};
        tie(Ai, Aj) << Tuple(p * Ai + q * Aj, Aiir * Aj - Aijr * Ai);
      }
      // for (ptrdiff_t k = 0; k < minMN; ++k) {
      //   int64_t Aki = A[i, k];
      //   int64_t Akj = A[j, k];
      //   int64_t Kki = K[i, k];
      //   int64_t Kkj = K[j, k];
      //   // when k == i, then
      //   // p * Aii + q * Akj == r, so we set A(i,i) = r
      //   A[i, k] = p * Aki + q * Akj;
      //   // Aii/r * Akj - Aij/r * Aki = 0
      //   A[j, k] = Aiir * Akj - Aijr * Aki;
      //   // Mirror for K
      //   K[i, k] = p * Kki + q * Kkj;
      //   K[j, k] = Aiir * Kkj - Aijr * Kki;
      // }
      // for (auto k = ptrdiff_t(N); k < M; ++k) {
      //   int64_t Kki = K[i, k];
      //   int64_t Kkj = K[j, k];
      //   K[i, k] = p * Kki + q * Kkj;
      //   K[j, k] = Aiir * Kkj - Aijr * Kki;
      // }
      // for (auto k = ptrdiff_t(M); k < N; ++k) {
      //   int64_t Aki = A[i, k];
      //   int64_t Akj = A[j, k];
      //   A[i, k] = p * Aki + q * Akj;
      //   A[j, k] = Aiir * Akj - Aijr * Aki;
      // }
    }
  }
}
// This method is only called by orthogonalize, hence we can assume
// (Akk == 1) || (Akk == -1)
constexpr void zeroSubDiagonal(MutPtrMatrix<int64_t> A,
                               MutSquarePtrMatrix<int64_t> K, ptrdiff_t k,
                               Row<> M, Col<> N) {
  int64_t Akk = A[k, k];
  if (Akk == -1) {
    for (ptrdiff_t m = 0; m < N; ++m) A[k, m] *= -1;
    for (ptrdiff_t m = 0; m < M; ++m) K[k, m] *= -1;
  } else {
    invariant(Akk == 1);
  }
  ptrdiff_t Mi = ptrdiff_t(M), Ni = ptrdiff_t(N), minMN = std::min(Mi, Ni);
  for (ptrdiff_t z = 0; z < k; ++z) {
    // eliminate `A(k,z)`
    int64_t Akz = A[z, k];
    if (!Akz) continue;
    // A(k, k) == 1, so A(k,z) -= Akz * 1;
    // A(z,_) -= Akz * A(k,_);
    // K(z,_) -= Akz * K(k,_);
    Tuple(A[z, _(0, minMN)], K[z, _(0, minMN)]) -=
      Tuple(Akz * A[k, _(0, minMN)], Akz * K[k, _(0, minMN)]);
    if (Mi > Ni) K[z, _(Ni, Mi)] -= Akz * K[k, _(Ni, Mi)];
    else if (Ni > Mi) A[z, _(Mi, Ni)] -= Akz * A[k, _(Mi, Ni)];
    // for (ptrdiff_t i = 0; i < minMN; ++i) {
    //   A[z, i] -= Akz * A[k, i];
    //   K[z, i] -= Akz * K[k, i];
    // }
    // for (auto i = ptrdiff_t(N); i < M; ++i) K[z, i] -= Akz * K[k, i];
    // for (auto i = ptrdiff_t(M); i < N; ++i) A[z, i] -= Akz * A[k, i];
  }
}

constexpr auto pivotRowsPair(std::array<MutPtrMatrix<int64_t>, 2> AK, Col<> i,
                             Row<> M, Row<> piv) -> bool {
  Row j = piv;
  while (AK[0][piv, i] == 0)
    if (++piv == M) return true;
  if (j != piv) {
    math::swap(AK[0], j, piv);
    math::swap(AK[1], j, piv);
  }
  return false;
}
constexpr auto pivotRows(MutPtrMatrix<int64_t> A, MutSquarePtrMatrix<int64_t> K,
                         ptrdiff_t i, Row<> M) -> bool {
  MutPtrMatrix<int64_t> B = K;
  return pivotRowsPair({A, B}, Col<>{i}, M, Row<>{i});
}
constexpr auto pivotRows(MutPtrMatrix<int64_t> A, Col<> i, Row<> M, Row<> piv)
  -> bool {
  Row j = piv;
  while (A[piv, i] == 0)
    if (++piv == ptrdiff_t(M)) return true;
  if (j != piv) swap(A, j, piv);
  return false;
}
constexpr auto pivotRows(MutPtrMatrix<int64_t> A, ptrdiff_t i, Row<> N)
  -> bool {
  return pivotRows(A, Col<>{i}, N, Row<>{i});
}

constexpr void dropCol(MutPtrMatrix<int64_t> A, ptrdiff_t i, Row<> M, Col<> N) {
  // if any rows are left, we shift them up to replace it
  if (N <= i) return;
  A[_(0, M), _(i, N)] << A[_(0, M), _(i, N) + 1];
  // for (ptrdiff_t m = 0; m < M; ++m) A[m, _(i, N)] << A[m, _(i, N) + 1];
  // for (ptrdiff_t n = i; n < N; ++n) A[m, n] = A[m, n + 1];
}

constexpr auto orthogonalizeBang(MutDensePtrMatrix<int64_t> &A)
  -> std::pair<SquareMatrix<int64_t>, Vector<unsigned>> {
  // we try to orthogonalize with respect to as many rows of `A` as we can
  // prioritizing earlier rows.
  auto [M, N] = shape(A);
  SquareMatrix<int64_t> K{identity(math::DefaultAlloc<int64_t>{}, unsigned(M))};
  Vector<unsigned> included;
  included.reserve(std::min(ptrdiff_t(M), ptrdiff_t(N)));
  for (ptrdiff_t i = 0, j = 0; i < std::min(ptrdiff_t(M), ptrdiff_t(N)); ++j) {
    // zero ith row
    if (pivotRows(A, K, i, Row<>{M})) {
      // cannot pivot, this is a linear combination of previous
      // therefore, we drop the row
      dropCol(A, i, Row<>{M}, Col<>{--N});
    } else {
      zeroSupDiagonal(A, K, i, Row<>{M}, Col<>{N});
      int64_t Aii = A[i, i];
      if (constexpr_abs(Aii) != 1) {
        // including this row renders the matrix not unimodular!
        // therefore, we drop the row.
        dropCol(A, i, Row<>{M}, Col<>{--N});
      } else {
        // we zero the sub diagonal
        zeroSubDiagonal(A, K, i++, Row<>{M}, Col<>{N});
        included.push_back(j);
      }
    }
  }
  return std::make_pair(std::move(K), std::move(included));
}

constexpr auto orthogonalize(IntMatrix<> A)
  -> std::pair<SquareMatrix<int64_t>, Vector<unsigned>> {
  return orthogonalizeBang(A);
}

constexpr void zeroSupDiagonal(MutPtrMatrix<int64_t> A, Col<> c, Row<> r) {
  auto [M, N] = shape(A);
  for (ptrdiff_t j = ptrdiff_t(r) + 1; j < M; ++j) {
    int64_t Aii = A[r, c];
    if (int64_t Aij = A[j, c]) {
      const auto [p, q, Aiir, Aijr] = gcdxScale(Aii, Aij);
      MutPtrVector<int64_t> Ar = A[r, _], Aj = A[j, _];
      tie(Ar, Aj) << Tuple(p * Ar + q * Aj, Aiir * Aj - Aijr * Ar);
    }
  }
}
constexpr void zeroSupDiagonal(std::array<MutPtrMatrix<int64_t>, 2> AB, Col<> c,
                               Row<> r) {
  auto [A, B] = AB;
  auto [M, N] = shape(A);
  invariant(M, ptrdiff_t(B.numRow()));
  for (ptrdiff_t j = ptrdiff_t(r) + 1; j < M; ++j) {
    int64_t Aii = A[r, c], Aij = A[j, c];
    if (!Aij) continue;
    const auto [p, q, Aiir, Aijr] = gcdxScale(Aii, Aij);
    MutPtrVector<int64_t> Ar = A[r, _], Aj = A[j, _];
    tie(Ar, Aj) << Tuple(p * Ar + q * Aj, Aiir * Aj - Aijr * Ar);
    MutPtrVector<int64_t> Br = B[r, _], Bj = B[j, _];
    tie(Br, Bj) << Tuple(p * Br + q * Bj, Aiir * Bj - Aijr * Br);
  }
}
constexpr void reduceSubDiagonal(MutPtrMatrix<int64_t> A, Col<> c, Row<> r) {
  int64_t Akk = A[r, c];
  if (Akk < 0) {
    Akk = -Akk;
    A[r, _] *= -1;
  }
  for (ptrdiff_t z = 0; z < r; ++z) {
    // try to eliminate `A(k,z)`
    // if Akk == 1, then this zeros out Akz
    if (int64_t Azr = A[z, c]) {
      // we want positive but smaller subdiagonals
      // e.g., `Akz = 5, Akk = 2`, then in the loop below when `i=k`, we
      // set A(k,z) = A(k,z) - (A(k,z)/Akk) * Akk
      //        =   5 - 2*2 = 1
      // or if `Akz = -5, Akk = 2`, then in the loop below we get
      // A(k,z) = A(k,z) - ((A(k,z)/Akk) - ((A(k,z) % Akk) != 0) * Akk
      //        =  -5 - (-2 - 1)*2 = = 6 - 5 = 1
      // if `Akk = 1`, then
      // A(k,z) = A(k,z) - (A(k,z)/Akk) * Akk
      //        = A(k,z) - A(k,z) = 0
      // or if `Akz = -7, Akk = 39`, then in the loop below we get
      // A(k,z) = A(k,z) - ((A(k,z)/Akk) - ((A(k,z) % Akk) != 0) * Akk
      //        =  -7 - ((-7/39) - 1)*39 = = 6 - 5 = 1
      int64_t oAzr = Azr;
      Azr /= Akk;
      if (oAzr < 0) Azr -= (oAzr != (Azr * Akk));
      A[z, _] -= Azr * A[r, _];
    }
  }
}
constexpr void reduceSubDiagonalStack(MutPtrMatrix<int64_t> A,
                                      MutPtrMatrix<int64_t> B, ptrdiff_t c,
                                      ptrdiff_t r) {
  int64_t Akk = A[r, c];
  if (Akk < 0) {
    Akk = -Akk;
    A[r, _] *= -1;
  }
  for (ptrdiff_t z = 0; z < r; ++z) {
    if (int64_t Akz = A[z, c]) {
      int64_t oAkz = Akz;
      Akz /= Akk;
      if (oAkz < 0) Akz -= (oAkz != (Akz * Akk));
      A[z, _] -= Akz * A[r, _];
    }
  }
  for (ptrdiff_t z = 0; z < B.numRow(); ++z) {
    if (int64_t Bzr = B[z, c]) {
      int64_t oBzr = Bzr;
      Bzr /= Akk;
      if (oBzr < 0) Bzr -= (oBzr != (Bzr * Akk));
      B[z, _] -= Bzr * A[r, _];
    }
  }
}
constexpr void reduceSubDiagonal(std::array<MutPtrMatrix<int64_t>, 2> AB,
                                 Col<> c, Row<> r) {
  auto [A, B] = AB;
  int64_t Akk = A[r, c];
  if (Akk < 0) {
    Akk = -Akk;
    A[r, _] *= -1;
    B[r, _] *= -1;
  }
  for (ptrdiff_t z = 0; z < r; ++z) {
    // try to eliminate `A(k,z)`
    int64_t Akz = A[z, c];
    if (!Akz) continue;
    // if Akk == 1, then this zeros out Akz
    if (Akk != 1) {
      // we want positive but smaller subdiagonals
      // e.g., `Akz = 5, Akk = 2`, then in the loop below when `i=k`,
      // we set A(k,z) = A(k,z) - (A(k,z)/Akk) * Akk
      //        =   5 - 2*2 = 1
      // or if `Akz = -5, Akk = 2`, then in the loop below we get
      // A(k,z) = A(k,z) - ((A(k,z)/Akk) - ((A(k,z) % Akk) != 0) * Akk
      //        =  -5 - (-2 - 1)*2 = = 6 - 5 = 1
      // if `Akk = 1`, then
      // A(k,z) = A(k,z) - (A(k,z)/Akk) * Akk
      //        = A(k,z) - A(k,z) = 0
      // or if `Akz = -7, Akk = 39`, then in the loop below we get
      // A(k,z) = A(k,z) - ((A(k,z)/Akk) - ((A(k,z) % Akk) != 0) * Akk
      //        =  -7 - ((-7/39) - 1)*39 = = 6 - 5 = 1
      int64_t oAkz = Akz;
      Akz /= Akk;
      if (oAkz < 0) Akz -= (oAkz != (Akz * Akk));
    }
    A[z, _] -= Akz * A[r, _];
    B[z, _] -= Akz * B[r, _];
  }
}

constexpr void reduceColumn(MutPtrMatrix<int64_t> A, Col<> c, Row<> r) {
  zeroSupDiagonal(A, c, r);
  reduceSubDiagonal(A, c, r);
}
// treats A as stacked on top of B
constexpr void reduceColumnStack(MutPtrMatrix<int64_t> A,
                                 MutPtrMatrix<int64_t> B, ptrdiff_t c,
                                 ptrdiff_t r) {
  zeroSupDiagonal(B, Col<>{c}, Row<>{r});
  reduceSubDiagonalStack(B, A, c, r);
}

/// numNonZeroRows(PtrMatrix<int64_t> A) -> Row
/// Assumes some number of the trailing rows have been
/// zeroed out.  Returns the number of rows that are remaining.
constexpr auto numNonZeroRows(PtrMatrix<int64_t> A) -> Row<> {
  Row newM = A.numRow();
  while (newM && allZero(A[ptrdiff_t(newM) - 1, _])) --newM;
  return newM;
}
// NormalForm version assumes zero rows are sorted to end due to pivoting
constexpr void removeZeroRows(MutDensePtrMatrix<int64_t> &A) {
  A.truncate(numNonZeroRows(A));
}

// pass by value, returns number of rows to truncate
constexpr auto simplifySystemImpl(MutPtrMatrix<int64_t> A,
                                  ptrdiff_t colInit = 0) -> Row<> {
  auto [M, N] = shape(A);
  for (ptrdiff_t r = 0, c = colInit; c < N && r < M; ++c)
    if (!pivotRows(A, Col<>{c}, Row<>{M}, Row<>{r}))
      reduceColumn(A, Col<>{c}, Row<>{r++});
  return numNonZeroRows(A);
}
constexpr void simplifySystem(EmptyMatrix<int64_t>, ptrdiff_t = 0) {}
constexpr void simplifySystem(MutPtrMatrix<int64_t> &E, ptrdiff_t colInit = 0) {
  E.truncate(simplifySystemImpl(E, colInit));
}
// TODO: `const IntMatrix &` can be copied to `MutPtrMatrix<int64_t>`
// this happens via `const IntMatrix &` -> `const MutPtrMatrix<int64_t> &` ->
// `MutPtrMatrix<int64_t>`.
// Perhaps we should define `MutPtrMatrix(const MutPtrMatrix &) = delete;`?
//
// NOLINTNEXTLINE(performance-unnecessary-value-param)
constexpr auto rank(IntMatrix<> A) -> ptrdiff_t {
  return ptrdiff_t(simplifySystemImpl(A, 0));
}
constexpr void reduceColumn(std::array<MutPtrMatrix<int64_t>, 2> AB, Col<> c,
                            Row<> r) {
  zeroSupDiagonal(AB, c, r);
  reduceSubDiagonal(AB, c, r);
}
constexpr void simplifySystemsImpl(std::array<MutPtrMatrix<int64_t>, 2> AB) {
  auto [M, N] = shape(AB[0]);
  for (ptrdiff_t r = 0, c = 0; c < N && r < M; ++c)
    if (!pivotRowsPair(AB, Col<>{c}, Row<>{M}, Row<>{r}))
      reduceColumn(AB, Col<>{c}, Row<>{r++});
}
template <MatrixDimension S0, MatrixDimension S1>
constexpr void simplifySystem(MutArray<int64_t, S0> &A,
                              MutArray<int64_t, S1> &B) {
  simplifySystemsImpl({A, B});
  if (Row newM = numNonZeroRows(A); newM < A.numRow()) {
    A.truncate(newM);
    B.truncate(newM);
  }
}
[[nodiscard]] constexpr auto hermite(IntMatrix<> A)
  -> std::pair<IntMatrix<>, SquareMatrix<int64_t>> {
  SquareMatrix<int64_t> U{
    SquareMatrix<int64_t>::identity(ptrdiff_t(A.numRow()))};
  simplifySystemsImpl({A, U});
  return std::make_pair(std::move(A), std::move(U));
}

/// use A(j,k) to zero A(i,k)
constexpr auto zeroWithRowOp(MutPtrMatrix<int64_t> A, Row<> i, Row<> j, Col<> k,
                             int64_t f) -> int64_t {
  int64_t Aik = A[i, k];
  if (!Aik) return f;
  int64_t Ajk = A[j, k];
  invariant(Ajk != 0);
  int64_t g = gcd(Aik, Ajk);
  Aik /= g;
  Ajk /= g;
  int64_t ret = f * Ajk;
  constexpr ptrdiff_t W = simd::Width<int64_t>;
  simd::Vec<W, int64_t> vAjk = simd::vbroadcast<W, int64_t>(Ajk),
                        vAik = simd::vbroadcast<W, int64_t>(Aik), vg = {ret};
  PtrMatrix<int64_t> B = A;
  ptrdiff_t L = ptrdiff_t(A.numCol()), l = 0;
  if (ret != 1) {
    for (;;) {
      auto u{simd::index::unrollmask<1, W>(L, l)};
      if (!u) break;
      simd::Vec<W, int64_t> Ail = vAjk * B[i, u].vec - vAik * B[j, u].vec;
      A[i, u] = Ail;
      vg = gcd<W>(Ail, vg);
      l += W;
      if (!bool(simd::cmp::ne<W, int64_t>(vg, simd::Vec<W, int64_t>{} + 1)))
        break;
    }
  }
  if (l < L) {
    for (;; l += W) {
      auto u{simd::index::unrollmask<1, W>(L, l)};
      if (!u) break;
      A[i, u] = vAjk * B[i, u].vec - vAik * B[j, u].vec;
    }
  } else if (simd::cmp::le<W, int64_t>(vg, simd::Vec<W, int64_t>{} + 1)) {
    g = gcdreduce<W>(vg);
    if (g > 1) {
      for (ptrdiff_t ll = 0; ll < A.numCol(); ++ll)
        if (int64_t Ail = A[i, ll]) A[i, ll] = Ail / g;
      int64_t r = ret / g;
      invariant(r * g, ret);
      ret = r;
    }
  }
  return ret;
}
constexpr void zeroWithRowOperation(MutPtrMatrix<int64_t> A, Row<> i, Row<> j,
                                    Col<> k, Range<ptrdiff_t, ptrdiff_t> skip) {
  if (int64_t Aik = A[i, k]) {
    int64_t Ajk = A[j, k];
    int64_t g = gcd(Aik, Ajk);
    Aik /= g;
    Ajk /= g;
    g = 0;
    for (ptrdiff_t l = 0; l < skip.b; ++l) {
      int64_t Ail = Ajk * A[i, l] - Aik * A[j, l];
      A[i, l] = Ail;
      g = gcd(Ail, g);
    }
    for (ptrdiff_t l = skip.e; l < A.numCol(); ++l) {
      int64_t Ail = Ajk * A[i, l] - Aik * A[j, l];
      A[i, l] = Ail;
      g = gcd(Ail, g);
    }
    if (g > 1) {
      for (ptrdiff_t l = 0; l < skip.b; ++l)
        if (int64_t Ail = A[i, l]) A[i, l] = Ail / g;
      for (ptrdiff_t l = skip.e; l < A.numCol(); ++l)
        if (int64_t Ail = A[i, l]) A[i, l] = Ail / g;
    }
  }
}

// use row `r` to zero the remaining rows of column `c`
constexpr void zeroColumnPair(std::array<MutPtrMatrix<int64_t>, 2> AB, Col<> c,
                              Row<> r) {
  auto [A, B] = AB;
  const Row M = A.numRow();
  invariant(M, B.numRow());
  for (ptrdiff_t j = 0; j < r; ++j) {
    int64_t Arc = A[r, c], Ajc = A[j, c];
    if (!Ajc) continue;
    int64_t g = gcd(Arc, Ajc), x = Arc / g, y = Ajc / g;
    // auto [x, y] = divgcd(Arc, Ajc);
    // MutPtrVector<int64_t> Ar = A[r, _], Aj = A[j, _];
    // Aj << x * Aj - y * Ar;
    // MutPtrVector<int64_t> Br = B[r, _], Bj = B[j, _];
    // Bj << x * Bj - y * Br;
    for (ptrdiff_t i = 0; i < 2; ++i) {
      MutPtrVector<int64_t> Ar = AB[i][r, _], Aj = AB[i][j, _];
      Aj << x * Aj - y * Ar;
    }
  }
  // greater rows in previous columns have been zeroed out
  // therefore it is safe to use them for row operations with this row
  for (ptrdiff_t j = ptrdiff_t(r) + 1; j < M; ++j) {
    int64_t Arc = A[r, c], Ajc = A[j, c];
    if (!Ajc) continue;
    const auto [p, q, Arcr, Ajcr] = gcdxScale(Arc, Ajc);
    // MutPtrVector<int64_t> Ar = A[r, _], Aj = A[j, _];
    // tie(Ar, Aj) << Tuple(q * Aj + p * Ar, Arcr * Aj - Ajcr * Ar);
    // MutPtrVector<int64_t> Br = B[r, _], Bj = B[j, _];
    // tie(Br, Bj) << Tuple(q * Bj + p * Br, Arcr * Bj - Ajcr * Br);
    for (ptrdiff_t i = 0; i < 2; ++i) {
      MutPtrVector<int64_t> Ar = AB[i][r, _], Aj = AB[i][j, _];
      tie(Ar, Aj) << Tuple(q * Aj + p * Ar, Arcr * Aj - Ajcr * Ar);
    }
  }
}
// use row `r` to zero the remaining rows of column `c`
constexpr void zeroColumn(MutPtrMatrix<int64_t> A, Col<> c, Row<> r) {
  const Row M = A.numRow();
  for (ptrdiff_t j = 0; j < r; ++j) {
    int64_t Arc = A[r, c], Ajc = A[j, c];
    invariant(Arc != std::numeric_limits<int64_t>::min());
    invariant(Ajc != std::numeric_limits<int64_t>::min());
    if (!Ajc) continue;
    int64_t g = gcd(Arc, Ajc);
    A[j, _] << (Arc / g) * A[j, _] - (Ajc / g) * A[r, _];
  }
  // greater rows in previous columns have been zeroed out
  // therefore it is safe to use them for row operations with this row
  for (ptrdiff_t j = ptrdiff_t(r) + 1; j < M; ++j) {
    int64_t Arc = A[r, c], Ajc = A[j, c];
    if (!Ajc) continue;
    const auto [p, q, Arcr, Ajcr] = gcdxScale(Arc, Ajc);
    MutPtrVector<int64_t> Ar = A[r, _], Aj = A[j, _];
    tie(Ar, Aj) << Tuple(q * Aj + p * Ar, Arcr * Aj - Ajcr * Ar);
  }
}

constexpr auto pivotRowsBareiss(MutPtrMatrix<int64_t> A, ptrdiff_t i, Row<> M,
                                Row<> piv) -> Optional<ptrdiff_t> {
  Row j = piv;
  while (A[piv, i] == 0)
    if (++piv == M) return {};
  if (j != piv) swap(A, j, piv);
  return ptrdiff_t(piv);
}
constexpr void bareiss(MutPtrMatrix<int64_t> A,
                       MutPtrVector<ptrdiff_t> pivots) {
  const auto [M, N] = shape(A);
  invariant(ptrdiff_t(pivots.size()), std::min(M, N));
  int64_t prev = 1, pivInd = 0;
  for (ptrdiff_t r = 0, c = 0; c < N && r < M; ++c) {
    if (auto piv = pivotRowsBareiss(A, c, Row<>{M}, Row<>{r})) {
      pivots[pivInd++] = *piv;
      auto j{_(c + 1, N)};
      for (ptrdiff_t k = r + 1; k < M; ++k) {
        A[k, j] << (A[r, c] * A[k, j] - A[k, c] * A[r, j]) / prev;
        A[k, r] = 0;
      }
      prev = A[r++, c];
    }
  }
}

[[nodiscard]] constexpr auto bareiss(IntMatrix<> &A) -> Vector<ptrdiff_t> {
  Vector<ptrdiff_t> pivots(A.minRowCol());
  bareiss(A, pivots);
  return pivots;
}

// update a reduced matrix for a new row
// doesn't reduce last row (assumes you're solving for it)
constexpr auto updateForNewRow(MutPtrMatrix<int64_t> A) -> ptrdiff_t {
  // use existing rows to reduce
  ptrdiff_t M = ptrdiff_t(A.numRow()), N = ptrdiff_t(A.numCol()), MM = M - 1,
            NN = N - 1, n = 0, i, j = std::numeric_limits<ptrdiff_t>::max();
  for (ptrdiff_t m = 0; m < MM; ++m) {
#ifndef NDEBUG
    if (!allZero(A[m, _(0, n)])) __builtin_trap();
#endif
    while (A[m, n] == 0) {
      if ((j > NN) && (A[MM, n] != 0)) {
        i = m;
        j = n;
      }
      invariant((++n) < NN);
    }
    if (int64_t Aln = A[MM, n]) {
      // use this to reduce last row
      auto [x, y] = divgcd(Aln, A[m, n]);
      A[MM, _] << A[MM, _] * y - A[m, _] * x;
      invariant(A[MM, n] == 0);
    }
    ++n;
  }
  // we've reduced the new row, now to use it...
  // swap A(i,_(j,end)) with A(MM,_(j,end))
  if (j <= NN) { // we could do with a lot less copying...
    for (ptrdiff_t l = i; l < MM; ++l)
      for (ptrdiff_t k = j; k < N; ++k) std::swap(A[l, k], A[MM, k]);
  } else {
    // maybe there is a non-zero value
    j = n;
    for (; j < NN; ++j)
      if (A[MM, j] != 0) break;
    if (j == NN) return MM;
    i = MM;
  }
  // zero out A(_(0,i),j) using A(i,j)
  for (ptrdiff_t k = 0; k < i; ++k) {
    if (int64_t Akj = A[k, j]) {
      auto [x, y] = divgcd(Akj, A[i, j]);
      A[k, _] << A[k, _] * y - A[i, _] * x;
    }
  }
  return M;
}

/// void solveSystem(IntMatrix &A, IntMatrix &B)
/// Say we wanted to solve \f$\textbf{AX} = \textbf{B}\f$.
/// `solveSystem` left-multiplies both sides by
/// a matrix \f$\textbf{W}\f$ that diagonalizes \f$\textbf{A}\f$.
/// Once \f$\textbf{A}\f$ has been diagonalized, the solution is trivial.
/// Both inputs are overwritten with the product of the left multiplications.
constexpr void solveSystem(MutPtrMatrix<int64_t> A, MutPtrMatrix<int64_t> B) {
  const auto [M, N] = shape(A);
  for (ptrdiff_t r = 0, c = 0; c < N && r < M; ++c)
    if (!pivotRowsPair({A, B}, Col<>{c}, Row<>{M}, Row<>{r}))
      zeroColumnPair({A, B}, Col<>{c}, Row<>{r++});
}
// diagonalizes A(0:K,0:K)
constexpr void solveSystem(MutPtrMatrix<int64_t> A, ptrdiff_t K) {
  Row M = A.numRow();
  for (ptrdiff_t r = 0, c = 0; c < K && r < M; ++c)
    if (!pivotRows(A, Col<>{c}, Row<>{M}, Row<>{r}))
      zeroColumn(A, Col<>{c}, Row<>{r++});
}
// diagonalizes A(0:K,1:K+1)
constexpr void solveSystemSkip(MutPtrMatrix<int64_t> A) {
  const auto [M, N] = shape(A);
  for (ptrdiff_t r = 0, c = 1; c < N && r < M; ++c)
    if (!pivotRows(A, Col<>{c}, Row<>{M}, Row<>{r}))
      zeroColumn(A, Col<>{c}, Row<>{r++});
}

// returns `true` if the solve failed, `false` otherwise
// diagonals contain denominators.
// Assumes the last column is the vector to solve for.
constexpr void solveSystem(MutPtrMatrix<int64_t> A) {
  solveSystem(A, ptrdiff_t(A.numCol()) - 1);
}
/// inv(A) -> (D, B)
/// Given a matrix \f$\textbf{A}\f$, returns two matrices \f$\textbf{D}\f$ and
/// \f$\textbf{B}\f$ so that \f$\textbf{D}^{-1}\textbf{B} = \textbf{A}^{-1}\f$,
/// and \f$\textbf{D}\f$ is diagonal.
/// NOTE: This function assumes non-singular
// NOLINTNEXTLINE(performance-unnecessary-value-param)
[[nodiscard]] constexpr auto inv(SquareMatrix<int64_t> A)
  -> std::pair<SquareMatrix<int64_t>, SquareMatrix<int64_t>> {
  auto B = SquareMatrix<int64_t>::identity(A.numCol());
  solveSystem(A, B);
  return std::make_pair(std::move(A), std::move(B));
}
/// inv(A) -> (B, s)
/// Given a matrix \f$\textbf{A}\f$, returns a matrix \f$\textbf{B}\f$ and a
/// scalar \f$s\f$ such that \f$\frac{1}{s}\textbf{B} = \textbf{A}^{-1}\f$.
/// NOTE: This function assumes non-singular
/// D0 * B^{-1} = Binv0
/// (s/s) * D0 * B^{-1} = Binv0
/// s * B^{-1} = (s/D0) * Binv0
[[nodiscard]] constexpr auto scaledInv(SquareMatrix<int64_t> A)
  -> std::pair<SquareMatrix<int64_t>, int64_t> {
  auto B = SquareMatrix<int64_t>::identity(A.numCol());
  solveSystem(A, B);
  auto [s, nonUnity] = lcmNonUnity(A.diag());
  if (nonUnity)
    for (ptrdiff_t i = 0; i < A.numRow(); ++i) B[i, _] *= s / A[i, i];
  return {std::move(B), s};
}
// one row per null dim
constexpr void nullSpace11(DenseMatrix<int64_t> &B, DenseMatrix<int64_t> &A) {
  const Row M = A.numRow();
  B.resizeForOverwrite(math::SquareDims{M});
  B << 0;
  B.diag() << 1;
  solveSystem(A, B);
  Row R = numNonZeroRows(A);
  // slice B[_(R,end), :]
  if (!R) return;
  // we keep last D columns
  Row D = Row<>{ptrdiff_t(M) - ptrdiff_t(R)};
  // we keep `D` columns
  // TODO: shift pointer instead?
  // This seems like a bad idea given ManagedArrays that must
  // free their own ptrs; we'd have to still store either the old pointer or the
  // offset.
  // However, this may be reasonable given an implementation
  // that takes a `Arena<>` as input to allocate `B`, as
  // then we don't need to track the pointer.
  std::copy_n(B.data() + ptrdiff_t(R) * ptrdiff_t(M),
              ptrdiff_t(D) * ptrdiff_t(M), B.data());
  B.truncate(D);
}
[[nodiscard]] constexpr auto nullSpace(DenseMatrix<int64_t> A)
  -> DenseMatrix<int64_t> {
  DenseMatrix<int64_t> B;
  nullSpace11(B, A);
  return B;
}

} // namespace poly::math::NormalForm
