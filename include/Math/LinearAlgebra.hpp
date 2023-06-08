#pragma once
#include "Math/Array.hpp"
#include "Math/Constructors.hpp"
#include "Math/Math.hpp"
#include "Math/MatrixDimensions.hpp"
#include "Rational.hpp"
#include "Utilities/Invariant.hpp"
#include <concepts>
namespace poly::math::LU {
[[nodiscard]] constexpr auto ldivrat(SquarePtrMatrix<Rational> F,
                                     PtrVector<unsigned> ipiv,
                                     MutPtrMatrix<Rational> rhs) -> bool {
  auto [M, N] = rhs.size();
  invariant(size_t(F.numRow()), size_t(M));
  // permute rhs
  for (size_t i = 0; i < M; ++i) {
    unsigned ip = ipiv[i];
    if (i != ip)
      for (size_t j = 0; j < M; ++j) std::swap(rhs(ip, j), rhs(i, j));
  }
  // LU x = rhs
  // L y = rhs // L is UnitLowerTriangular
  for (size_t n = 0; n < N; ++n) {
    for (size_t m = 0; m < M; ++m) {
      Rational Ymn = rhs(m, n);
      for (size_t k = 0; k < m; ++k)
        if (Ymn.fnmadd(F(m, k), rhs(k, n))) return true;
      rhs(m, n) = Ymn;
    }
  }
  // U x = y
  for (size_t n = 0; n < N; ++n) {
    for (auto m = size_t(M); m--;) {
      Rational Ymn = rhs(m, n);
      for (size_t k = m + 1; k < M; ++k)
        if (Ymn.fnmadd(F(m, k), rhs(k, n))) return true;
      if (auto div = Ymn.safeDiv(F(m, m))) rhs(m, n) = *div;
      else return true;
    }
  }
  return false;
}
template <class S>
constexpr void ldiv(SquarePtrMatrix<S> F, PtrVector<unsigned> ipiv,
                    MutPtrMatrix<S> rhs) {
  auto [M, N] = rhs.size();
  invariant(size_t(F.numRow()), size_t(M));
  // permute rhs
  for (size_t i = 0; i < M; ++i) {
    unsigned ip = ipiv[i];
    if (i != ip)
      for (size_t j = 0; j < M; ++j) std::swap(rhs(ip, j), rhs(i, j));
  }
  // LU x = rhs
  // L y = rhs // L is UnitLowerTriangular
  for (size_t n = 0; n < N; ++n) {
    for (size_t m = 0; m < M; ++m) {
      S Ymn = rhs(m, n);
      for (size_t k = 0; k < m; ++k) Ymn -= F(m, k) * rhs(k, n);
      rhs(m, n) = Ymn;
    }
  }
  // U x = y
  for (size_t n = 0; n < N; ++n) {
    for (auto m = size_t(M); m--;) {
      S Ymn = rhs(m, n);
      for (size_t k = m + 1; k < M; ++k) Ymn -= F(m, k) * rhs(k, n);
      rhs(m, n) = Ymn / F(m, m);
    }
  }
}

[[nodiscard]] constexpr auto rdivrat(SquarePtrMatrix<Rational> F,
                                     PtrVector<unsigned> ipiv,
                                     MutPtrMatrix<Rational> rhs) -> bool {
  auto [M, N] = rhs.size();
  invariant(size_t(F.numCol()), size_t(N));
  // PA = LU
  // x LU = rhs
  // y U = rhs
  for (size_t n = 0; n < N; ++n) {
    for (size_t m = 0; m < M; ++m) {

      Rational Ymn = rhs(m, n);
      for (size_t k = 0; k < n; ++k)
        if (Ymn.fnmadd(rhs(m, k), F(k, n))) return true;
      if (auto div = Ymn.safeDiv(F(n, n))) rhs(m, n) = *div;
      else return true;
    }
  }
  // x L = y
  for (auto n = size_t(N); n--;) {
    // for (size_t n = 0; n < N; ++n) {
    for (size_t m = 0; m < M; ++m) {
      Rational Xmn = rhs(m, n);
      for (size_t k = n + 1; k < N; ++k)
        if (Xmn.fnmadd(rhs(m, k), F(k, n))) return true;
      rhs(m, n) = Xmn;
    }
  }
  // permute rhs
  for (auto j = size_t(N); j--;) {
    unsigned jp = ipiv[j];
    if (j != jp)
      for (size_t i = 0; i < M; ++i) std::swap(rhs(i, jp), rhs(i, j));
  }

  return false;
}
template <class S>
constexpr void rdiv(SquarePtrMatrix<S> F, PtrVector<unsigned> ipiv,
                    MutPtrMatrix<S> rhs) {
  auto [M, N] = rhs.size();
  invariant(size_t(F.numCol()), size_t(N));
  // PA = LU
  // x LU = rhs
  // y U = rhs
  for (size_t n = 0; n < N; ++n) {
    for (size_t m = 0; m < M; ++m) {
      S Ymn = rhs(m, n);
      for (size_t k = 0; k < n; ++k) Ymn -= rhs(m, k) * F(k, n);
      rhs(m, n) = Ymn / F(n, n);
    }
  }
  // x L = y
  for (auto n = size_t(N); n--;) {
    // for (size_t n = 0; n < N; ++n) {
    for (size_t m = 0; m < M; ++m) {
      S Xmn = rhs(m, n);
      for (size_t k = n + 1; k < N; ++k) Xmn -= rhs(m, k) * F(k, n);
      rhs(m, n) = Xmn;
    }
  }
  // permute rhs
  for (auto j = size_t(N); j--;) {
    unsigned jp = ipiv[j];
    if (j != jp)
      for (size_t i = 0; i < M; ++i) std::swap(rhs(i, jp), rhs(i, j));
  }
}

template <class T, size_t L> class Fact {
  SquareMatrix<T, L> F;
  Vector<unsigned> ipiv;

public:
  constexpr void ldiv(MutPtrMatrix<T> rhs) const { LU::ldiv(F, ipiv, rhs); }
  constexpr void rdiv(MutPtrMatrix<T> rhs) const { LU::rdiv(F, ipiv, rhs); }
  constexpr auto ldivrat(MutPtrMatrix<T> rhs) const -> bool {
    return LU::ldivrat(F, ipiv, rhs);
  }
  constexpr auto rdivrat(MutPtrMatrix<T> rhs) const -> bool {
    return LU::rdivrat(F, ipiv, rhs);
  }
  constexpr Fact(SquareMatrix<T, L> f, Vector<unsigned> ip)
    : F(std::move(f)), ipiv(std::move(ip)) {
    invariant(size_t(F.numRow()), size_t(ipiv.size()));
  }

  [[nodiscard]] constexpr auto inv() const
    -> std::optional<SquareMatrix<Rational, L>> {
    SquareMatrix<Rational, L> A{
      SquareMatrix<Rational, L>::identity(size_t(F.numCol()))};
    if (!ldivrat(A)) return A;
    return {};
  }
  [[nodiscard]] constexpr auto det() const -> std::optional<Rational> {
    Rational d = F(0, 0);
    for (size_t i = 1; i < F.numCol(); ++i)
      if (auto di = d.safeMul(F(i, i))) d = *di;
      else return {};
    return d;
  }
  [[nodiscard]] constexpr auto perm() const -> Vector<unsigned> {
    Col M = F.numCol();
    Vector<unsigned> perm;
    for (size_t m = 0; m < M; ++m) perm.push_back(m);
    for (size_t m = 0; m < M; ++m) std::swap(perm[m], perm[ipiv[m]]);
    return perm;
  }
  friend auto operator<<(std::ostream &os, const Fact &lu) -> std::ostream & {
    return os << "LU fact:\n" << lu.F << "\nperm = \n" << lu.ipiv << '\n';
  }
};
template <size_t L>
[[nodiscard]] constexpr auto fact(const SquareMatrix<int64_t, L> &B)
  -> std::optional<Fact<Rational, L>> {
  Row M = B.numRow();
  SquareMatrix<Rational, L> A(B);
  // auto ipiv = Vector<unsigned>{.s = unsigned(M)};
  auto ipiv{vector(std::allocator<unsigned>{}, unsigned(M))};
  // Vector<unsigned> ipiv{.s = unsigned(M)};
  invariant(size_t(ipiv.size()), size_t(M));
  for (size_t i = 0; i < M; ++i) ipiv[i] = i;
  for (size_t k = 0; k < M; ++k) {
    size_t kp = k;
    for (; kp < M; ++kp) {
      if (A(kp, k) == 0) continue;
      ipiv[k] = kp;
      break;
    }
    if (kp != k)
      for (size_t j = 0; j < M; ++j) std::swap(A(kp, j), A(k, j));
    Rational invAkk = A(k, k).inv();
    for (size_t i = k + 1; i < M; ++i)
      if (std::optional<Rational> Aik = A(i, k).safeMul(invAkk)) A(i, k) = *Aik;
      else return {};
    for (size_t i = k + 1; i < M; ++i) {
      for (size_t j = k + 1; j < M; ++j) {
        if (std::optional<Rational> kAij = A(i, k).safeMul(A(k, j))) {
          if (std::optional<Rational> Aij = A(i, j).safeSub(*kAij)) {
            A(i, j) = *Aij;
            continue;
          }
        }
        return {};
      }
    }
  }
  return Fact<Rational, L>{std::move(A), std::move(ipiv)};
}
template <typename S> constexpr auto factImpl(MutSquarePtrMatrix<S> A) {
  Row M = A.numRow();
  auto ipiv{vector(std::allocator<unsigned>{}, unsigned(M))};
  invariant(size_t(ipiv.size()), size_t(M));
  for (size_t i = 0; i < M; ++i) ipiv[i] = i;
  for (size_t k = 0; k < M; ++k) {
    size_t kp = k;
    for (; kp < M; ++kp) {
      if (A(kp, k) == 0) continue;
      ipiv[k] = kp;
      break;
    }
    if (kp != k)
      for (size_t j = 0; j < M; ++j) std::swap(A(kp, j), A(k, j));
    S invAkk = 1.0 / A(k, k);
    for (size_t i = k + 1; i < M; ++i) A(i, k) = A(i, k) * invAkk;
    for (size_t i = k + 1; i < M; ++i)
      for (size_t j = k + 1; j < M; ++j) A(i, j) = A(i, j) - A(i, k) * A(k, j);
  }
  return ipiv;
}
template <class S, size_t L>
[[nodiscard]] constexpr auto fact(SquareMatrix<S, L> A) -> Fact<S, L> {
  auto &&ipiv{factImpl(A)};
  return Fact<S, L>{std::move(A), std::move(ipiv)};
}
/// ldiv(A, B)
/// computes A \ B, modifying A and B
/// Note that this computes an LU factorization;
/// if you are performing more than one division,
/// it would be more efficient to precompute an
/// `auto F = LU::fact(A)`, and use this for multiple
/// `F.ldiv(B)` calls.
template <typename T>
constexpr void ldiv(MutSquarePtrMatrix<T> A, MutPtrMatrix<T> B) {
  auto ipiv{factImpl(A)};
  ldiv(A, ipiv, B);
}
/// rdiv(A, B)
/// Computes B / A, modifying A and B
/// Note that this computes an LU factorization;
/// if you are performing more than one division,
/// it would be more efficient to precompute an
/// `auto F = LU::fact(A)`, and use this for multiple
/// `F.rdiv(B)` calls.
template <typename T>
constexpr void rdiv(MutSquarePtrMatrix<T> A, MutPtrMatrix<T> B) {
  auto ipiv{factImpl(A)};
  rdiv(A, ipiv, B);
}

} // namespace poly::math::LU
