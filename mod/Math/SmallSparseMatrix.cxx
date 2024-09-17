#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifndef USE_MODULE
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <utility>

#include "Math/Array.cxx"
#include "Math/AxisTypes.cxx"
#include "Math/ManagedArray.cxx"
#include "Math/MatrixDimensions.cxx"
#else
export module SmallSparseMatrix;

import Array;
import AxisTypes;
import ManagedArray;
import MatDim;
import STL;
#endif

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif
// this file is not used at the moment
template <typename T> class SmallSparseMatrix {
  // non-zeros
  [[no_unique_address]] Vector<T> nonZeros{};
  // masks, the upper 8 bits give the number of elements in previous rows
  // the remaining 24 bits are a mask indicating non-zeros within this row
  static constexpr ptrdiff_t maxElemPerRow = 24;
  [[no_unique_address]] Vector<uint32_t> rows;
  [[no_unique_address]] Col<> ncol;

public:
  [[nodiscard]] constexpr auto getNonZeros() const -> PtrVector<T> {
    return nonZeros;
  }
  [[nodiscard]] constexpr auto getRows() const -> PtrVector<uint32_t> {
    return rows;
  }

  [[nodiscard]] constexpr auto numRow() const -> Row<> {
    return row(rows.size());
  }
  [[nodiscard]] constexpr auto numCol() const -> Col<> { return ncol; }
  [[nodiscard]] constexpr auto
  size() const -> CartesianIndex<ptrdiff_t, ptrdiff_t> {
    return {numRow(), numCol()};
  }
  [[nodiscard]] constexpr auto dim() const -> DenseDims<> {
    return {numRow(), numCol()};
  }
  // [[nodiscard]] constexpr auto view() const -> auto & { return *this; };
  constexpr SmallSparseMatrix(Row<> numRows, Col<> numCols)
    : rows(length(ptrdiff_t(numRows)), 0), ncol{numCols} {
    invariant(ptrdiff_t(ncol) <= maxElemPerRow);
  }
  constexpr auto get(Row<> i, Col<> j) const -> T {
    invariant(j < ncol);
    uint32_t r(rows[ptrdiff_t(i)]);
    uint32_t jshift = uint32_t(1) << uint32_t(ptrdiff_t(j));
    if (!(r & jshift)) return T{};
    // offset from previous rows
    uint32_t prevRowOffset = r >> maxElemPerRow;
    uint32_t rowOffset = std::popcount(r & (jshift - 1));
    return nonZeros[rowOffset + prevRowOffset];
  }
  constexpr auto operator[](ptrdiff_t i, ptrdiff_t j) const -> T {
    return get(row(i), col(j));
  }
  constexpr void insert(T x, Row<> i, Col<> j) {
    invariant(j < ncol);
    uint32_t r{rows[ptrdiff_t(i)]};
    uint32_t jshift = uint32_t(1) << ptrdiff_t(j);
    // offset from previous rows
    uint32_t prevRowOffset = r >> maxElemPerRow;
    uint32_t rowOffset = std::popcount(r & (jshift - 1));
    ptrdiff_t k = rowOffset + prevRowOffset;
    if (r & jshift) {
      nonZeros[k] = std::move(x);
    } else {
      nonZeros.insert(nonZeros.begin() + k, std::move(x));
      rows[ptrdiff_t(i)] = r | jshift;
      for (ptrdiff_t l = ptrdiff_t(i) + 1; l < rows.size(); ++l)
        rows[l] += uint32_t(1) << maxElemPerRow;
    }
  }

  struct Reference {
    [[no_unique_address]] SmallSparseMatrix<T> *A;
    [[no_unique_address]] ptrdiff_t i, j;
    constexpr operator T() const { return A->get(row(i), col(j)); }
    constexpr auto operator=(T x) -> Reference & {
      A->insert(std::move(x), row(i), col(j));
      return *this;
    }
  };
  constexpr auto operator[](ptrdiff_t i, ptrdiff_t j) -> Reference {
    return Reference{this, i, j};
  }
  template <std::convertible_to<T> Y, MatrixDimension S, ptrdiff_t L,
            typename A>
  operator ManagedArray<Y, S, L, A>() const {
    ManagedArray<Y, S, L, A> B(dim(), 0);
    ptrdiff_t k = 0;
    for (ptrdiff_t i = 0; i < numRow(); ++i) {
      uint32_t m = getRows()[i] & 0x00ffffff;
      ptrdiff_t j = 0;
      while (m) {
        uint32_t tz = std::countr_zero(m);
        m >>= tz + 1;
        j += tz;
        B[i, j++] = T(getNonZeros()[k++]);
      }
    }
    invariant(k == getNonZeros().size());
    return B;
  }

private:
  friend inline auto operator<<(std::ostream &os, const SmallSparseMatrix<T> &A)
    -> std::ostream & {
    ptrdiff_t k = 0;
    os << "[ ";
    for (ptrdiff_t i = 0; i < A.numRow(); ++i) {
      if (i) os << "  ";
      uint32_t m = A.rows[i] & 0x00ffffff;
      ptrdiff_t j = 0;
      while (m) {
        if (j) os << " ";
        uint32_t tz = std::countr_zero(m);
        m >>= (tz + 1);
        j += (tz + 1);
        while (tz--) os << " 0 ";
        const T &x = A.nonZeros[k++];
        if (x >= 0) os << " ";
        os << x;
      }
      for (; j < A.numCol(); ++j) os << "  0";
      os << "\n";
    }
    os << " ]";
    invariant(k == A.nonZeros.size());
    return os;
  }
  template <std::convertible_to<T> Y, MatrixDimension S>
  [[gnu::flatten]] friend constexpr auto
  operator<<(MutArray<Y, S> A, const SmallSparseMatrix &B) -> MutArray<Y, S> {
    ptrdiff_t M = ptrdiff_t(A.numRow()), N = ptrdiff_t(A.numCol()),
              X = ptrdiff_t(A.rowStride()), k = 0;
    invariant(M, ptrdiff_t(B.numRow()));
    invariant(N, ptrdiff_t(B.numCol()));
    T *mem = A.data();
    PtrVector<T> nz = B.getNonZeros();
    PtrVector<uint32_t> rws = B.getRows();
    for (ptrdiff_t i = 0; i < M; ++i) {
      uint32_t m = rws[i] & 0x00ffffff;
      ptrdiff_t j = 0, l = X * i;
      while (m) {
        uint32_t tz = std::countr_zero(m);
        m >>= tz + 1;
        for (; tz; --tz) mem[l + j++] = T{};
        mem[l + j++] = nz[k++];
      }
      for (; j < N; ++j) mem[l + j] = T{};
    }
    invariant(k == nz.size());
    return A;
  }
};

}; // namespace math
