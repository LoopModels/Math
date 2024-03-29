#pragma once

#include "Array.hpp"
#include "Math/Array.hpp"
#include "Math/ArrayOps.hpp"
#include <cstddef>
#include <cstdint>

namespace poly::math {
// this file is not used at the moment
template <typename T> class SmallSparseMatrix {
  // non-zeros
  [[no_unique_address]] Vector<T> nonZeros{};
  // masks, the upper 8 bits give the number of elements in previous rows
  // the remaining 24 bits are a mask indicating non-zeros within this row
  static constexpr ptrdiff_t maxElemPerRow = 24;
  [[no_unique_address]] Vector<uint32_t> rows;
  [[no_unique_address]] Col<> col;

public:
  [[nodiscard]] constexpr auto getNonZeros() const -> PtrVector<T> {
    return nonZeros;
  }
  [[nodiscard]] constexpr auto getRows() const -> PtrVector<uint32_t> {
    return rows;
  }

  [[nodiscard]] constexpr auto numRow() const -> Row<> {
    return Row<>{rows.size()};
  }
  [[nodiscard]] constexpr auto numCol() const -> Col<> { return col; }
  [[nodiscard]] constexpr auto size() const
    -> CartesianIndex<ptrdiff_t, ptrdiff_t> {
    return {numRow(), numCol()};
  }
  [[nodiscard]] constexpr auto dim() const -> DenseDims<> {
    return {numRow(), numCol()};
  }
  // [[nodiscard]] constexpr auto view() const -> auto & { return *this; };
  constexpr SmallSparseMatrix(Row<> numRows, Col<> numCols)
    : rows(ptrdiff_t(numRows), 0), col{numCols} {
    invariant(ptrdiff_t(col) <= maxElemPerRow);
  }
  constexpr auto get(Row<> i, Col<> j) const -> T {
    invariant(j < col);
    uint32_t r(rows[ptrdiff_t(i)]);
    uint32_t jshift = uint32_t(1) << uint32_t(ptrdiff_t(j));
    if (!(r & jshift)) return T{};
    // offset from previous rows
    uint32_t prevRowOffset = r >> maxElemPerRow;
    uint32_t rowOffset = std::popcount(r & (jshift - 1));
    return nonZeros[rowOffset + prevRowOffset];
  }
  constexpr auto operator[](ptrdiff_t i, ptrdiff_t j) const -> T {
    return get(Row<>{i}, Col<>{j});
  }
  constexpr void insert(T x, Row<> i, Col<> j) {
    invariant(j < col);
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
    constexpr operator T() const { return A->get(Row<>{i}, Col<>{j}); }
    constexpr auto operator=(T x) -> Reference & {
      A->insert(std::move(x), Row<>{i}, Col<>{j});
      return *this;
    }
  };
  constexpr auto operator[](ptrdiff_t i, ptrdiff_t j) -> Reference {
    return Reference{this, i, j};
  }
};

template <class T, class S, class P>
[[gnu::flatten]] constexpr auto
ArrayOps<T, S, P>::operator<<(const SmallSparseMatrix<T> &B) -> P & {
  static_assert(MatrixDimension<S>);
  ptrdiff_t M = ptrdiff_t(nr()), N = ptrdiff_t(nc()), k = 0;
  invariant(M, ptrdiff_t(B.numRow()));
  invariant(N, ptrdiff_t(B.numCol()));
  T *mem = data_();
  PtrVector<T> nz = B.getNonZeros();
  PtrVector<uint32_t> rws = B.getRows();
  for (ptrdiff_t i = 0; i < M; ++i) {
    uint32_t m = rws[i] & 0x00ffffff;
    ptrdiff_t j = 0, l = ptrdiff_t(rs()) * i;
    while (m) {
      uint32_t tz = std::countr_zero(m);
      m >>= tz + 1;
      for (; tz; --tz) mem[l + j++] = T{};
      mem[l + j++] = nz[k++];
    }
    for (; j < N; ++j) mem[l + j] = T{};
  }
  invariant(k == nz.size());
  return *static_cast<P *>(this);
}

}; // namespace poly::math
