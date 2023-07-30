#pragma once

#include "Math/AxisTypes.hpp"
#include "Utilities/Invariant.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace poly::math {
template <class R, class C> struct CartesianIndex {
  R row;
  C col;
  explicit constexpr operator Row() const { return row; }
  explicit constexpr operator Col() const { return col; }
  [[nodiscard]] constexpr auto operator==(const CartesianIndex &other) const
    -> bool {
    return (row == other.row) && (col == other.col);
  }
};
template <class R, class C> CartesianIndex(R r, C c) -> CartesianIndex<R, C>;

struct SquareDims;
struct DenseDims;
struct StridedDims {
  unsigned int M{};
  unsigned int N{};
  unsigned int strideM{};
  constexpr StridedDims() = default;
  constexpr StridedDims(Row m, Col n) : M(m), N(n), strideM(n) {}
  constexpr StridedDims(Row m, Col n, RowStride x) : M(m), N(n), strideM(x) {}
  constexpr StridedDims(CartesianIndex<Row, Col> ind)
    : M(unsigned(ind.row)), N(unsigned(ind.col)), strideM(unsigned(ind.col)) {}
  constexpr explicit operator int32_t() const { return int32_t(M * strideM); }
  constexpr explicit operator int64_t() const { return int64_t(M) * strideM; }
  constexpr explicit operator uint32_t() const { return uint32_t(M * strideM); }
  constexpr explicit operator uint64_t() const { return uint64_t(M) * strideM; }
  constexpr auto operator=(const DenseDims &D) -> StridedDims &;
  constexpr auto operator=(const SquareDims &D) -> StridedDims &;
  constexpr operator CarInd() const { return {M, N}; }
  constexpr explicit operator Row() const { return M; }
  constexpr explicit operator Col() const { return N; }
  constexpr explicit operator RowStride() const { return strideM; }
  [[nodiscard]] constexpr auto operator==(const StridedDims &D) const -> bool {
    return (M == D.M) && (N == D.N) && (strideM == D.strideM);
  }
  [[nodiscard]] constexpr auto truncate(Row r) const -> StridedDims {
    invariant(r <= Row{M});
    return {unsigned(r), N, strideM};
  }
  [[nodiscard]] constexpr auto truncate(Col c) const -> StridedDims {
    invariant(c <= Col{M});
    return {M, unsigned(c), strideM};
  }
  constexpr auto set(Row r) -> StridedDims & {
    M = unsigned(r);
    return *this;
  }
  constexpr auto set(Col c) -> StridedDims & {
    N = unsigned(c);
    strideM = std::max(strideM, N);
    return *this;
  }
  [[nodiscard]] constexpr auto similar(Row r) const -> StridedDims {
    return {unsigned(r), N, strideM};
  }
  [[nodiscard]] constexpr auto similar(Col c) const -> StridedDims {
    return {M, unsigned(c), strideM};
  }
  friend inline auto operator<<(std::ostream &os, StridedDims x)
    -> std::ostream & {
    return os << x.M << " x " << x.N << " (stride " << x.strideM << ")";
  }
};
/// Dimensions with a capacity
// struct CapDims : StridedDims {
//   unsigned int rowCapacity;
// };
struct DenseDims {
  unsigned int M{};
  unsigned int N{};
  constexpr explicit operator int32_t() const { return int32_t(M * N); }
  constexpr explicit operator int64_t() const { return int64_t(M) * N; }
  constexpr explicit operator uint32_t() const { return uint32_t(M * N); }
  constexpr explicit operator uint64_t() const { return uint64_t(M) * N; }
  constexpr DenseDims() = default;
  constexpr DenseDims(Row m, Col n) : M(unsigned(m)), N(unsigned(n)) {}
  constexpr explicit DenseDims(StridedDims d) : M(d.M), N(d.N) {}
  constexpr DenseDims(CartesianIndex<Row, Col> ind)
    : M(unsigned(ind.row)), N(unsigned(ind.col)) {}
  constexpr operator StridedDims() const { return {M, N, N}; }
  constexpr operator CarInd() const { return {M, N}; }
  constexpr auto operator=(const SquareDims &D) -> DenseDims &;
  constexpr explicit operator Row() const { return M; }
  constexpr explicit operator Col() const { return N; }
  constexpr explicit operator RowStride() const { return N; }
  [[nodiscard]] constexpr auto truncate(Row r) const -> DenseDims {
    invariant(r <= Row{M});
    return {unsigned(r), N};
  }
  [[nodiscard]] constexpr auto truncate(Col c) const -> StridedDims {
    invariant(c <= Col{M});
    return {M, c, N};
  }
  constexpr auto set(Row r) -> DenseDims & {
    M = unsigned(r);
    return *this;
  }
  constexpr auto set(Col c) -> DenseDims & {
    N = unsigned(c);
    return *this;
  }
  [[nodiscard]] constexpr auto similar(Row r) const -> DenseDims {
    return {unsigned(r), N};
  }
  [[nodiscard]] constexpr auto similar(Col c) const -> DenseDims {
    return {M, unsigned(c)};
  }
  friend inline auto operator<<(std::ostream &os, DenseDims x)
    -> std::ostream & {
    return os << x.M << " x " << x.N;
  }
};
struct SquareDims {
  unsigned int M{};
  constexpr explicit operator int32_t() const { return int32_t(M * M); }
  constexpr explicit operator int64_t() const { return int64_t(M) * M; }
  constexpr explicit operator uint32_t() const { return uint32_t(M * M); }
  constexpr explicit operator uint64_t() const { return uint64_t(M) * M; }
  constexpr SquareDims() = default;
  constexpr SquareDims(unsigned int d) : M{d} {}
  constexpr SquareDims(Row d) : M{unsigned(d)} {}
  constexpr SquareDims(Col d) : M{unsigned(d)} {}
  constexpr SquareDims(CartesianIndex<Row, Col> ind) : M(unsigned(ind.row)) {
    invariant(ptrdiff_t(ind.row), ptrdiff_t(ind.col));
  }
  constexpr operator StridedDims() const { return {M, M, M}; }
  constexpr operator DenseDims() const { return {M, M}; }
  constexpr operator CarInd() const { return {M, M}; }
  constexpr explicit operator Row() const { return M; }
  constexpr explicit operator Col() const { return M; }
  constexpr explicit operator RowStride() const { return M; }
  [[nodiscard]] constexpr auto truncate(Row r) const -> DenseDims {
    invariant(r <= Row{M});
    return {unsigned(r), M};
  }
  [[nodiscard]] constexpr auto truncate(Col c) const -> StridedDims {
    invariant(c <= Col{M});
    return {M, unsigned(c), M};
  }
  [[nodiscard]] constexpr auto similar(Row r) const -> DenseDims {
    return {unsigned(r), M};
  }
  [[nodiscard]] constexpr auto similar(Col c) const -> DenseDims {
    return {M, unsigned(c)};
  }
  friend inline auto operator<<(std::ostream &os, SquareDims x)
    -> std::ostream & {
    return os << x.M << " x " << x.M;
  }
};
// [[nodiscard]] constexpr auto capacity(std::integral auto c) { return c; }
// [[nodiscard]] constexpr auto capacity(auto c) -> unsigned int { return c; }
// [[nodiscard]] constexpr auto capacity(CapDims c) -> unsigned int {
//   return c.rowCapacity * c.strideM;
// }

constexpr auto StridedDims::operator=(const DenseDims &D) -> StridedDims & {
  M = D.M;
  N = D.N;
  strideM = N;
  return *this;
}
constexpr auto StridedDims::operator=(const SquareDims &D) -> StridedDims & {
  M = D.M;
  N = M;
  strideM = M;
  return *this;
}
constexpr auto DenseDims::operator=(const SquareDims &D) -> DenseDims & {
  M = D.M;
  N = M;
  return *this;
}
template <typename D>
concept MatrixDimension = requires(D d) {
  { d } -> std::convertible_to<StridedDims>;
};
static_assert(MatrixDimension<SquareDims>);
static_assert(MatrixDimension<DenseDims>);
static_assert(MatrixDimension<StridedDims>);

} // namespace poly::math
