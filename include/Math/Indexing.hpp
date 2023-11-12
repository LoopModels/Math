#pragma once
#include "Math/AxisTypes.hpp"
#include "Math/Iterators.hpp"
#include "Math/MatrixDimensions.hpp"
#include "Math/SIMD.hpp"
#include <cstddef>

namespace poly::math {

[[maybe_unused]] static inline constexpr struct Begin {
  friend inline auto operator<<(std::ostream &os, Begin) -> std::ostream & {
    return os << 0;
  }
} begin;
[[maybe_unused]] static inline constexpr struct End {
  friend inline auto operator<<(std::ostream &os, End) -> std::ostream & {
    return os << "end";
  }
} end;
struct OffsetBegin {
  [[no_unique_address]] ptrdiff_t offset;
  friend inline auto operator<<(std::ostream &os, OffsetBegin r)
    -> std::ostream & {
    return os << r.offset;
  }
};

constexpr auto operator+(ptrdiff_t x, Begin) -> OffsetBegin {
  return OffsetBegin{x};
}
constexpr auto operator+(Begin, ptrdiff_t x) -> OffsetBegin {
  return OffsetBegin{x};
}
constexpr auto operator+(ptrdiff_t x, OffsetBegin y) -> OffsetBegin {
  return OffsetBegin{x + y.offset};
}
constexpr auto operator+(OffsetBegin y, ptrdiff_t x) -> OffsetBegin {
  return OffsetBegin{ptrdiff_t(x) + y.offset};
}
[[maybe_unused]] static constexpr inline struct OffsetEnd {
  [[no_unique_address]] ptrdiff_t offset;
  friend inline auto operator<<(std::ostream &os, OffsetEnd r)
    -> std::ostream & {
    return os << "end - " << r.offset;
  }
} last{1};
constexpr auto operator-(End, ptrdiff_t x) -> OffsetEnd { return OffsetEnd{x}; }
constexpr auto operator-(OffsetEnd y, ptrdiff_t x) -> OffsetEnd {
  return OffsetEnd{y.offset + x};
}
constexpr auto operator+(OffsetEnd y, ptrdiff_t x) -> OffsetEnd {
  return OffsetEnd{y.offset - x};
}

// Union type
template <typename T>
concept ScalarRelativeIndex =
  std::same_as<T, End> || std::same_as<T, Begin> ||
  std::same_as<T, OffsetBegin> || std::same_as<T, OffsetEnd>;

namespace simd {
template <class T> struct IsSimdScalarIndex : std::false_type {};
template <ptrdiff_t N, typename I>
struct IsSimdScalarIndex<Unroll<N, I>> : std::true_type {};
template <ptrdiff_t N, typename U>
struct IsSimdScalarIndex<VectorIndex<N, U>> : std::true_type {};
template <typename T>
concept SIMDIndex = simd::IsSimdScalarIndex<T>::value;
} // namespace simd

template <typename T>
concept ScalarIndex =
  std::convertible_to<T, ptrdiff_t> || ScalarRelativeIndex<T> || SIMDIndex<T>;

[[maybe_unused]] static constexpr inline struct Colon {
  [[nodiscard]] inline constexpr auto operator()(auto B, auto E) const {
    return Range{standardizeRangeBound(B), standardizeRangeBound(E)};
  }
} _;

constexpr auto canonicalize(ptrdiff_t e, ptrdiff_t) -> ptrdiff_t { return e; }
constexpr auto canonicalize(Begin, ptrdiff_t) -> ptrdiff_t { return 0; }
constexpr auto canonicalize(OffsetBegin b, ptrdiff_t) -> ptrdiff_t {
  return b.offset;
}
constexpr auto canonicalize(End, ptrdiff_t M) -> ptrdiff_t { return M; }
constexpr auto canonicalize(OffsetEnd e, ptrdiff_t M) -> ptrdiff_t {
  return M - e.offset;
}
template <typename B, typename E>
constexpr auto canonicalizeRange(Range<B, E> r, ptrdiff_t M)
  -> Range<ptrdiff_t, ptrdiff_t> {
  return Range<ptrdiff_t, ptrdiff_t>{canonicalize(r.b, M),
                                     canonicalize(r.e, M)};
}
constexpr auto canonicalizeRange(Colon, ptrdiff_t M)
  -> Range<ptrdiff_t, ptrdiff_t> {
  return Range<ptrdiff_t, ptrdiff_t>{0, M};
}

static_assert(ScalarIndex<OffsetEnd>);

template <typename T>
concept AbstractSlice = requires(T t, ptrdiff_t M) {
  { canonicalizeRange(t, M) } -> std::same_as<Range<ptrdiff_t, ptrdiff_t>>;
};
static_assert(AbstractSlice<Range<ptrdiff_t, ptrdiff_t>>);
static_assert(AbstractSlice<Colon>);

[[nodiscard]] inline constexpr auto calcOffset(ptrdiff_t len, ptrdiff_t i)
  -> ptrdiff_t {
  invariant(i < len);
  return i;
}
[[nodiscard]] inline constexpr auto calcOffset(ptrdiff_t, Begin) -> ptrdiff_t {
  return 0;
}
[[nodiscard]] inline constexpr auto calcOffset(ptrdiff_t len, OffsetBegin i)
  -> ptrdiff_t {
  return calcOffset(len, i.offset);
}
[[nodiscard]] inline constexpr auto calcOffset(ptrdiff_t len, OffsetEnd i)
  -> ptrdiff_t {
  invariant(i.offset <= len);
  return len - i.offset;
}
[[nodiscard]] inline constexpr auto calcRangeOffset(ptrdiff_t len, ptrdiff_t i)
  -> ptrdiff_t {
  invariant(i <= len);
  return i;
}
[[nodiscard]] inline constexpr auto calcRangeOffset(ptrdiff_t, Begin)
  -> ptrdiff_t {
  return 0;
}
[[nodiscard]] inline constexpr auto calcRangeOffset(ptrdiff_t len,
                                                    OffsetBegin i)
  -> ptrdiff_t {
  return calcRangeOffset(len, i.offset);
}
[[nodiscard]] inline constexpr auto calcRangeOffset(ptrdiff_t len, OffsetEnd i)
  -> ptrdiff_t {
  invariant(i.offset <= len);
  return len - i.offset;
}
// note that we don't check i.b < len because we want to allow
// empty ranges, and r.b <= r.e <= len is checked in calcNewDim.
template <class B, class E>
constexpr auto calcOffset(ptrdiff_t len, Range<B, E> i) -> ptrdiff_t {
  return calcRangeOffset(len, i.b);
}
constexpr auto calcOffset(ptrdiff_t, Colon) -> ptrdiff_t { return 0; }

constexpr auto calcOffset(SquareDims<>, ptrdiff_t i) -> ptrdiff_t { return i; }
constexpr auto calcOffset(DenseDims<>, ptrdiff_t i) -> ptrdiff_t { return i; }

template <class R, class C>
[[nodiscard]] inline constexpr auto calcOffset(StridedDims<> d,
                                               CartesianIndex<R, C> i)
  -> ptrdiff_t {
  ptrdiff_t r =
    ptrdiff_t(RowStride<>(d)) * calcOffset(ptrdiff_t(Row<>(d)), i.row);
  ptrdiff_t c = calcOffset(ptrdiff_t(Col<>(d)), i.col);
  return r + c;
}

struct StridedRange {
  [[no_unique_address]] ptrdiff_t len;
  [[no_unique_address]] ptrdiff_t stride;
  explicit constexpr operator ptrdiff_t() const { return len; }
  friend inline auto operator<<(std::ostream &os, StridedRange x)
    -> std::ostream & {
    return os << "Length: " << x.len << " (stride: " << x.stride << ")";
  }
};
template <class I> constexpr auto calcOffset(StridedRange d, I i) -> ptrdiff_t {
  return d.stride * calcOffset(d.len, i);
}
// constexpr auto is_integral_const(auto) -> bool { return false; }
// template <typename T, T V>
// constexpr auto is_integral_const(std::integral_constant<T, V>) -> bool {
//   return true;
// }
constexpr auto row(StridedRange r) -> Row<> { return {r.len}; }
constexpr auto col(StridedRange) -> Col<1> { return {}; }
constexpr auto stride(StridedRange r) -> RowStride<> { return {r.stride}; }

template <typename T>
concept StaticInt =
  std::is_same_v<T, std::integral_constant<typename T::value_type, T::value>>;

template <typename T>
concept DenseLayout =
  std::integral<T> || std::is_convertible_v<T, DenseDims<>> || StaticInt<T>;

static_assert(StaticInt<std::integral_constant<ptrdiff_t, 3>>);
static_assert(!StaticInt<int64_t>);

template <typename D>
concept VectorDimension =
  std::integral<D> || std::same_as<D, StridedRange> || StaticInt<D>;

// Concept for aligning array dimensions with indices.
template <class I, class D>
concept Index =
  (VectorDimension<D> && (ScalarIndex<I> || AbstractSlice<I>)) ||
  (DenseLayout<D> && ScalarIndex<I>) || (MatrixDimension<D> && requires(I i) {
    { i.row };
    { i.col };
  });
static_assert(Index<CartesianIndex<ptrdiff_t, ptrdiff_t>, DenseDims<>>);
struct Empty {};

constexpr auto calcNewDim(VectorDimension auto, ScalarIndex auto) -> Empty {
  return {};
}
constexpr auto calcNewDim(SquareDims<>, ptrdiff_t) -> Empty { return {}; }
constexpr auto calcNewDim(DenseDims<>, ptrdiff_t) -> Empty { return {}; }

constexpr auto calcNewDim(ptrdiff_t len, Range<ptrdiff_t, ptrdiff_t> r)
  -> ptrdiff_t {
  invariant(r.e <= len);
  invariant(r.b <= r.e);
  return ptrdiff_t(r.e - r.b);
}
template <class B, class E>
constexpr auto calcNewDim(ptrdiff_t len, Range<B, E> r) -> ptrdiff_t {
  return calcNewDim(len, canonicalizeRange(r, len));
}
constexpr auto calcNewDim(StridedRange len, Range<ptrdiff_t, ptrdiff_t> r)
  -> StridedRange {
  return StridedRange{ptrdiff_t(calcNewDim(len.len, r)), len.stride};
}
template <class B, class E>
constexpr auto calcNewDim(StridedRange len, Range<B, E> r) -> StridedRange {
  return StridedRange{ptrdiff_t(calcNewDim(len.len, r)), len.stride};
}
template <ScalarIndex R, ScalarIndex C>
constexpr auto calcNewDim(StridedDims<>, CartesianIndex<R, C>) -> Empty {
  return {};
}
constexpr auto calcNewDim(std::integral auto len, Colon) -> ptrdiff_t {
  return ptrdiff_t(len);
};
constexpr auto calcNewDim(StaticInt auto len, Colon) { return len; };
constexpr auto calcNewDim(StridedRange len, Colon) { return len; };

template <AbstractSlice B, ScalarIndex C>
constexpr auto calcNewDim(StridedDims<> d, CartesianIndex<B, C> i) {
  ptrdiff_t rowDims = ptrdiff_t(calcNewDim(ptrdiff_t(Row(d)), i.row));
  return StridedRange{rowDims, ptrdiff_t(RowStride(d))};
}

template <ScalarIndex R, AbstractSlice C>
constexpr auto calcNewDim(StridedDims<> d, CartesianIndex<R, C> i) {
  return calcNewDim(ptrdiff_t(Col(d)), i.col);
}

template <AbstractSlice B, AbstractSlice C>
constexpr auto calcNewDim(StridedDims<> d, CartesianIndex<B, C> i) {
  auto rowDims = calcNewDim(ptrdiff_t(Row(d)), i.row);
  auto colDims = calcNewDim(ptrdiff_t(Col(d)), i.col);
  return StridedDims(row(rowDims), col(colDims), RowStride(d));
}
template <AbstractSlice B>
constexpr auto calcNewDim(DenseDims<> d, CartesianIndex<B, Colon> i) {
  auto rowDims = calcNewDim(ptrdiff_t(Row(d)), i.row);
  return DenseDims(row(rowDims), Col(d));
}
template <AbstractSlice B>
constexpr auto calcNewDim(SquareDims<> d, CartesianIndex<B, Colon> i) {
  auto rowDims = calcNewDim(ptrdiff_t(Row(d)), i.row);
  return DenseDims(row(rowDims), Col(d));
}

} // namespace poly::math
