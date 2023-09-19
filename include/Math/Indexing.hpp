#pragma once
#include "Math/AxisTypes.hpp"
#include "Math/Iterators.hpp"
#include "Math/MatrixDimensions.hpp"
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
// FIXME: we currently lose strong typing of Row and Col when using relative
// indexing; we should preserve it, perhaps within the OffsetBegin row/struct,
// making them templated?
template <typename T>
concept ScalarValueIndex =
  std::integral<T> || std::same_as<T, Row> || std::same_as<T, Col>;

constexpr auto operator+(ScalarValueIndex auto x, Begin) -> OffsetBegin {
  return OffsetBegin{ptrdiff_t(x)};
}
constexpr auto operator+(Begin, ScalarValueIndex auto x) -> OffsetBegin {
  return OffsetBegin{ptrdiff_t(x)};
}
constexpr auto operator+(ScalarValueIndex auto x, OffsetBegin y)
  -> OffsetBegin {
  return OffsetBegin{ptrdiff_t(x) + y.offset};
}
constexpr auto operator+(OffsetBegin y, ScalarValueIndex auto x)
  -> OffsetBegin {
  return OffsetBegin{ptrdiff_t(x) + y.offset};
}
[[maybe_unused]] static constexpr inline struct OffsetEnd {
  [[no_unique_address]] ptrdiff_t offset;
  friend inline auto operator<<(std::ostream &os, OffsetEnd r)
    -> std::ostream & {
    return os << "end - " << r.offset;
  }
} last{1};
constexpr auto operator-(End, ScalarValueIndex auto x) -> OffsetEnd {
  return OffsetEnd{ptrdiff_t(x)};
}
constexpr auto operator-(OffsetEnd y, ScalarValueIndex auto x) -> OffsetEnd {
  return OffsetEnd{y.offset + ptrdiff_t(x)};
}
constexpr auto operator+(OffsetEnd y, ScalarValueIndex auto x) -> OffsetEnd {
  return OffsetEnd{y.offset - ptrdiff_t(x)};
}
template <typename T>
concept RelativeOffset = std::same_as<T, End> || std::same_as<T, OffsetEnd> ||
                         std::same_as<T, Begin> || std::same_as<T, OffsetBegin>;

// Union type
template <typename T>
concept ScalarRelativeIndex =
  std::same_as<T, End> || std::same_as<T, Begin> ||
  std::same_as<T, OffsetBegin> || std::same_as<T, OffsetEnd>;

template <typename T>
concept ScalarIndex = std::integral<T> || ScalarRelativeIndex<T>;

[[maybe_unused]] static constexpr inline struct Colon {
  [[nodiscard]] inline constexpr auto operator()(auto B, auto E) const {
    return Range{standardizeRangeBound(B), standardizeRangeBound(E)};
  }
} _; // NOLINT(bugprone-reserved-identifier)
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

template <typename T>
concept ScalarRowIndex = ScalarIndex<T> || std::same_as<T, Row>;
template <typename T>
concept ScalarColIndex = ScalarIndex<T> || std::same_as<T, Col>;

static_assert(ScalarColIndex<OffsetEnd>);

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
[[nodiscard]] inline constexpr auto calcOffset(ptrdiff_t len, Col i)
  -> ptrdiff_t {
  invariant(*i < len);
  return *i;
}
[[nodiscard]] inline constexpr auto calcOffset(ptrdiff_t len, Row i)
  -> ptrdiff_t {
  invariant(*i < len);
  return *i;
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
[[nodiscard]] inline constexpr auto calcRangeOffset(ptrdiff_t len, Col i)
  -> ptrdiff_t {
  invariant(*i <= len);
  return *i;
}
[[nodiscard]] inline constexpr auto calcRangeOffset(ptrdiff_t len, Row i)
  -> ptrdiff_t {
  invariant(*i <= len);
  return *i;
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

constexpr auto calcOffset(SquareDims, ptrdiff_t i) -> ptrdiff_t { return i; }
constexpr auto calcOffset(DenseDims, ptrdiff_t i) -> ptrdiff_t { return i; }

template <class R, class C>
[[nodiscard]] inline constexpr auto calcOffset(StridedDims d,
                                               CartesianIndex<R, C> i)
  -> ptrdiff_t {
  ptrdiff_t r = ptrdiff_t(RowStride{d} * calcOffset(ptrdiff_t(Row{d}), i.row));
  ptrdiff_t c = calcOffset(ptrdiff_t(Col{d}), i.col);
  return r + c;
}

struct StridedRange {
  [[no_unique_address]] unsigned len;
  [[no_unique_address]] unsigned stride;
  explicit constexpr operator unsigned() const { return len; }
  explicit constexpr operator ptrdiff_t() const { return len; }
  friend inline auto operator<<(std::ostream &os, StridedRange x)
    -> std::ostream & {
    return os << "Length: " << x.len << " (stride: " << x.stride << ")";
  }
};
template <class I> constexpr auto calcOffset(StridedRange d, I i) -> ptrdiff_t {
  return d.stride * calcOffset(d.len, i);
}
constexpr auto is_integral_const(auto) -> bool { return false; }
template <typename T, T V>
constexpr auto is_integral_const(std::integral_constant<T, V>) -> bool {
  return true;
}

template <typename T>
concept StaticInt =
  std::is_same_v<T, std::integral_constant<typename T::value_type, T::value>>;

template <class T>
concept DenseLayout = std::integral<T> || std::is_same_v<T, DenseDims> ||
                      std::is_same_v<T, SquareDims> || StaticInt<T>;

static_assert(StaticInt<std::integral_constant<unsigned int, 3>>);
static_assert(!StaticInt<int64_t>);

template <class D>
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

struct Empty {};

constexpr auto calcNewDim(VectorDimension auto, ScalarIndex auto) -> Empty {
  return {};
}
constexpr auto calcNewDim(SquareDims, ptrdiff_t) -> Empty { return {}; }
constexpr auto calcNewDim(DenseDims, ptrdiff_t) -> Empty { return {}; }

constexpr auto calcNewDim(ptrdiff_t len, Range<ptrdiff_t, ptrdiff_t> r)
  -> unsigned {
  invariant(r.e <= len);
  invariant(r.b <= r.e);
  return unsigned(r.e - r.b);
}
template <class B, class E>
constexpr auto calcNewDim(ptrdiff_t len, Range<B, E> r) -> unsigned {
  return calcNewDim(len, canonicalizeRange(r, len));
}
constexpr auto calcNewDim(StridedRange len, Range<ptrdiff_t, ptrdiff_t> r)
  -> StridedRange {
  return StridedRange{unsigned(calcNewDim(len.len, r)), len.stride};
}
template <class B, class E>
constexpr auto calcNewDim(StridedRange len, Range<B, E> r) -> StridedRange {
  return StridedRange{unsigned(calcNewDim(len.len, r)), len.stride};
}
template <ScalarRowIndex R, ScalarColIndex C>
constexpr auto calcNewDim(StridedDims, CartesianIndex<R, C>) -> Empty {
  return {};
}
constexpr auto calcNewDim(std::integral auto len, Colon) -> unsigned {
  return unsigned(len);
};
constexpr auto calcNewDim(StaticInt auto len, Colon) { return len; };
constexpr auto calcNewDim(StridedRange len, Colon) { return len; };

template <AbstractSlice B, ScalarColIndex C>
constexpr auto calcNewDim(StridedDims d, CartesianIndex<B, C> i) {
  unsigned rowDims = unsigned(calcNewDim(ptrdiff_t(Row{d}), i.row));
  return StridedRange{rowDims, unsigned(RowStride{d})};
}

template <ScalarRowIndex R, AbstractSlice C>
constexpr auto calcNewDim(StridedDims d, CartesianIndex<R, C> i) {
  return calcNewDim(ptrdiff_t(Col{d}), i.col);
}

template <AbstractSlice B, AbstractSlice C>
constexpr auto calcNewDim(StridedDims d, CartesianIndex<B, C> i) {
  auto rowDims = calcNewDim(ptrdiff_t(Row{d}), i.row);
  auto colDims = calcNewDim(ptrdiff_t(Col{d}), i.col);
  return StridedDims{Row{rowDims}, Col{colDims}, RowStride{d}};
}
template <AbstractSlice B>
constexpr auto calcNewDim(DenseDims d, CartesianIndex<B, Colon> i) {
  auto rowDims = calcNewDim(ptrdiff_t(Row{d}), i.row);
  return DenseDims{Row{rowDims}, Col{d}};
}
template <AbstractSlice B>
constexpr auto calcNewDim(SquareDims d, CartesianIndex<B, Colon> i) {
  auto rowDims = calcNewDim(ptrdiff_t(Row{d}), i.row);
  return DenseDims{Row{rowDims}, Col{d}};
}

} // namespace poly::math
