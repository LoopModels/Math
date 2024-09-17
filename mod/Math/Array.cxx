#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#include "Owner.hxx"
#ifndef USE_MODULE
#include "Alloc/Arena.cxx"
#include "Containers/Pair.cxx"
#include "Containers/Storage.cxx"
#include "Math/ArrayConcepts.cxx"
#include "Math/ArrayOps.cxx"
#include "Math/AxisTypes.cxx"
#include "Math/ExpressionTemplates.cxx"
#include "Math/Indexing.cxx"
#include "Math/MatrixDimensions.cxx"
#include "Math/Ranges.cxx"
#include "Math/ScalarizeViaCastArrayOps.cxx"
#include "Utilities/ArrayPrint.cxx"
#include "Utilities/Invariant.cxx"
#include "Utilities/Optional.cxx"
#include "Utilities/Parameters.cxx"
#include "Utilities/Reference.cxx"
#include "Utilities/TypeCompression.cxx"
#include "Utilities/Valid.cxx"
#include <algorithm>
#include <array>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <memory>
#include <ranges>
#include <type_traits>
#include <utility>
#include <version>
#else
export module Array;

export import ArrayConcepts;
export import AssignExprTemplates;
export import AxisTypes;
export import Indexing;
export import Range;
import Allocator;
import Arena;
import ArrayPrint;
import CompressReference;
import ExprTemplates;
import Invariant;
import Optional;
import Pair;
import Param;
import Rational;
import ScalarizeViaCast;
import SIMD;
import STL;
import Storage;
import TypeCompression;
import Valid;
#endif

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif
using utils::compressed_t, utils::decompressed_t;

static_assert(Dimension<Length<>>);
static_assert(Dimension<DenseDims<3, 2>>);
static_assert(
  std::same_as<Col<1>, decltype(Col(std::declval<DenseDims<3, 1>>()))>);
static_assert(Dimension<DenseDims<3, 1>>);
static_assert(VectorDimension<StridedRange<>>);
static_assert(VectorDimension<DenseDims<3, 1>>);
static_assert(!MatrixDimension<DenseDims<3, 1>>);

using utils::Valid, utils::Optional;

template <class T, Dimension S, bool Compress = utils::Compressible<T>>
struct MATH_GSL_POINTER Array;
template <class T, Dimension S, bool Compress = utils::Compressible<T>>
struct MATH_GSL_POINTER MutArray;

// Cases we need to consider:
// 1. Slice-indexing
// 2.a. `ptrdiff_t` indexing, not compressed
// 2.b. `ptrdiff_t` indexing, compressed
// 3.a.i. Vector indexing, contig, no mask
// 3.a.ii. Vector indexing, contig, mask
// 3.b.i. Vector indexing, discontig, no mask
// 3.b.ii. Vector indexing, discontig, mask
// all of the above for `T*` and `const T*`
template <typename T, typename P, typename S, typename I>
[[gnu::flatten, gnu::always_inline]] constexpr auto
index(P *ptr, S shape, I i) noexcept -> decltype(auto) {
  auto offset = calcOffset(shape, i);
  auto new_dim = calcNewDim(shape, i);
  invariant(ptr != nullptr);
  using D = decltype(new_dim);
  if constexpr (!simd::index::issimd<D>) {
    constexpr bool compress = !std::same_as<T, std::remove_const_t<P>>;
    if constexpr (!std::same_as<D, Empty>)
      if constexpr (std::is_const_v<P>)
        return Array<T, D, compress>{ptr + offset, new_dim};
      else return MutArray<T, D, compress>{ptr + offset, new_dim};
    else if constexpr (!compress) return ptr[offset];
    else if constexpr (std::is_const_v<P>) return T::decompress(ptr + offset);
    else return utils::Reference<T>{ptr + offset};
  } else return simd::ref(ptr + offset, new_dim);
}
// for (row/col)vectors, we drop the row/col, essentially broadcasting
template <typename T, typename P, typename S, typename R, typename C>
[[gnu::flatten, gnu::always_inline]] constexpr auto
index(P *ptr, S shape, R wr, C wc) noexcept -> decltype(auto) {
  if constexpr (MatrixDimension<S>) {
    auto r = unwrapRow(wr);
    auto c = unwrapCol(wc);
    auto offset = calcOffset(shape, r, c);
    auto new_dim = calcNewDim(shape, r, c);
    using D = decltype(new_dim);
    if constexpr (!simd::index::issimd<D>) {
      constexpr bool compress = !std::same_as<T, std::remove_const_t<P>>;
      if constexpr (!std::same_as<D, Empty>)
        if constexpr (std::is_const_v<P>)
          return Array<T, D, compress>{ptr + offset, new_dim};
        else return MutArray<T, D, compress>{ptr + offset, new_dim};
      else if constexpr (!compress) return ptr[offset];
      else if constexpr (std::is_const_v<P>) return T::decompress(ptr + offset);
      else return utils::Reference<T>{ptr + offset};
    } else return simd::ref(ptr + offset, new_dim);
  } else if constexpr (ColVectorDimension<S>)
    return index<T>(ptr, shape, unwrapRow(wr));
  else return index<T>(ptr, shape, unwrapCol(wc));
}
static_assert(
  std::same_as<StridedDims<3>,
               decltype(calcNewDim(std::declval<DenseDims<3>>(), _, _(1, 5)))>);
static_assert(
  std::same_as<StridedDims<3>, decltype(calcNewDim(
                                 std::declval<StridedDims<3>>(), _, _(1, 5)))>);

template <typename T, bool Column, ptrdiff_t L, ptrdiff_t X>
struct SliceIterator {
  using dim_type = std::conditional_t<Column, StridedRange<L, X>, Length<L>>;
  using value_type =
    std::conditional_t<std::is_const_v<T>,
                       Array<std::remove_cvref_t<T>, dim_type>,
                       MutArray<std::remove_reference_t<T>, dim_type>>;
  using storage_type =
    std::conditional_t<std::is_const_v<T>, const compressed_t<T>,
                       compressed_t<T>>;
  storage_type *data_;
  [[no_unique_address]] Length<L> len_;
  [[no_unique_address]] RowStride<X> row_stride_;
  ptrdiff_t idx_;
  // constexpr auto operator=(const SliceIterator &) -> SliceIterator & =
  // default;
  // constexpr auto operator*() -> value_type;
  constexpr auto operator*() const -> value_type;
  constexpr auto operator++() -> SliceIterator & {
    idx_++;
    return *this;
  }
  constexpr auto operator++(int) -> SliceIterator {
    SliceIterator ret{*this};
    ++(*this);
    return ret;
  }
  constexpr auto operator--() -> SliceIterator & {
    idx_--;
    return *this;
  }
  constexpr auto operator--(int) -> SliceIterator {
    SliceIterator ret{*this};
    --(*this);
    return ret;
  }
  friend constexpr auto operator-(SliceIterator a,
                                  SliceIterator b) -> ptrdiff_t {
    return a.idx_ - b.idx_;
  }
  friend constexpr auto operator+(SliceIterator a,
                                  ptrdiff_t i) -> SliceIterator {
    return {a.data_, a.len_, a.row_stride_, a.idx_ + i};
  }
  friend constexpr auto operator==(SliceIterator a, SliceIterator b) -> bool {
    return a.idx_ == b.idx_;
  }
  friend constexpr auto operator<=>(SliceIterator a,
                                    SliceIterator b) -> std::strong_ordering {
    return a.idx_ <=> b.idx_;
  }
  friend constexpr auto operator==(SliceIterator a, Row<> r) -> bool
  requires(!Column)
  {
    return a.idx_ == r;
  }
  friend constexpr auto operator<=>(SliceIterator a,
                                    Row<> r) -> std::strong_ordering
  requires(!Column)
  {
    return a.idx_ <=> r;
  }
  friend constexpr auto operator==(SliceIterator a, Col<> r) -> bool
  requires(Column)
  {
    return a.idx_ == r;
  }
  friend constexpr auto operator<=>(SliceIterator a,
                                    Col<> r) -> std::strong_ordering
  requires(Column)
  {
    return a.idx_ <=> r;
  }
};

template <typename T, bool Column, ptrdiff_t L = -1, ptrdiff_t X = -1>
struct SliceRange {
  T *data_;
  [[no_unique_address]] Length<L> len_;
  [[no_unique_address]] RowStride<X> row_stride_;
  ptrdiff_t stop_;
  [[nodiscard]] constexpr auto begin() const -> SliceIterator<T, Column, L, X> {
    return {data_, len_, row_stride_, 0};
  }
  [[nodiscard]] constexpr auto end() const {
    if constexpr (Column) return col(stop_);
    else return row(stop_);
  }
  template <std::invocable<SliceRange> F>
  constexpr auto operator|(F &&f) const {
    return std::forward<F>(f)(*this);
  }
};
/// Constant Array
template <class T, Dimension S, bool Compress>
struct Array : public Expr<T, Array<T, S, Compress>> {
  static_assert(!std::is_const_v<T>, "T shouldn't be const");

  using storage_type = std::conditional_t<Compress, compressed_t<T>, T>;
  using value_type = T;
  using reference = T &;
  using const_reference = const T &;
  using size_type = ptrdiff_t;
  using difference_type = int;
  using iterator = storage_type *;
  using const_iterator = const storage_type *;
  using pointer = storage_type *;
  using const_pointer = const storage_type *;
  using concrete = std::true_type;
  static constexpr bool trivial =
    std::is_trivially_default_constructible_v<T> &&
    std::is_trivially_destructible_v<T>;
  static constexpr bool isdense = DenseLayout<S>;
  static constexpr bool flatstride =
    isdense || std::same_as<S, StridedRange<S::nrow, S::nstride>>;
  // static_assert(flatstride != std::same_as<S, StridedDims<>>);

  explicit constexpr Array() = default;
  constexpr Array(const Array &) = default;
  constexpr Array(Array &&) noexcept = default;
  constexpr auto operator=(const Array &) -> Array & = default;
  constexpr auto operator=(Array &&) noexcept -> Array & = default;
  constexpr Array(const storage_type *p, S s) : ptr(p), sz(s) {}
  constexpr Array(Valid<const storage_type> p, S s) : ptr(p), sz(s) {}
  template <ptrdiff_t R, ptrdiff_t C>
  constexpr Array(const storage_type *p, Row<R> r, Col<C> c)
    : ptr(p), sz(S{r, c}) {}
  template <ptrdiff_t R, ptrdiff_t C>
  constexpr Array(Valid<const storage_type> p, Row<R> r, Col<C> c)
    : ptr(p), sz(dimension<S>(r, c)) {}
  template <std::convertible_to<S> V>
  constexpr Array(Array<T, V> a) : ptr(a.data()), sz(a.dim()) {}
  template <size_t N>
  constexpr Array(const std::array<T, N> &a) : ptr(a.data()), sz(length(N)) {}
  [[nodiscard]] constexpr auto data() const noexcept -> const storage_type * {
    invariant(ptr != nullptr || ptrdiff_t(sz) == 0);
    return ptr;
  }
  [[nodiscard]] constexpr auto wrappedPtr() noexcept -> Valid<T> { return ptr; }

  [[nodiscard]] constexpr auto
  begin() const noexcept -> StridedIterator<const T>
  requires(std::is_same_v<S, StridedRange<S::nrow, S::nstride>>)
  {
    const storage_type *p = ptr;
    return StridedIterator{p, sz.stride_};
  }
  [[nodiscard]] constexpr auto begin() const noexcept
    -> const storage_type *requires(isdense) { return ptr; }

  [[nodiscard]] constexpr auto end() const noexcept
  requires(flatstride)
  {
    return begin() + ptrdiff_t(sz);
  }
  [[nodiscard]] constexpr auto rbegin() const noexcept
  requires(flatstride)
  {
    return std::reverse_iterator(end());
  }
  [[nodiscard]] constexpr auto rend() const noexcept
  requires(flatstride)
  {
    return std::reverse_iterator(begin());
  }
  [[nodiscard]] constexpr auto front() const noexcept -> const T & {
    return *ptr;
  }
  [[nodiscard]] constexpr auto back() const noexcept -> const T & {
    if constexpr (flatstride) return *(end() - 1);
    else return ptr[(sride(sz) * ptrdiff_t(row(sz))) - 1];
  }
  // indexing has two components:
  // 1. offsetting the pointer
  // 2. calculating new dim
  // static constexpr auto slice(Valid<T>, Index<S> auto i){
  //   auto
  // }
  [[gnu::flatten, gnu::always_inline]] constexpr auto
  operator[](Index<S> auto i) const noexcept -> decltype(auto) {
    return index<T>(ptr, sz, i);
  }
  // for (row/col)vectors, we drop the row/col, essentially broadcasting
  template <class R, class C>
  [[gnu::flatten, gnu::always_inline]] constexpr auto
  operator[](R r, C c) const noexcept -> decltype(auto) {
    return index<T>(ptr, sz, r, c);
  }
  template <size_t I>
  [[nodiscard]] constexpr auto get() const -> const T &
  requires(VectorDimension<S>)
  {
    return index<T>(ptr, sz, I);
  }

  [[nodiscard]] constexpr auto minRowCol() const -> ptrdiff_t {
    return std::min(ptrdiff_t(numRow()), ptrdiff_t(numCol()));
  }

  [[nodiscard]] constexpr auto diag() const noexcept {
    StridedRange<> r{length(minRowCol()), ++stride(sz)};
    invariant(ptr != nullptr);
    return Array<T, StridedRange<>>{ptr, r};
  }
  [[nodiscard]] constexpr auto antiDiag() const noexcept {
    StridedRange<> r{length(minRowCol()), --stride(sz)};
    invariant(ptr != nullptr);
    return Array<T, StridedRange<>>{ptr + ptrdiff_t(Col(sz)) - 1, r};
  }
  [[nodiscard]] constexpr auto isSquare() const noexcept -> bool {
    return ptrdiff_t(Row(sz)) == ptrdiff_t(Col(sz));
  }
  [[nodiscard]] constexpr auto checkSquare() const -> Optional<ptrdiff_t> {
    ptrdiff_t N = ptrdiff_t(numRow());
    if (N != ptrdiff_t(numCol())) return {};
    return N;
  }

  [[nodiscard]] constexpr auto numRow() const noexcept {
    if constexpr (RowVectorDimension<S>) return Row<1>{};
    else return row(sz);
  }
  [[nodiscard]] constexpr auto numCol() const noexcept { return col(sz); }
  [[nodiscard]] constexpr auto rowStride() const noexcept {
    if constexpr (std::same_as<S, Length<>>) return RowStride<1>{};
    else return stride(sz);
  }
  [[nodiscard]] constexpr auto empty() const -> bool {
    if constexpr (StaticLength<S>) return S::staticint() == 0;
    else return sz == S{};
  }
  [[nodiscard]] constexpr auto size() const noexcept {
    if constexpr (StaticLength<S>) return S::staticint();
    else return ptrdiff_t(sz);
  }
  [[nodiscard]] constexpr auto dim() const noexcept -> S { return sz; }
  constexpr void clear()
  requires(std::same_as<S, Length<>>)
  {
    sz = S{};
  }
  [[nodiscard]] constexpr auto t() const -> Transpose<T, Array> {
    return {*this};
  }
  [[nodiscard]] constexpr auto isExchangeMatrix() const -> bool
  requires(MatrixDimension<S>)
  {
    ptrdiff_t N = ptrdiff_t(numRow());
    if (N != ptrdiff_t(numCol())) return false;
    for (ptrdiff_t i = 0; i < N; ++i) {
      for (ptrdiff_t j = 0; j < N; ++j)
        if ((*this)[i, j] != (i + j == N - 1)) return false;
    }
  }
  [[nodiscard]] constexpr auto isDiagonal() const -> bool
  requires(MatrixDimension<S>)
  {
    for (ptrdiff_t r = 0; r < numRow(); ++r)
      for (ptrdiff_t c = 0; c < numCol(); ++c)
        if (r != c && (*this)[r, c] != 0) return false;
    return true;
  }
  [[nodiscard]] constexpr auto view() const noexcept -> Array<T, S> {
    return *this;
  }
  [[nodiscard]] constexpr auto flatview() const noexcept
  requires(flatstride)
  {
    auto szf = sz.flat();
    return Array<T, std::remove_reference_t<decltype(szf)>, Compress>{
      ptr, sz.flat()};
  }

  [[nodiscard]] constexpr auto
  operator==(const Array &other) const noexcept -> bool {
    if constexpr (MatrixDimension<S> && !DenseLayout<S>) {
      ptrdiff_t M = ptrdiff_t(other.numRow());
      if ((numRow() != M) || (numCol() != other.numCol())) return false;
      // may not be dense, iterate over rows
      for (ptrdiff_t i = 0; i < M; ++i)
        if ((*this)[i, _] != other[i, _]) return false;
    } else if constexpr (!MatrixDimension<S>) {
      constexpr ptrdiff_t W = simd::Width<T>;
      ptrdiff_t N = size();
      if (N != other.size()) return false;
      if constexpr (W == 1) {
        for (ptrdiff_t i = 0; i < N; ++i)
          if ((*this)[i] != other[i]) return false;
      } else {
        for (ptrdiff_t i = 0;; i += W) {
          auto u{simd::index::unrollmask<1, W>(N, i)};
          if (!u) break;
          if (simd::cmp::ne<W, T>((*this)[u], other[u])) return false;
        }
      }
    } else return flatview() == other.flatview();
    return true;
  }
  template <MatrixDimension D>
  [[nodiscard]] constexpr auto
  operator==(Array<T, D, Compress> other) const noexcept -> bool
  requires(MatrixDimension<S> && (((S::ncol != -1) && (D::ncol == -1)) ||
                                  ((S::nrow != -1) && (D::nrow == -1)) ||
                                  ((S::nstride != -1) && (D::nstride == -1))))
  {
    return Expr<T, Array<T, S, Compress>>::operator==(other);
  }
  // FIXME: strided should skip over elements
  [[nodiscard]] constexpr auto norm2() const noexcept -> value_type {
    static_assert(DenseLayout<S>);
    value_type ret{0};
    for (auto x : *this) ret += x * x;
    return ret;
    // return std::transform_reduce(begin(), end(), begin(), 0.0);
  }
  // FIXME: strided should skips over elements
  [[nodiscard]] constexpr auto sum() const noexcept -> value_type {
    static_assert(VectorDimension<S> || DenseLayout<S>);
    value_type ret{0};
    for (auto x : *this) ret += x;
    return ret;
    // return std::reduce(begin(), end());
  }
  // interpret a bigger object as smaller
  template <typename U> [[nodiscard]] auto reinterpretImpl() const {
    static_assert(sizeof(storage_type) % sizeof(U) == 0);
    static_assert(std::same_as<U, double>);
    if constexpr (std::same_as<U, T>) return *this;
    else {
      auto r = unwrapRow(numRow());
      auto c = unwrapCol(numCol());
      constexpr auto ratio = sizeof(storage_type) / sizeof(U);
#ifdef __cpp_lib_start_lifetime_as
      U *p = std::start_lifetime_as_array(reinterpret_cast<const U *>(data()),
                                          ratio * ptrdiff_t(rowStride()) * r);
#else
      const U *p = std::launder(reinterpret_cast<const U *>(data()));
#endif
      if constexpr (IsOne<decltype(r)>) {
        if constexpr (StaticInt<decltype(c)>)
          return Array<U, std::integral_constant<ptrdiff_t, c * ratio>>{p, {}};
        else return Array<U, ptrdiff_t>{p, c * ratio};
      } else if constexpr (DenseLayout<S>) {
        return Array<U, DenseDims<>>(p, DenseDims(row(r), col(c * ratio)));
      } else {
        ptrdiff_t str = ptrdiff_t(rowStride()) * ratio;
        if constexpr (IsOne<decltype(c)>) {
          constexpr auto sr = std::integral_constant<ptrdiff_t, ratio>{};
          return Array<U, StridedDims<-1, ratio, -1>>(
            p, StridedDims(row(r), col(sr), stride(str)));
        } else
          return Array<U, StridedDims<>>(
            p, StridedDims(row(r), col(c * ratio), stride(str)));
      }
    }
  }
  friend void PrintTo(const Array &x, ::std::ostream *os) { *os << x; }
  [[nodiscard]] friend constexpr auto
  operator<=>(Array x, Array y) -> std::strong_ordering {
    ptrdiff_t M = x.size();
    ptrdiff_t N = y.size();
    for (ptrdiff_t i = 0, L = std::min(M, N); i < L; ++i)
      if (auto cmp = x[i] <=> y[i]; cmp != 0) return cmp;
    return M <=> N;
  };
  [[nodiscard]] constexpr auto
  eachRow() const -> SliceRange<const T, false, S::ncol, S::nstride>
  requires(MatrixDimension<S>)
  {
    return {data(), aslength(col(this->sz)), stride(this->sz),
            ptrdiff_t(row(this->sz))};
  }
  [[nodiscard]] constexpr auto
  eachCol() const -> SliceRange<const T, true, S::nrow, S::nstride>
  requires(MatrixDimension<S>)
  {
    return {data(), aslength(row(this->sz)), stride(this->sz),
            ptrdiff_t(col(this->sz))};
  }
  friend auto operator<<(std::ostream &os, Array x) -> std::ostream &
  requires(utils::Printable<T>)
  {
    if constexpr (MatrixDimension<S>)
      return utils::printMatrix(os, x.data(), ptrdiff_t(x.numRow()),
                                ptrdiff_t(x.numCol()),
                                ptrdiff_t(x.rowStride()));
    else return utils::printVector(os, x.begin(), x.end());
  }
  [[nodiscard]] constexpr auto split(ptrdiff_t at) const
    -> containers::Pair<Array<T, Length<>>, Array<T, Length<>>>
  requires(VectorDimension<S>)
  {
    invariant(at <= size());
    return {(*this)[_(0, at)], (*this)[_(at, math::end)]};
  }
  [[nodiscard]] constexpr auto
  popFront() const -> containers::Pair<T, Array<T, Length<>>>
  requires(VectorDimension<S>)
  {
    invariant(0 < size());
    return {ptr[0], (*this)[_(1, math::end)]};
  }
#ifndef NDEBUG
  [[gnu::used]] void dump() const {
    // we can't combine `gnu::used` with `requires(utils::Printable<T>)`
    // requires(utils::Printable<T>)
    if constexpr (utils::Printable<T>)
      std::cout << "Size: " << ptrdiff_t(sz) << " " << *this << "\n";
  }
  // [[gnu::used]] void dump(const char *filename) const {
  //   if constexpr (std::integral<T>) {
  //     std::FILE *f = std::fopen(filename, "w");
  //     if (f == nullptr) return;
  //     (void)std::fprintf(f, "C= [");
  //     if constexpr (MatrixDimension<S>) {
  //       for (ptrdiff_t i = 0; i < Row(sz); ++i) {
  //         if (i) (void)std::fprintf(f, "\n");
  //         (void)std::fprintf(f, "%ld", long((*this)[i, 0]));
  //         for (ptrdiff_t j = 1; j < Col(sz); ++j)
  //           (void)std::fprintf(f, " %ld", long((*this)[i, j]));
  //       }
  //     } else {
  //       (void)std::fprintf(f, "%ld", long((*this)[0]));
  //       for (ptrdiff_t i = 1; (i < ptrdiff_t(sz)); ++i)
  //         (void)std::fprintf(f, ", %ld", long((*this)[i]));
  //     }
  //     (void)std::fprintf(f, "]");
  //     (void)std::fclose(f);
  //   }
  // }
#endif
protected:
  constexpr Array(S s) : sz(s) {}
  // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
  const storage_type *ptr;
  // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
  [[no_unique_address]] S sz;
};
template <class T, class S>
[[gnu::always_inline]] constexpr auto view(Array<T, S> x) -> Array<T, S> {
  return x;
}

static_assert(
  std::same_as<utils::eltype_t<Array<int64_t, SquareDims<>>>, int64_t>);
static_assert(AbstractMatrix<Array<int64_t, SquareDims<>>>);

static_assert(
  std::is_trivially_default_constructible_v<Array<int64_t, DenseDims<>>>);

template <class T, Dimension S, bool Compress>
struct MutArray : Array<T, S, Compress>,
                  ArrayOps<T, S, MutArray<T, S, Compress>> {
  using BaseT = Array<T, S, Compress>;
  // using BaseT::BaseT;
  using BaseT::operator[], BaseT::data, BaseT::begin, BaseT::end, BaseT::rbegin,
    BaseT::rend, BaseT::front, BaseT::back;
  using storage_type = typename BaseT::storage_type;

  explicit constexpr MutArray() = default;
  explicit constexpr MutArray(const MutArray &) = default;
  explicit constexpr MutArray(MutArray &&) noexcept = default;
  // constexpr auto operator=(const MutArray &) -> MutArray & = delete;
  constexpr auto operator=(const MutArray &) -> MutArray & = default;
  constexpr auto operator=(MutArray &&) noexcept -> MutArray & = default;
  constexpr MutArray(storage_type *p, S s) : BaseT(p, s) {}

  constexpr void truncate(S nz) {
    S oz = this->sz;
    this->sz = nz;
    if constexpr (std::same_as<S, Length<>>) {
      invariant(ptrdiff_t(nz) <= ptrdiff_t(oz));
      if constexpr (!std::is_trivially_destructible_v<T>)
        if (nz < oz) std::destroy_n(this->data() + ptrdiff_t(nz), oz - nz);
    } else if constexpr (std::convertible_to<S, DenseDims<>>) {
      static_assert(
        std::is_trivially_destructible_v<T>,
        "Truncating matrices holding non-is_trivially_destructible_v "
        "objects is not yet supported.");
      auto new_x = ptrdiff_t{RowStride(nz)}, old_x = ptrdiff_t{RowStride(oz)},
           new_n = ptrdiff_t{Col(nz)}, old_n = ptrdiff_t{Col(oz)},
           new_m = ptrdiff_t{Row(nz)}, old_m = ptrdiff_t{Row(oz)};
      invariant(new_m <= old_m);
      invariant(new_n <= old_n);
      invariant(new_x <= old_x);
      ptrdiff_t cols_to_copy = new_n, rows_to_copy = new_m;
      // we only need to copy if memory shifts position
      bool copy_cols = ((cols_to_copy > 0) && (new_x != old_x));
      // if we're in place, we have 1 less row to copy
      if (((--rows_to_copy) > 0) && (copy_cols)) {
        // truncation, we need to copy rows to increase stride
        storage_type *src = data(), *dst = src;
        do {
          src += old_x;
          dst += new_x;
          std::copy_n(src, cols_to_copy, dst);
        } while (--rows_to_copy);
      }
    } else {
      static_assert(MatrixDimension<S>, "Can only resize 1 or 2d containers.");
      invariant(nz.row() <= oz.row());
      invariant(nz.col() <= oz.col());
    }
  }
  constexpr void truncate(ptrdiff_t nz)
  requires(std::same_as<S, Length<>>)
  {
    truncate(length(nz));
  }

  constexpr void truncate(Row<> r) {
    if constexpr (std::same_as<S, Length<>>) {
      return truncate(S(r));
    } else if constexpr (std::convertible_to<S, DenseDims<>>) {
      static_assert(!std::convertible_to<S, SquareDims<>>,
                    "if truncating a row, matrix must be strided or dense.");
      invariant(r <= Row(this->sz));
      DenseDims new_sz = this->sz;
      truncate(new_sz.set(r));
    } else {
      static_assert(std::convertible_to<S, StridedDims<>>);
      invariant(r <= Row(this->sz));
      this->sz.set(r);
    }
  }
  constexpr void truncate(Col<> c) {
    if constexpr (std::same_as<S, Length<>>) {
      return truncate(S(c));
    } else if constexpr (std::is_same_v<S, DenseDims<>>) {
      static_assert(!std::convertible_to<S, SquareDims<>>,
                    "if truncating a col, matrix must be strided or dense.");
      invariant(c <= Col(this->sz));
      DenseDims new_sz = this->sz;
      truncate(new_sz.set(c));
    } else {
      static_assert(std::convertible_to<S, StridedDims<>>);
      invariant(c <= Col(this->sz));
      this->sz.set(c);
    }
  }

  template <class... Args>
  constexpr MutArray(Args &&...args)
  requires(std::constructible_from<Array<T, S, Compress>, Args...>)
    : Array<T, S, Compress>(std::forward<Args>(args)...) {}

  template <std::convertible_to<T> U, std::convertible_to<S> V>
  constexpr MutArray(Array<U, V> a) : Array<T, S>(a) {}
  template <size_t N>
  constexpr MutArray(std::array<T, N> &a) : Array<T, S>(a.data(), length(N)) {}
  [[nodiscard]] constexpr auto data() noexcept -> storage_type * {
    invariant(this->ptr != nullptr || ptrdiff_t(this->sz) == 0);
    return const_cast<storage_type *>(this->ptr);
  }
  [[nodiscard]] constexpr auto wrappedPtr() noexcept -> Valid<T> {
    return data();
  }

  [[nodiscard]] constexpr auto begin() noexcept -> StridedIterator<T>
  requires(std::is_same_v<S, StridedRange<S::nrow, S::nstride>>)
  {
    return StridedIterator{const_cast<storage_type *>(this->ptr),
                           this->sz.stride_};
  }
  [[nodiscard]] constexpr auto
  begin() noexcept -> storage_type *requires(
                     !std::is_same_v<S, StridedRange<S::nrow, S::nstride>>) {
    return const_cast<storage_type *>(this->ptr);
  }

  [[nodiscard]] constexpr auto end() noexcept {
    return begin() + ptrdiff_t(this->sz);
  }
  [[nodiscard]] constexpr auto rbegin() noexcept {
    return std::reverse_iterator(end());
  }
  [[nodiscard]] constexpr auto rend() noexcept {
    return std::reverse_iterator(begin());
  }
  constexpr auto front() noexcept -> T & {
    utils::assume(this->size() > 0);
    return *begin();
  }
  constexpr auto back() noexcept -> T & {
    utils::assume(this->size() > 0);
    return *(end() - 1);
  }
  [[gnu::flatten, gnu::always_inline]] constexpr auto
  operator[](Index<S> auto i) noexcept -> decltype(auto) {
    return index<T>(data(), this->sz, i);
  }
  template <class R, class C>
  [[gnu::flatten, gnu::always_inline]] constexpr auto
  operator[](R r, C c) noexcept -> decltype(auto) {
    return index<T>(data(), this->sz, r, c);
  }

  template <size_t I>
  [[nodiscard]] constexpr auto get() -> T &
  requires(VectorDimension<S>)
  {
    return index<T>(data(), this->sz, I);
  }

  constexpr void fill(T value) {
    std::fill_n(this->data(), ptrdiff_t(this->dim()), value);
  }
  constexpr void zero() {
    if constexpr (std::is_trivially_default_constructible_v<T> &&
                  std::is_trivially_destructible_v<T>)
      std::memset(data(), 0, this->size() * sizeof(storage_type));
    else std::fill_n(this->data(), this->size(), T{});
  }
  [[nodiscard]] constexpr auto diag() noexcept {
    Length l =
      length(std::min(ptrdiff_t(Row(this->sz)), ptrdiff_t(Col(this->sz))));
    RowStride rs = RowStride(this->sz);
    StridedRange<> r{l, ++rs};
    return MutArray<T, StridedRange<>>{data(), r};
  }
  [[nodiscard]] constexpr auto antiDiag() noexcept {
    Col<> c = Col(this->sz);
    Length l =
      length(ptrdiff_t(std::min(ptrdiff_t(Row(this->sz)), ptrdiff_t(c))));
    RowStride rs = RowStride(this->sz);
    StridedRange<> r{l, --rs};
    return MutArray<T, StridedRange<>>{data() + ptrdiff_t(c) - 1, r};
  }
  constexpr void erase(ptrdiff_t i)
  requires(std::same_as<S, Length<>>)
  {
    auto old_len = ptrdiff_t(this->sz--);
    if (i < this->sz) std::copy(data() + i + 1, data() + old_len, data() + i);
  }
  constexpr void erase_swap_last(ptrdiff_t i)
  requires(std::same_as<S, Length<>>)
  {
    --(this->sz);
    if (i < this->sz) std::swap(data()[i], data()[ptrdiff_t(this->sz)]);
  }
  constexpr void erase(Row<> r) {
    if constexpr (std::same_as<S, Length<>>) {
      return erase(S(r));
    } else if constexpr (std::convertible_to<S, DenseDims<>>) {
      static_assert(!std::convertible_to<S, SquareDims<>>,
                    "if erasing a row, matrix must be strided or dense.");
      auto col = ptrdiff_t{Col(this->sz)},
           new_row = ptrdiff_t{Row(this->sz)} - 1;
      this->sz.set(row(new_row));
      if ((col == 0) || (r == new_row)) return;
      storage_type *dst = data() + (ptrdiff_t(r) * col);
      std::copy_n(dst + col, (new_row - ptrdiff_t(r)) * col, dst);
    } else {
      static_assert(std::convertible_to<S, StridedDims<>>);
      auto stride = ptrdiff_t{RowStride(this->sz)},
           col = ptrdiff_t{Col(this->sz)},
           new_row = ptrdiff_t{Row(this->sz)} - 1;
      this->sz.set(row(new_row));
      if ((col == 0) || (r == new_row)) return;
      invariant(col <= stride);
      if ((col + (512 / (sizeof(T)))) <= stride) {
        storage_type *dst = data() + (ptrdiff_t(r) * stride);
        for (auto m = ptrdiff_t(r); m < new_row; ++m) {
          storage_type *src = dst + stride;
          std::copy_n(src, col, dst);
          dst = src;
        }
      } else {
        storage_type *dst = data() + (ptrdiff_t(r) * stride);
        std::copy_n(dst + stride, (new_row - ptrdiff_t(r)) * stride, dst);
      }
    }
  }
  constexpr void erase(Col<> c) {
    if constexpr (std::same_as<S, Length<>>) {
      return erase(S(c));
    } else if constexpr (std::convertible_to<S, DenseDims<>>) {
      static_assert(!std::convertible_to<S, SquareDims<>>,
                    "if erasing a col, matrix must be strided or dense.");
      auto new_col = ptrdiff_t{Col(this->sz)}, old_col = new_col--,
           row = ptrdiff_t{Row(this->sz)};
      this->sz.set(col(new_col));
      ptrdiff_t cols_to_copy = new_col - ptrdiff_t(c);
      if ((cols_to_copy == 0) || (row == 0)) return;
      // we only need to copy if memory shifts position
      for (ptrdiff_t m = 0; m < row; ++m) {
        storage_type *dst = data() + (m * new_col) + ptrdiff_t(c);
        storage_type *src = data() + (m * old_col) + ptrdiff_t(c) + 1;
        std::copy_n(src, cols_to_copy, dst);
      }
    } else {
      static_assert(std::convertible_to<S, StridedDims<>>);
      auto stride = ptrdiff_t{RowStride(this->sz)},
           new_col = ptrdiff_t{Col(this->sz)} - 1,
           row = ptrdiff_t{Row(this->sz)};
      this->sz.set(col(new_col));
      ptrdiff_t cols_to_copy = new_col - ptrdiff_t(c);
      if ((cols_to_copy == 0) || (row == 0)) return;
      // we only need to copy if memory shifts position
      for (ptrdiff_t m = 0; m < row; ++m) {
        storage_type *dst = data() + (m * stride) + ptrdiff_t(c);
        std::copy_n(dst + 1, cols_to_copy, dst);
      }
    }
  }
  constexpr void moveLast(Col<> j) {
    static_assert(MatrixDimension<S>);
    if (j == this->numCol()) return;
    Col Nd = this->numCol() - 1;
    for (ptrdiff_t m = 0; m < this->numRow(); ++m) {
      auto x = (*this)[m, ptrdiff_t(j)];
      for (auto n = ptrdiff_t(j); n < Nd;) {
        ptrdiff_t o = n++;
        (*this)[m, o] = (*this)[m, n];
      }
      (*this)[m, Nd] = x;
    }
  }
  constexpr auto eachRow() -> SliceRange<T, false, S::ncol, S::nstride>
  requires(MatrixDimension<S>)
  {
    return {data(), aslength(col(this->sz)), stride(this->sz),
            ptrdiff_t(row(this->sz))};
  }
  constexpr auto eachCol() -> SliceRange<T, true, S::nrow, S::nstride>
  requires(MatrixDimension<S>)
  {
    return {data(), aslength(row(this->sz)), stride(this->sz),
            ptrdiff_t(col(this->sz))};
  }
  template <typename U> [[nodiscard]] auto reinterpretImpl() {
    static_assert(sizeof(storage_type) % sizeof(U) == 0);
    static_assert(std::same_as<U, double>);
    if constexpr (std::same_as<U, T>) return *this;
    else {
      auto r = unwrapRow(this->numRow());
      auto c = unwrapCol(this->numCol());
      constexpr size_t ratio = sizeof(storage_type) / sizeof(U);
#ifdef __cpp_lib_start_lifetime_as
      U *p = std::start_lifetime_as_array(reinterpret_cast<const U *>(data()),
                                          ratio * ptrdiff_t(rowStride()) * r);
#else
      U *p = std::launder(reinterpret_cast<U *>(data()));
#endif
      if constexpr (IsOne<decltype(r)>) {
        if constexpr (StaticInt<decltype(c)>)
          return MutArray<U, std::integral_constant<ptrdiff_t, c * ratio>>(
            p, std::integral_constant<ptrdiff_t, c * ratio>{});
        else return MutArray<U, ptrdiff_t>{p, c * ratio};
      } else if constexpr (DenseLayout<S>) {
        return MutArray<U, DenseDims<>>(p, DenseDims(row(r), col(c * ratio)));
      } else {
        ptrdiff_t str = ptrdiff_t(this->rowStride()) * ratio;
        if constexpr (IsOne<decltype(c)>) {
          constexpr auto sr = std::integral_constant<ptrdiff_t, ratio>{};
          return MutArray<U, StridedDims<-1, ratio, -1>>(
            p, StridedDims(row(r), col(sr), stride(str)));
        } else
          return MutArray<U, StridedDims<>>(
            p, StridedDims(row(r), col(c * ratio), stride(str)));
      }
    }
  }
  [[nodiscard]] constexpr auto mview() noexcept -> MutArray<T, S> {
    return *this;
  }

protected:
  constexpr MutArray(S s) : BaseT(s) {}
};

template <typename T, typename S> Array(T *, S) -> Array<decompressed_t<T>, S>;
template <typename T, typename S>
MutArray(T *, S) -> MutArray<decompressed_t<T>, S>;

template <typename T, typename S> MutArray(MutArray<T, S>) -> MutArray<T, S>;

static_assert(std::convertible_to<Array<int64_t, SquareDims<>>,
                                  Array<int64_t, DenseDims<>>>);
static_assert(std::convertible_to<Array<int64_t, DenseDims<8, 8>>,
                                  Array<int64_t, DenseDims<>>>);
static_assert(std::convertible_to<Array<int64_t, SquareDims<>>,
                                  Array<int64_t, StridedDims<>>>);
static_assert(std::convertible_to<Array<int64_t, DenseDims<>>,
                                  Array<int64_t, StridedDims<>>>);
static_assert(std::convertible_to<MutArray<int64_t, SquareDims<>>,
                                  Array<int64_t, DenseDims<>>>);
static_assert(std::convertible_to<MutArray<int64_t, SquareDims<>>,
                                  Array<int64_t, StridedDims<>>>);
static_assert(std::convertible_to<MutArray<int64_t, DenseDims<>>,
                                  Array<int64_t, StridedDims<>>>);
static_assert(std::convertible_to<MutArray<int64_t, SquareDims<>>,
                                  MutArray<int64_t, DenseDims<>>>);
static_assert(std::convertible_to<MutArray<int64_t, SquareDims<>>,
                                  MutArray<int64_t, StridedDims<>>>);
static_assert(std::convertible_to<MutArray<int64_t, DenseDims<>>,
                                  MutArray<int64_t, StridedDims<>>>);
static_assert(AbstractVector<Array<int64_t, Length<>>>);
static_assert(!AbstractVector<Array<int64_t, StridedDims<>>>);
static_assert(AbstractMatrix<Array<int64_t, StridedDims<>>>);
static_assert(RowVector<Array<int64_t, Length<>>>);
static_assert(ColVector<Transpose<int64_t, Array<int64_t, Length<>>>>);
static_assert(ColVector<Array<int64_t, StridedRange<>>>);
static_assert(RowVector<Transpose<int64_t, Array<int64_t, StridedRange<>>>>);

// template <typename T, bool Column>
// inline constexpr auto SliceIterator<T, Column>::operator*()
//   -> SliceIterator<T, Column>::value_type {
//   if constexpr (Column) return {data + idx, StridedRange<>{len, rowStride}};
//   else return {data + rowStride * idx, len};
// }
template <typename T, bool Column, ptrdiff_t L, ptrdiff_t X>
constexpr auto SliceIterator<T, Column, L, X>::operator*() const
  -> SliceIterator<T, Column, L, X>::value_type {
  if constexpr (Column)
    return {data_ + idx_, StridedRange<L, X>{len_, row_stride_}};
  else return {data_ + (ptrdiff_t(row_stride_) * idx_), len_};
}
// template <typename T, bool Column>
// inline constexpr auto operator*(SliceIterator<T, Column> it)
//   -> SliceIterator<T, Column>::value_type {
//   if constexpr (Column)
//     return {it.data + it.idx, StridedRange<>{it.len, it.rowStride}};
//   else return {it.data + it.rowStride * it.idx, it.len};
// }

static_assert(std::weakly_incrementable<SliceIterator<int64_t, false, -1, -1>>);
static_assert(std::forward_iterator<SliceIterator<int64_t, false, -1, -1>>);
static_assert(std::ranges::forward_range<SliceRange<int64_t, false>>);
static_assert(std::ranges::range<SliceRange<int64_t, false>>);
using ITEST =
  std::iter_rvalue_reference_t<SliceIterator<int64_t, true, -1, -1>>;
static_assert(std::is_same_v<ITEST, MutArray<int64_t, StridedRange<>, false>>);

/// Non-owning view of a managed array, capable of resizing,
/// but not of re-allocating in case the capacity is exceeded.
template <class T, Dimension S>
struct MATH_GSL_POINTER ResizeableView : MutArray<T, S> {
  using BaseT = MutArray<T, S>;
  using U = containers::default_capacity_type_t<S>;
  using storage_type = typename BaseT::storage_type;
  constexpr ResizeableView() noexcept : BaseT(nullptr, S{}), capacity_(U{}) {}
  constexpr ResizeableView(storage_type *p, S s, U c) noexcept
    : BaseT(p, s), capacity_(c) {}
  constexpr ResizeableView(alloc::Arena<> *a, U c) noexcept
    : ResizeableView{a->template allocate<storage_type>(ptrdiff_t(c)), S{}, c} {
  }
  constexpr ResizeableView(alloc::Arena<> *a, S s, U c) noexcept
    : ResizeableView{a->template allocate<storage_type>(ptrdiff_t(c)), s, c} {}

  [[nodiscard]] constexpr auto isFull() const -> bool {
    return ptrdiff_t(this->sz) == capacity_;
  }

  template <class... Args>
  constexpr auto emplace_back_within_capacity(Args &&...args) -> decltype(auto)
  requires(std::same_as<S, Length<>>)
  {
    auto s = ptrdiff_t(this->sz), c = ptrdiff_t(this->capacity_);
    invariant(s < c);
    T &ret =
      *std::construct_at<T>(this->data() + s, std::forward<Args>(args)...);
    this->sz = length(s + 1z);
    return ret;
  }
  /// Allocates extra space if needed
  /// Has a different name to make sure we avoid ambiguities.
  template <class... Args>
  constexpr auto emplace_backa(alloc::Arena<> *alloc,
                               Args &&...args) -> decltype(auto)
  requires(std::same_as<S, Length<>>)
  {
    if (isFull()) reserve(alloc, (ptrdiff_t(capacity_) + 1z) * 2z);
    return *std::construct_at(this->data() + ptrdiff_t(this->sz++),
                              std::forward<Args>(args)...);
  }
  constexpr void push_back_within_capacity(T value)
  requires(std::same_as<S, Length<>>)
  {
    auto s = ptrdiff_t(this->sz), c = ptrdiff_t(this->capacity_);
    invariant(s < c);
    std::construct_at<T>(this->data() + s, std::move(value));
    this->sz = length(s + 1z);
  }
  constexpr void push_back(alloc::Arena<> *alloc, T value)
  requires(std::same_as<S, Length<>>)
  {
    if (isFull()) reserve(alloc, (ptrdiff_t(capacity_) + 1z) * 2z);
    std::construct_at<T>(this->data() + ptrdiff_t(this->sz++),
                         std::move(value));
  }
  constexpr void pop_back()
  requires(std::same_as<S, Length<>>)
  {
    invariant(this->sz > 0);
    if constexpr (std::is_trivially_destructible_v<T>) --this->sz;
    else std::destroy_at(this->data() + ptrdiff_t(--this->sz));
  }
  constexpr auto pop_back_val() -> T
  requires(std::same_as<S, Length<>>)
  {
    invariant(this->sz > 0);
    return std::move(this->data()[ptrdiff_t(--this->sz)]);
  }
  constexpr void resize(ptrdiff_t nz)
  requires(std::same_as<S, Length<>>)
  {
    resize(length(nz));
  }
  // behavior
  // if S is StridedDims, then we copy data.
  // If the new dims are larger in rows or cols, we fill with 0.
  // If the new dims are smaller in rows or cols, we truncate.
  // New memory outside of dims (i.e., stride larger), we leave uninitialized.
  //
  constexpr void resize(S nz) {
    S oz = this->sz;
    this->sz = nz;
    if constexpr (std::same_as<S, Length<>>) {
      auto ozs = ptrdiff_t(oz), nzs = ptrdiff_t(nz);
      invariant(nzs <= capacity_);
      if constexpr (!std::is_trivially_destructible_v<T>) {
        if (nzs < ozs) std::destroy_n(this->data() + nz, oz - nz);
        else if (nzs > ozs)
          for (ptrdiff_t i = ozs; i < nzs; ++i)
            std::construct_at(this->data() + i);
      } else if (nz > oz)
        std::fill(this->data() + ozs, this->data() + nzs, T{});
    } else {
      static_assert(std::is_trivially_destructible_v<T>,
                    "Resizing matrices holding non-is_trivially_destructible_v "
                    "objects is not yet supported.");
      static_assert(MatrixDimension<S>, "Can only resize 1 or 2d containers.");
      auto new_x = ptrdiff_t{RowStride(nz)}, old_x = ptrdiff_t{RowStride(oz)},
           new_n = ptrdiff_t{Col(nz)}, old_n = ptrdiff_t{Col(oz)},
           new_m = ptrdiff_t{Row(nz)}, old_m = ptrdiff_t{Row(oz)};
      invariant(U(nz) <= capacity_);
      U len = U(nz);
      T *npt = this->data();
      // we can copy forward so long as the new stride is smaller
      // so that the start of the dst range is outside of the src range
      // we can also safely forward copy if we allocated a new ptr
      bool forward_copy = (new_x <= old_x);
      ptrdiff_t cols_to_copy = std::min(old_n, new_n);
      // we only need to copy if memory shifts position
      bool copy_cols = ((cols_to_copy > 0) && (new_x != old_x));
      // if we're in place, we have 1 less row to copy
      ptrdiff_t rows_to_copy = std::min(old_m, new_m) - 1;
      ptrdiff_t fill_count = new_n - cols_to_copy;
      if ((rows_to_copy) && (copy_cols || fill_count)) {
        if (forward_copy) {
          // truncation, we need to copy rows to increase stride
          T *src = this->data() + old_x;
          T *dst = npt + new_x;
          do {
            if (copy_cols) std::copy_n(src, cols_to_copy, dst);
            if (fill_count) std::fill_n(dst + cols_to_copy, fill_count, T{});
            src += old_x;
            dst += new_x;
          } while (--rows_to_copy);
        } else /* [[unlikely]] */ {
          // backwards copy, only needed when we increasing stride but not
          // reallocating, which should be comparatively uncommon.
          // Should probably benchmark or determine actual frequency
          // before adding `[[unlikely]]`.
          T *src = this->data() + (rows_to_copy + 1) * old_x;
          T *dst = npt + (rows_to_copy + 1) * new_x;
          do {
            src -= old_x;
            dst -= new_x;
            if (cols_to_copy)
              std::copy_backward(src, src + cols_to_copy, dst + cols_to_copy);
            if (fill_count) std::fill_n(dst + cols_to_copy, fill_count, T{});
          } while (--rows_to_copy);
        }
      }
      // zero init remaining rows
      for (ptrdiff_t m = old_m; m < new_m; ++m)
        std::fill_n(npt + m * new_x, new_n, T{});
    }
  }

  constexpr void resize(Row<> r) {
    if constexpr (std::same_as<S, Length<>>) {
      return resize(S(r));
    } else if constexpr (MatrixDimension<S>) {
      S nz = this->sz;
      return resize(nz.set(r));
    }
  }
  constexpr void resize(Col<> c) {
    if constexpr (std::same_as<S, Length<>>) {
      return resize(S(c));
    } else if constexpr (MatrixDimension<S>) {
      S nz = this->sz;
      return resize(nz.set(c));
    }
  }
  constexpr void resizeForOverwrite(S M) {
    invariant(ptrdiff_t(M) <= ptrdiff_t(this->sz));
    if constexpr (!std::is_trivially_destructible_v<T>) {
      ptrdiff_t nz = ptrdiff_t(M), oz = ptrdiff_t(this->sz);
      if (nz < oz) std::destroy_n(this->data() + nz, oz - nz);
      else if (nz > oz) // FIXME: user should initialize?
        for (ptrdiff_t i = oz; i < nz; ++i) std::construct_at(this->data() + i);
    }
    this->sz = M;
  }
  constexpr void resizeForOverwrite(ptrdiff_t M)
  requires(std::same_as<S, Length<>>)
  {
    resizeForOverwrite(length(M));
  }
  constexpr void resizeForOverwrite(Row<> r) {
    if constexpr (std::same_as<S, Length<>>) {
      return resizeForOverwrite(S(r));
    } else if constexpr (MatrixDimension<S>) {
      S nz = this->sz;
      return resizeForOverwrite(nz.set(r));
    }
  }
  constexpr void resizeForOverwrite(Col<> c) {
    if constexpr (std::same_as<S, Length<>>) {
      return resizeForOverwrite(S(c));
    } else if constexpr (MatrixDimension<S>) {
      S nz = this->sz;
      return resizeForOverwrite(nz.set(c));
    }
  }
  [[nodiscard]] constexpr auto getCapacity() const -> U { return capacity_; }

  // set size and 0.
  constexpr void setSize(Row<> r, Col<> c) {
    resizeForOverwrite({r, c});
    this->fill(0);
  }
  constexpr void resize(Row<> MM, Col<> NN) { resize(DenseDims{MM, NN}); }
  constexpr void resizeForOverwrite(Row<> M, Col<> N, RowStride<> X) {
    invariant(X >= N);
    if constexpr (std::is_same_v<S, StridedDims<>>)
      resizeForOverwrite(S{M, N, X});
    else if constexpr (std::is_same_v<S, SquareDims<>>) {
      invariant(ptrdiff_t(M) == ptrdiff_t(N));
      resizeForOverwrite(S{M});
    } else {
      static_assert(std::is_same_v<S, DenseDims<>>);
      resizeForOverwrite(S{M, N});
    }
  }
  constexpr void resizeForOverwrite(Row<> M, Col<> N) {
    if constexpr (std::is_same_v<S, StridedDims<>>)
      resizeForOverwrite(S{M, N, {ptrdiff_t(N)}});
    else if constexpr (std::is_same_v<S, SquareDims<>>) {
      invariant(ptrdiff_t(M) == ptrdiff_t(N));
      resizeForOverwrite(S{M});
    } else resizeForOverwrite(S{M, N});
  }

  constexpr auto
  insert_within_capacity(T *p, T x) -> T *requires(std::same_as<S, Length<>>) {
    invariant(p >= this->data());
    T *e = this->data() + ptrdiff_t(this->sz);
    invariant(p <= e);
    invariant(this->sz < capacity_);
    if constexpr (BaseT::trivial) {
      if (p < e) std::copy_backward(p, e, e + 1);
      *p = std::move(x);
    } else if (p < e) {
      std::construct_at<T>(e, std::move(*(e - 1)));
      for (; --e != p;) *e = std::move(*(e - 1));
      *p = std::move(x);
    } else std::construct_at<T>(e, std::move(x));
    ++this->sz;
    return p;
  }

  template <size_t SlabSize, bool BumpUp>
  constexpr void reserve(alloc::Arena<SlabSize, BumpUp> *alloc,
                         ptrdiff_t newCapacity) {
    if (newCapacity <= capacity_) return;
    this->ptr = alloc->template reallocate<false, T>(
      const_cast<T *>(this->ptr), ptrdiff_t(capacity_), newCapacity,
      ptrdiff_t(this->sz));
    // T *oldPtr =
    //   std::exchange(this->data(), alloc->template
    //   allocate<T>(newCapacity));
    // std::copy_n(oldPtr, U(this->sz), this->data());
    // alloc->deallocate(oldPtr, capacity);
    capacity_ = capacity(newCapacity);
  }

protected:
  constexpr ResizeableView(S s, U c) noexcept : BaseT(s), capacity_(c) {}
  // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
  [[no_unique_address]] U capacity_{0};
};

static_assert(std::is_copy_assignable_v<Array<void *, Length<>>>);
static_assert(std::is_copy_assignable_v<MutArray<void *, Length<>>>);
static_assert(std::is_trivially_copyable_v<MutArray<void *, Length<>>>);
static_assert(std::is_trivially_move_assignable_v<MutArray<void *, Length<>>>);
[[nodiscard]] constexpr auto newCapacity(ptrdiff_t c) -> ptrdiff_t {
  return c ? c + c : 4z;
}

template <class T, class S>
concept AbstractSimilar = (MatrixDimension<S> && AbstractMatrix<T>) ||
                          (VectorDimension<S> && AbstractVector<T>);

template <class T> using PtrVector = Array<T, Length<>>;
template <class T> using MutPtrVector = MutArray<T, Length<>>;

static_assert(AbstractVector<Array<int64_t, Length<>>>);
static_assert(AbstractVector<MutArray<int64_t, Length<>>>);
static_assert(!AbstractVector<int64_t>);

template <typename T> using StridedVector = Array<T, StridedRange<>>;
template <typename T> using MutStridedVector = MutArray<T, StridedRange<>>;

static_assert(!AbstractMatrix<StridedVector<int64_t>>);

static_assert(std::is_trivially_copyable_v<MutStridedVector<int64_t>>);
// static_assert(std::is_trivially_copyable_v<
//               Elementwise<std::negate<>, StridedVector<int64_t>>>);
// static_assert(Trivial<Elementwise<std::negate<>, StridedVector<int64_t>>>);
static_assert(AbstractVector<StridedVector<int64_t>>);
static_assert(AbstractVector<MutStridedVector<int64_t>>);
static_assert(std::is_trivially_copyable_v<StridedVector<int64_t>>);

template <class T, ptrdiff_t R = -1, ptrdiff_t C = -1, ptrdiff_t X = -1>
using PtrMatrix = Array<T, StridedDims<R, C, X>>;
template <class T, ptrdiff_t R = -1, ptrdiff_t C = -1, ptrdiff_t X = -1>
using MutPtrMatrix = MutArray<T, StridedDims<R, C, X>>;
template <class T, ptrdiff_t R = -1, ptrdiff_t C = -1>
using DensePtrMatrix = Array<T, DenseDims<R, C>>;
template <class T, ptrdiff_t R = -1, ptrdiff_t C = -1>
using MutDensePtrMatrix = MutArray<T, DenseDims<R, C>>;
template <class T> using SquarePtrMatrix = Array<T, SquareDims<>>;
template <class T> using MutSquarePtrMatrix = MutArray<T, SquareDims<>>;

// static_assert(AbstractMatrix<Elementwise<std::negate<>,
// PtrMatrix<int64_t>>>);
static_assert(utils::ElementOf<int, DensePtrMatrix<int64_t>>);
static_assert(utils::ElementOf<int64_t, DensePtrMatrix<int64_t>>);
static_assert(utils::ElementOf<int64_t, DensePtrMatrix<double>>);
static_assert(
  !utils::ElementOf<DensePtrMatrix<double>, DensePtrMatrix<double>>);
// static_assert(HasConcreteSize<DensePtrMatrix<int64_t>>);
static_assert(sizeof(PtrMatrix<int64_t>) ==
              3 * sizeof(ptrdiff_t) + sizeof(int64_t *));
static_assert(sizeof(MutPtrMatrix<int64_t>) ==
              3 * sizeof(ptrdiff_t) + sizeof(int64_t *));
static_assert(sizeof(DensePtrMatrix<int64_t>) ==
              2 * sizeof(ptrdiff_t) + sizeof(int64_t *));
static_assert(sizeof(MutDensePtrMatrix<int64_t>) ==
              2 * sizeof(ptrdiff_t) + sizeof(int64_t *));
static_assert(sizeof(SquarePtrMatrix<int64_t>) ==
              sizeof(ptrdiff_t) + sizeof(int64_t *));
static_assert(sizeof(MutSquarePtrMatrix<int64_t>) ==
              sizeof(ptrdiff_t) + sizeof(int64_t *));
static_assert(std::is_trivially_copyable_v<PtrMatrix<int64_t>>,
              "PtrMatrix<int64_t> is not trivially copyable!");
static_assert(std::is_trivially_copyable_v<PtrVector<int64_t>>,
              "PtrVector<int64_t,0> is not trivially copyable!");
// static_assert(std::is_trivially_copyable_v<MutPtrMatrix<int64_t>>,
//               "MutPtrMatrix<int64_t> is not trivially copyable!");

static_assert(!AbstractVector<PtrMatrix<int64_t>>,
              "PtrMatrix<int64_t> isa AbstractVector succeeded");
static_assert(!AbstractVector<MutPtrMatrix<int64_t>>,
              "PtrMatrix<int64_t> isa AbstractVector succeeded");
static_assert(!AbstractVector<const PtrMatrix<int64_t>>,
              "PtrMatrix<int64_t> isa AbstractVector succeeded");

static_assert(AbstractMatrix<PtrMatrix<int64_t>>,
              "PtrMatrix<int64_t> isa AbstractMatrix failed");
static_assert(
  std::same_as<decltype(PtrMatrix<int64_t>(nullptr, row(0),
                                           col(0))[ptrdiff_t(0), ptrdiff_t(0)]),
               const int64_t &>);
static_assert(
  std::same_as<std::remove_reference_t<decltype(MutPtrMatrix<int64_t>(
                 nullptr, row(0), col(0))[ptrdiff_t(0), ptrdiff_t(0)])>,
               int64_t>);

static_assert(AbstractMatrix<MutPtrMatrix<int64_t>>,
              "PtrMatrix<int64_t> isa AbstractMatrix failed");
static_assert(AbstractMatrix<const PtrMatrix<int64_t>>,
              "PtrMatrix<int64_t> isa AbstractMatrix failed");
static_assert(AbstractMatrix<const MutPtrMatrix<int64_t>>,
              "PtrMatrix<int64_t> isa AbstractMatrix failed");

static_assert(AbstractVector<MutPtrVector<int64_t>>,
              "PtrVector<int64_t> isa AbstractVector failed");
static_assert(AbstractVector<PtrVector<int64_t>>,
              "PtrVector<const int64_t> isa AbstractVector failed");
static_assert(AbstractVector<const PtrVector<int64_t>>,
              "PtrVector<const int64_t> isa AbstractVector failed");
static_assert(AbstractVector<const MutPtrVector<int64_t>>,
              "PtrVector<const int64_t> isa AbstractVector failed");

static_assert(!AbstractMatrix<MutPtrVector<int64_t>>,
              "PtrVector<int64_t> isa AbstractMatrix succeeded");
static_assert(!AbstractMatrix<PtrVector<int64_t>>,
              "PtrVector<const int64_t> isa AbstractMatrix succeeded");
static_assert(!AbstractMatrix<const PtrVector<int64_t>>,
              "PtrVector<const int64_t> isa AbstractMatrix succeeded");
static_assert(!AbstractMatrix<const MutPtrVector<int64_t>>,
              "PtrVector<const int64_t> isa AbstractMatrix succeeded");

template <class S> using IntArray = Array<int64_t, S>;

static_assert(std::convertible_to<Array<int64_t, SquareDims<>>,
                                  Array<int64_t, StridedDims<>>>);

static_assert(std::same_as<const int64_t &,
                           decltype(std::declval<PtrMatrix<int64_t>>()[0, 0])>);
static_assert(std::is_trivially_copyable_v<MutArray<int64_t, Length<>>>);

constexpr void swap(MutPtrMatrix<int64_t> A, Row<> i, Row<> j) {
  if (i == j) return;
  Col N = A.numCol();
  invariant((i < A.numRow()) && (j < A.numRow()));
  for (ptrdiff_t n = 0; n < N; ++n)
    std::swap(A[ptrdiff_t(i), n], A[ptrdiff_t(j), n]);
}
constexpr void swap(MutPtrMatrix<int64_t> A, Col<> i, Col<> j) {
  if (i == j) return;
  Row M = A.numRow();
  invariant((i < A.numCol()) && (j < A.numCol()));
  for (ptrdiff_t m = 0; m < M; ++m)
    std::swap(A[m, ptrdiff_t(i)], A[m, ptrdiff_t(j)]);
}
// static_assert(
//   AbstractMatrix<MatMatMul<PtrMatrix<int64_t>, PtrMatrix<int64_t>>>);

static_assert(std::copy_constructible<PtrMatrix<int64_t>>);
// static_assert(std::is_trivially_copyable_v<MutPtrMatrix<int64_t>>);
static_assert(std::is_trivially_copyable_v<PtrMatrix<int64_t>>);
static_assert(utils::TriviallyCopyable<PtrMatrix<int64_t>>);
// static_assert(
//   Trivial<ElementwiseBinaryOp<PtrMatrix<int64_t>, int, std::multiplies<>>>);
// static_assert(Trivial<MatMatMul<PtrMatrix<int64_t>, PtrMatrix<int64_t>>>);
// static_assert(AbstractMatrix<ElementwiseBinaryOp<PtrMatrix<int64_t>, int,
//                                                  std::multiplies<>>>,
//               "ElementwiseBinaryOp isa AbstractMatrix failed");

// static_assert(
//   !AbstractVector<MatMatMul<PtrMatrix<int64_t>, PtrMatrix<int64_t>>>,
//   "MatMul should not be an AbstractVector!");
// static_assert(AbstractMatrix<MatMatMul<PtrMatrix<int64_t>,
// PtrMatrix<int64_t>>>,
//               "MatMul is not an AbstractMatrix!");
static_assert(AbstractMatrix<Transpose<int64_t, PtrMatrix<int64_t>>>);
static_assert(ColVector<StridedVector<int64_t>>);
static_assert(
  AbstractVector<decltype(-std::declval<StridedVector<int64_t>>())>);
static_assert(ColVector<decltype(-std::declval<StridedVector<int64_t>>() * 0)>);

static_assert(RowVector<math::Array<double, math::Length<-1, long>, false>>);
static_assert(
  ColVector<math::MutArray<double, math::StridedRange<-1, -1>, false>>);
// static_assert(!std::common_with<math::MutArray<double, math::StridedRange<-1,
// -1>, false>, double>);

// template <typename T> constexpr auto countNonZero(PtrMatrix<T> x) ->
// ptrdiff_t {
//   ptrdiff_t count = 0;
//   for (ptrdiff_t r = 0; r < x.numRow(); ++r) count += countNonZero(x[r, _]);
//   return count;
// }

template <typename T, typename S> struct ScalarizeViaCast<Array<T, S, true>> {
  using type = scalarize_elt_cast_t<utils::compressed_t<T>>;
};
template <typename T, typename S>
struct ScalarizeViaCast<MutArray<T, S, true>> {
  using type = scalarize_elt_cast_t<utils::compressed_t<T>>;
};
} // namespace math

template <class T, math::Dimension S, bool Compress>
inline constexpr bool
  std::ranges::enable_borrowed_range<math::Array<T, S, Compress>> = true;
template <class T, math::Dimension S, bool Compress>
inline constexpr bool
  std::ranges::enable_borrowed_range<math::MutArray<T, S, Compress>> = true;
static_assert(
  std::same_as<utils::eltype_t<math::Array<unsigned, math::Length<>>>,
               unsigned>);

template <class T, ptrdiff_t L>
requires(L != -1) // NOLINTNEXTLINE(cert-dcl58-cpp)
struct std::tuple_size<::math::Array<T, ::math::Length<L>>>
  : std::integral_constant<ptrdiff_t, L> {};

template <size_t I, class T, ptrdiff_t L>
requires(L != -1) // NOLINTNEXTLINE(cert-dcl58-cpp)
struct std::tuple_element<I, ::math::Array<T, ::math::Length<L>>> {
  using type = T;
};

template <class T, ptrdiff_t L,
          ptrdiff_t X>
requires(L != -1) // NOLINTNEXTLINE(cert-dcl58-cpp)
struct std::tuple_size<::math::Array<T, ::math::StridedRange<L, X>>>
  : std::integral_constant<ptrdiff_t, L> {};

template <size_t I, class T, ptrdiff_t L,
          ptrdiff_t X>
requires(L != -1) // NOLINTNEXTLINE(cert-dcl58-cpp)
struct std::tuple_element<I, ::math::Array<T, ::math::StridedRange<L, X>>> {
  using type = T;
};
template <class T, ptrdiff_t L,
          ptrdiff_t X>
requires(L != -1) // NOLINTNEXTLINE(cert-dcl58-cpp)
struct std::tuple_size<::math::MutArray<T, ::math::StridedRange<L, X>>>
  : std::integral_constant<ptrdiff_t, L> {};

template <size_t I, class T, ptrdiff_t L,
          ptrdiff_t X>
requires(L != -1) // NOLINTNEXTLINE(cert-dcl58-cpp)
struct std::tuple_element<I, ::math::MutArray<T, ::math::StridedRange<L, X>>> {
  using type = T;
};
