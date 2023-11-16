#pragma once

#include "Alloc/Arena.hpp"
#include "Containers/Storage.hpp"
#include "Math/ArrayOps.hpp"
#include "Math/AxisTypes.hpp"
#include "Math/Indexing.hpp"
#include "Math/Iterators.hpp"
#include "Math/Matrix.hpp"
#include "Math/MatrixDimensions.hpp"
#include "Math/Rational.hpp"
#include "Utilities/Invariant.hpp"
#include "Utilities/Optional.hpp"
#include "Utilities/TypeCompression.hpp"
#include "Utilities/TypePromotion.hpp"
#include "Utilities/Valid.hpp"
#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>
#include <version>

// https://llvm.org/doxygen/Compiler_8h_source.html#l00307
#ifndef POLY_MATH_HAS_CPP_ATTRIBUTE
#if defined(__cplusplus) && defined(__has_cpp_attribute)
#define POLY_MATH_HAS_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
#define POLY_MATH_HAS_CPP_ATTRIBUTE(x) 0
#endif
#endif
#if POLY_MATH_HAS_CPP_ATTRIBUTE(gsl::Owner)
#define POLY_MATH_GSL_OWNER [[gsl::Owner]]
#else
#define POLY_MATH_GSL_OWNER
#endif
/// POLY_MATH_GSL_POINTER - Apply this to non-owning classes like
/// StringRef to enable lifetime warnings.
#if POLY_MATH_HAS_CPP_ATTRIBUTE(gsl::Pointer)
#define POLY_MATH_GSL_POINTER [[gsl::Pointer]]
#else
#define POLY_MATH_GSL_POINTER
#endif

namespace poly::math {
template <typename S>
concept Dimension = VectorDimension<S> != MatrixDimension<S>;
template <class T, Dimension S,
          ptrdiff_t N = containers::PreAllocStorage<T, S>(),
          class A = alloc::Mallocator<T>>
struct ManagedArray;

template <typename T>
concept Printable = requires(std::ostream &os, T x) {
  { os << x } -> std::convertible_to<std::ostream &>;
};
static_assert(Printable<int64_t>);
void print_obj(std::ostream &os, Printable auto x) { os << x; };
template <typename F, typename S>
void print_obj(std::ostream &os, const std::pair<F, S> &x) {
  os << "(" << x.first << ", " << x.second << ")";
};
using utils::Valid, utils::Optional;

template <class T, Dimension S> struct Array;
template <class T, Dimension S> struct MutArray;

// Cases we need to consider:
// 1. Slice-indexing
// 2.a. `ptrdiff_t` indexing, not compressed
// 2.b. `ptrdiff_t` indexing, compressed
// 3.a.i. Vector indexing, contig, no mask
// 3.a.ii. Vector indexing, contig, mask
// 3.b.i. Vector indexing, discontig, no mask
// 3.b.ii. Vector indexing, discontig, mask
// all of the above for `T*` and `const T*`
template <typename T, typename S, typename I>
[[gnu::flatten, gnu::always_inline]] constexpr auto index(const T *ptr, S shape,
                                                          I i) noexcept
  -> decltype(auto) {
  auto offset = calcOffset(shape, i);
  auto newDim = calcNewDim(shape, i);
  invariant(ptr != nullptr);
  if constexpr (std::same_as<decltype(newDim), Empty>)
    if constexpr (utils::Compressible<T>) return decompress(ptr + offset);
    else return ptr[offset];
  else if constexpr (simd::index::issimd<decltype(newDim)>)
    return simd::ref(ptr + offset, newDim);
  else return Array<T, decltype(newDim)>{ptr + offset, newDim};
}
// for (row/col)vectors, we drop the row/col, essentially broadcasting
template <typename T, typename S, typename R, typename C>
[[gnu::flatten, gnu::always_inline]] constexpr auto index(const T *ptr, S shape,
                                                          R r, C c) noexcept
  -> decltype(auto) {
  if constexpr (MatrixDimension<S>) {
    auto offset = calcOffset(shape, unwrapRow(r), unwrapCol(c));
    auto newDim = calcNewDim(shape, unwrapRow(r), unwrapCol(c));
    if constexpr (std::same_as<decltype(newDim), Empty>)
      // 3.a.i. Vector indexing, contig, no mask
      // 3.a.ii. Vector indexing, contig, mask
      if constexpr (utils::Compressible<T>)
        return utils::decompress(ptr + offset);
      else return ptr[offset];
    else if constexpr (simd::index::issimd<decltype(newDim)>)
      return simd::ref(ptr + offset, newDim);
    else return Array<T, decltype(newDim)>{ptr + offset, newDim};
  } else if constexpr (std::same_as<S, StridedRange>)
    return index(ptr, shape, r);
  else return index(ptr, shape, c);
}

template <typename T, typename S, typename I>
[[gnu::flatten, gnu::always_inline]] constexpr auto index(T *ptr, S shape,
                                                          I i) noexcept
  -> decltype(auto) {
  auto offset = calcOffset(shape, i);
  auto newDim = calcNewDim(shape, i);
  invariant(ptr != nullptr);
  if constexpr (std::same_as<decltype(newDim), Empty>)
    if constexpr (utils::Compressible<T>) return ref(ptr + offset);
    else return ptr[offset];
  else if constexpr (simd::index::issimd<decltype(newDim)>)
    return simd::ref(ptr + offset, newDim);
  else return MutArray<T, decltype(newDim)>{ptr + offset, newDim};
}
// for (row/col)vectors, we drop the row/col, essentially broadcasting
template <typename T, typename S, typename R, typename C>
[[gnu::flatten, gnu::always_inline]] constexpr auto index(T *ptr, S shape, R r,
                                                          C c) noexcept
  -> decltype(auto) {
  if constexpr (MatrixDimension<S>) {
    auto offset = calcOffset(shape, unwrapRow(r), unwrapCol(c));
    auto newDim = calcNewDim(shape, unwrapRow(r), unwrapCol(c));
    if constexpr (std::same_as<decltype(newDim), Empty>)
      // 3.a.i. Vector indexing, contig, no mask
      // 3.a.ii. Vector indexing, contig, mask
      if constexpr (utils::Compressible<T>) return ref(ptr + offset);
      else return ptr[offset];
    else if constexpr (simd::index::issimd<decltype(newDim)>)
      return simd::ref(ptr + offset, newDim);
    else return MutArray<T, decltype(newDim)>{ptr + offset, newDim};
  } else if constexpr (std::same_as<S, StridedRange>)
    return index(ptr, shape, r);
  else return index(ptr, shape, c);
}

template <typename T, bool Column = false> struct SliceIterator {
  using stride_type = std::conditional_t<Column, StridedRange, ptrdiff_t>;
  using value_type =
    std::conditional_t<std::is_const_v<T>,
                       Array<std::remove_cvref_t<T>, stride_type>,
                       MutArray<std::remove_reference_t<T>, stride_type>>;
  T *data;
  ptrdiff_t len;
  ptrdiff_t rowStride;
  ptrdiff_t idx;
  // constexpr auto operator*() -> value_type;
  constexpr auto operator++() -> SliceIterator & {
    idx++;
    return *this;
  }
  constexpr auto operator++(int) -> SliceIterator {
    SliceIterator ret{*this};
    ++(*this);
    return ret;
  }
  constexpr auto operator--() -> SliceIterator & {
    idx--;
    return *this;
  }
  constexpr auto operator--(int) -> SliceIterator {
    SliceIterator ret{*this};
    --(*this);
    return ret;
  }
};
template <typename T, bool Column>
constexpr auto operator-(SliceIterator<T, Column> a, SliceIterator<T, Column> b)
  -> ptrdiff_t {
  return a.idx - b.idx;
}
template <typename T, bool Column>
constexpr auto operator+(SliceIterator<T, Column> a, ptrdiff_t i)
  -> SliceIterator<T, Column> {
  return {a.data, a.len, a.rowStride, a.idx + i};
}
template <typename T>
constexpr auto operator==(SliceIterator<T> a, SliceIterator<T> b) -> bool {
  return a.idx == b.idx;
}
template <typename T>
constexpr auto operator<=>(SliceIterator<T> a, SliceIterator<T> b)
  -> std::strong_ordering {
  return a.idx <=> b.idx;
}
template <typename T>
constexpr auto operator==(SliceIterator<T, false> a, Row<> r) -> bool {
  return a.idx == r;
}
template <typename T>
constexpr auto operator<=>(SliceIterator<T, false> a, Row<> r)
  -> std::strong_ordering {
  return a.idx <=> r;
}
template <typename T>
constexpr auto operator==(SliceIterator<T, true> a, Col<> r) -> bool {
  return a.idx == r;
}
template <typename T>
constexpr auto operator<=>(SliceIterator<T, true> a, Col<> r)
  -> std::strong_ordering {
  return a.idx <=> r;
}

template <typename T, bool Column = false> struct SliceRange {
  T *data;
  ptrdiff_t len;
  ptrdiff_t rowStride;
  ptrdiff_t stop;
  [[nodiscard]] constexpr auto begin() const -> SliceIterator<T, Column> {
    return {data, len, rowStride, 0};
  }
  [[nodiscard]] constexpr auto end() const {
    if constexpr (Column) return Col<>{stop};
    else return Row<>{stop};
  }
};
/// Constant Array
template <class T, Dimension S> struct POLY_MATH_GSL_POINTER Array {
  static_assert(!std::is_const_v<T>, "T shouldn't be const");
  static_assert(std::is_trivially_destructible_v<T>,
                "maybe should add support for destroying");
  using value_type = T;
  using reference = T &;
  using const_reference = const T &;
  using size_type = ptrdiff_t;
  using difference_type = int;
  using iterator = T *;
  using const_iterator = const T *;
  using pointer = T *;
  using const_pointer = const T *;
  using concrete = std::true_type;

  static constexpr bool isdense =
    std::convertible_to<S, ptrdiff_t> || std::convertible_to<S, DenseDims<>>;
  static constexpr bool flatstride = isdense || std::same_as<S, StridedRange>;
  static_assert(flatstride != std::same_as<S, StridedDims<>>);

  constexpr Array() = default;
  constexpr Array(const Array &) = default;
  constexpr Array(Array &&) noexcept = default;
  constexpr auto operator=(const Array &) -> Array & = default;
  constexpr auto operator=(Array &&) noexcept -> Array & = default;
  constexpr Array(const T *p, S s) : ptr(p), sz(s) {}
  constexpr Array(Valid<const T> p, S s) : ptr(p), sz(s) {}
  template <ptrdiff_t R, ptrdiff_t C>
  constexpr Array(const T *p, Row<R> r, Col<C> c) : ptr(p), sz(S{r, c}) {}
  template <ptrdiff_t R, ptrdiff_t C>
  constexpr Array(Valid<const T> p, Row<R> r, Col<C> c)
    : ptr(p), sz(dimension<S>(r, c)) {}
  template <std::convertible_to<S> V>
  constexpr Array(Array<T, V> a) : ptr(a.data()), sz(a.dim()) {}
  template <size_t N>
  constexpr Array(const std::array<T, N> &a) : ptr(a.data()), sz(N) {}
  [[nodiscard, gnu::returns_nonnull]] constexpr auto data() const noexcept
    -> const T * {
    invariant(ptr != nullptr);
    return ptr;
  }
  [[nodiscard]] constexpr auto wrappedPtr() noexcept -> Valid<T> { return ptr; }

  [[nodiscard]] constexpr auto begin() const noexcept
    -> StridedIterator<const T>
  requires(std::is_same_v<S, StridedRange>)
  {
    const T *p = ptr;
    return StridedIterator{p, sz.stride};
  }
  [[nodiscard]] constexpr auto begin() const noexcept
    -> const T *requires(isdense) { return ptr; }

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
    else return ptr[sride(sz) * ptrdiff_t(row(sz)) - 1];
  }
  // indexing has two components:
  // 1. offsetting the pointer
  // 2. calculating new dim
  // static constexpr auto slice(Valid<T>, Index<S> auto i){
  //   auto
  // }
  [[gnu::flatten, gnu::always_inline]] constexpr auto
  operator[](Index<S> auto i) const noexcept -> decltype(auto) {
    return index(ptr, sz, i);
  }
  // for (row/col)vectors, we drop the row/col, essentially broadcasting
  template <class R, class C>
  [[gnu::flatten, gnu::always_inline]] constexpr auto
  operator[](R r, C c) const noexcept -> decltype(auto) {
    return index(ptr, sz, r, c);
  }
  [[nodiscard]] constexpr auto minRowCol() const -> ptrdiff_t {
    return std::min(ptrdiff_t(numRow()), ptrdiff_t(numCol()));
  }

  [[nodiscard]] constexpr auto diag() const noexcept {
    StridedRange r{minRowCol(), ptrdiff_t(RowStride(sz)) + 1};
    invariant(ptr != nullptr);
    return Array<T, StridedRange>{ptr, r};
  }
  [[nodiscard]] constexpr auto antiDiag() const noexcept {
    StridedRange r{minRowCol(), ptrdiff_t(RowStride(sz)) - 1};
    invariant(ptr != nullptr);
    return Array<T, StridedRange>{ptr + ptrdiff_t(Col(sz)) - 1, r};
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
    if constexpr (std::convertible_to<S, ptrdiff_t>) return Row<1>{};
    else return row(sz);
  }
  [[nodiscard]] constexpr auto numCol() const noexcept { return col(sz); }
  [[nodiscard]] constexpr auto rowStride() const noexcept {
    if constexpr (std::integral<S>) return RowStride<1>{};
    else return stride(sz);
  }
  [[nodiscard]] constexpr auto empty() const -> bool { return sz == S{}; }
  [[nodiscard]] constexpr auto size() const noexcept {
    if constexpr (StaticInt<S>) return S{};
    else return ptrdiff_t(sz);
  }
  [[nodiscard]] constexpr auto dim() const noexcept -> S { return sz; }
  constexpr void clear() { sz = S{}; }
  [[nodiscard]] constexpr auto t() const { return Transpose{*this}; }
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
    invariant(ptr != nullptr);
    return Array<T, S>{ptr, this->sz};
  }

  [[nodiscard]] constexpr auto deleteCol(ptrdiff_t c) const
    -> ManagedArray<T, S> {
    static_assert(MatrixDimension<S>);
    auto newDim = dim().similar(numRow() - 1);
    ManagedArray<T, decltype(newDim)> A(newDim);
    for (ptrdiff_t m = 0; m < numRow(); ++m) {
      A[m, _(0, c)] = (*this)[m, _(0, c)];
      A[m, _(c, math::end)] = (*this)[m, _(c + 1, math::end)];
    }
    return A;
  }
  [[nodiscard]] constexpr auto operator==(const Array &other) const noexcept
    -> bool {
    if (size() != other.size()) return false;
    if constexpr (MatrixDimension<S> && !DenseLayout<S>) {
      // may not be dense, iterate over rows
      for (ptrdiff_t i = 0; i < numRow(); ++i)
        if ((*this)[i, _] != other[i, _]) return false;
      return true;
    } else return std::equal(begin(), end(), other.begin());
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
    static_assert(DenseLayout<S>);
    value_type ret{0};
    for (auto x : *this) ret += x;
    return ret;
    // return std::reduce(begin(), end());
  }
  friend inline void PrintTo(const Array &x, ::std::ostream *os) { *os << x; }
#ifndef NDEBUG
  [[gnu::used]] void dump() const {
    if constexpr (Printable<T>) std::cout << "Size: " << sz << *this << "\n";
  }
  [[gnu::used]] void dump(const char *filename) const {
    if constexpr (std::integral<T>) {
      std::FILE *f = std::fopen(filename, "w");
      if (f == nullptr) return;
      (void)std::fprintf(f, "C= [");
      if constexpr (MatrixDimension<S>) {
        for (ptrdiff_t i = 0; i < Row(sz); ++i) {
          if (i) (void)std::fprintf(f, "\n");
          (void)std::fprintf(f, "%ld", int64_t((*this)[i, 0]));
          for (ptrdiff_t j = 1; j < Col(sz); ++j)
            (void)std::fprintf(f, " %ld", int64_t((*this)[i, j]));
        }
      } else {
        (void)std::fprintf(f, "%ld", int64_t((*this)[0]));
        for (ptrdiff_t i = 1; (i < ptrdiff_t(sz)); ++i)
          (void)std::fprintf(f, ", %ld", int64_t((*this)[i]));
      }
      (void)std::fprintf(f, "]");
      (void)std::fclose(f);
    }
  }
#endif
protected:
  // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
  const T *ptr;
  // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
  [[no_unique_address]] S sz{};
};

template <class T, DenseLayout S>
[[nodiscard]] constexpr auto operator<=>(Array<T, S> x, Array<T, S> y) {
  ptrdiff_t M = x.size();
  ptrdiff_t N = y.size();
  for (ptrdiff_t i = 0, L = std::min(M, N); i < L; ++i)
    if (auto cmp = x[i] <=> y[i]; cmp != 0) return cmp;
  return M <=> N;
};

template <class T, Dimension S>
struct POLY_MATH_GSL_POINTER MutArray : Array<T, S>,
                                        ArrayOps<T, S, MutArray<T, S>> {
  using BaseT = Array<T, S>;
  // using BaseT::BaseT;
  using BaseT::operator[], BaseT::data, BaseT::begin, BaseT::end, BaseT::rbegin,
    BaseT::rend, BaseT::front, BaseT::back;

  constexpr MutArray(const MutArray &) = default;
  constexpr MutArray(MutArray &&) noexcept = default;
  constexpr auto operator=(const MutArray &) -> MutArray & = delete;
  // constexpr auto operator=(const MutArray &) -> MutArray & = default;
  constexpr auto operator=(MutArray &&) noexcept -> MutArray & = default;
  constexpr MutArray(T *p, S s) : BaseT(p, s) {}

  constexpr void truncate(S nz) {
    S oz = this->sz;
    this->sz = nz;
    if constexpr (std::integral<S>) {
      invariant(ptrdiff_t(nz) <= ptrdiff_t(oz));
    } else if constexpr (std::convertible_to<S, DenseDims<>>) {
      auto newX = ptrdiff_t{RowStride(nz)}, oldX = ptrdiff_t{RowStride(oz)},
           newN = ptrdiff_t{Col(nz)}, oldN = ptrdiff_t{Col(oz)},
           newM = ptrdiff_t{Row(nz)}, oldM = ptrdiff_t{Row(oz)};
      invariant(newM <= oldM);
      invariant(newN <= oldN);
      invariant(newX <= oldX);
      ptrdiff_t colsToCopy = newN;
      // we only need to copy if memory shifts position
      bool copyCols = ((colsToCopy > 0) && (newX != oldX));
      // if we're in place, we have 1 less row to copy
      ptrdiff_t rowsToCopy = newM;
      if (rowsToCopy && (--rowsToCopy) && (copyCols)) {
        // truncation, we need to copy rows to increase stride
        T *src = data(), *dst = src;
        do {
          src += oldX;
          dst += newX;
          std::copy_n(src, colsToCopy, dst);
        } while (--rowsToCopy);
      }
    } else {
      static_assert(MatrixDimension<S>, "Can only resize 1 or 2d containers.");
      invariant(nz.row() <= oz.row());
      invariant(nz.col() <= oz.col());
    }
  }

  constexpr void truncate(Row<> r) {
    if constexpr (std::integral<S>) {
      return truncate(S(r));
    } else if constexpr (std::convertible_to<S, DenseDims<>>) {
      static_assert(!std::convertible_to<S, SquareDims<>>,
                    "if truncating a row, matrix must be strided or dense.");
      invariant(r <= Row(this->sz));
      DenseDims newSz = this->sz;
      truncate(newSz.set(r));
    } else {
      static_assert(std::convertible_to<S, StridedDims<>>);
      invariant(r <= Row(this->sz));
      this->sz.set(r);
    }
  }
  constexpr void truncate(Col<> c) {
    if constexpr (std::integral<S>) {
      return truncate(S(c));
    } else if constexpr (std::is_same_v<S, DenseDims<>>) {
      static_assert(!std::convertible_to<S, SquareDims<>>,
                    "if truncating a col, matrix must be strided or dense.");
      invariant(c <= Col(this->sz));
      DenseDims newSz = this->sz;
      truncate(newSz.set(c));
    } else {
      static_assert(std::convertible_to<S, StridedDims<>>);
      invariant(c <= Col(this->sz));
      this->sz.set(c);
    }
  }

  template <class... Args>
  constexpr MutArray(Args &&...args)
    : Array<T, S>(std::forward<Args>(args)...) {}

  template <std::convertible_to<T> U, std::convertible_to<S> V>
  constexpr MutArray(Array<U, V> a) : Array<T, S>(a) {}
  template <size_t N>
  constexpr MutArray(std::array<T, N> &a) : Array<T, S>(a.data(), N) {}
  [[nodiscard, gnu::returns_nonnull]] constexpr auto data() noexcept -> T * {
    invariant(this->ptr != nullptr);
    return const_cast<T *>(this->ptr);
  }
  [[nodiscard]] constexpr auto wrappedPtr() noexcept -> Valid<T> {
    return data();
  }

  [[nodiscard]] constexpr auto begin() noexcept -> StridedIterator<T>
  requires(std::is_same_v<S, StridedRange>)
  {
    return StridedIterator{const_cast<T *>(this->ptr), this->sz.stride};
  }
  [[nodiscard]] constexpr auto begin() noexcept
    -> T *requires(!std::is_same_v<S, StridedRange>) {
      return const_cast<T *>(this->ptr);
    }

  [[nodiscard]] constexpr auto end() noexcept {
    return begin() + ptrdiff_t(this->sz);
  }
  // [[nodiscard, gnu::returns_nonnull]] constexpr auto begin() noexcept -> T
  // *
  // {
  //   return this->ptr;
  // }
  // [[nodiscard, gnu::returns_nonnull]] constexpr auto end() noexcept -> T *
  // {
  //   return this->ptr + ptrdiff_t(this->sz);
  // }
  [[nodiscard]] constexpr auto rbegin() noexcept {
    return std::reverse_iterator(end());
  }
  [[nodiscard]] constexpr auto rend() noexcept {
    return std::reverse_iterator(begin());
  }
  constexpr auto front() noexcept -> T & { return *begin(); }
  constexpr auto back() noexcept -> T & { return *(end() - 1); }
  [[gnu::flatten, gnu::always_inline]] constexpr auto
  operator[](Index<S> auto i) noexcept -> decltype(auto) {
    return index(data(), this->sz, i);
  }
  // TODO: switch to operator[] when we enable c++23
  template <class R, class C>
  [[gnu::flatten, gnu::always_inline]] constexpr auto operator[](R r,
                                                                 C c) noexcept
    -> decltype(auto) {
    return index(data(), this->sz, r, c);
  }
  constexpr void fill(T value) {
    std::fill_n(this->data(), ptrdiff_t(this->dim()), value);
  }
  [[nodiscard]] constexpr auto diag() noexcept {
    StridedRange r{std::min(ptrdiff_t(Row(this->sz)), ptrdiff_t(Col(this->sz))),
                   ptrdiff_t(RowStride(this->sz)) + 1};
    return MutArray<T, StridedRange>{data(), r};
  }
  [[nodiscard]] constexpr auto antiDiag() noexcept {
    Col<> c = Col(this->sz);
    StridedRange r{ptrdiff_t(std::min(ptrdiff_t(Row(this->sz)), ptrdiff_t(c))),
                   ptrdiff_t(RowStride(this->sz)) - 1};
    return MutArray<T, StridedRange>{data() + ptrdiff_t(c) - 1, r};
  }
  constexpr void erase(S i) {
    static_assert(std::integral<S>, "erase requires integral size");
    S oldLen = this->sz--;
    if (i < this->sz)
      std::copy(this->data() + i + 1, data() + oldLen, this->data() + i);
  }
  constexpr void erase(Row<> r) {
    if constexpr (std::integral<S>) {
      return erase(S(r));
    } else if constexpr (std::convertible_to<S, DenseDims<>>) {
      static_assert(!std::convertible_to<S, SquareDims<>>,
                    "if erasing a row, matrix must be strided or dense.");
      auto col = ptrdiff_t{Col(this->sz)},
           newRow = ptrdiff_t{Row(this->sz)} - 1;
      this->sz.set(Row<>{newRow});
      if ((col == 0) || (r == newRow)) return;
      T *dst = data() + ptrdiff_t(r) * col;
      std::copy_n(dst + col, (newRow - ptrdiff_t(r)) * col, dst);
    } else {
      static_assert(std::convertible_to<S, StridedDims<>>);
      auto stride = ptrdiff_t{RowStride(this->sz)},
           col = ptrdiff_t{Col(this->sz)},
           newRow = ptrdiff_t{Row(this->sz)} - 1;
      this->sz.set(Row<>{newRow});
      if ((col == 0) || (r == newRow)) return;
      invariant(col <= stride);
      if ((col + (512 / (sizeof(T)))) <= stride) {
        T *dst = data() + ptrdiff_t(r) * stride;
        for (ptrdiff_t m = ptrdiff_t(r); m < newRow; ++m) {
          T *src = dst + stride;
          std::copy_n(src, col, dst);
          dst = src;
        }
      } else {
        T *dst = data() + ptrdiff_t(r) * stride;
        std::copy_n(dst + stride, (newRow - ptrdiff_t(r)) * stride, dst);
      }
    }
  }
  constexpr void erase(Col<> c) {
    if constexpr (std::integral<S>) {
      return erase(S(c));
    } else if constexpr (std::convertible_to<S, DenseDims<>>) {
      static_assert(!std::convertible_to<S, SquareDims<>>,
                    "if erasing a col, matrix must be strided or dense.");
      auto newCol = ptrdiff_t{Col(this->sz)}, oldCol = newCol--,
           row = ptrdiff_t{Row(this->sz)};
      this->sz.set(Col<>{newCol});
      ptrdiff_t colsToCopy = newCol - ptrdiff_t(c);
      if ((colsToCopy == 0) || (row == 0)) return;
      // we only need to copy if memory shifts position
      for (ptrdiff_t m = 0; m < row; ++m) {
        T *dst = data() + m * newCol + ptrdiff_t(c);
        T *src = data() + m * oldCol + ptrdiff_t(c) + 1;
        std::copy_n(src, colsToCopy, dst);
      }
    } else {
      static_assert(std::convertible_to<S, StridedDims<>>);
      auto stride = ptrdiff_t{RowStride(this->sz)},
           newCol = ptrdiff_t{Col(this->sz)} - 1,
           row = ptrdiff_t{Row(this->sz)};
      this->sz.set(Col<>{newCol});
      ptrdiff_t colsToCopy = newCol - ptrdiff_t(c);
      if ((colsToCopy == 0) || (row == 0)) return;
      // we only need to copy if memory shifts position
      for (ptrdiff_t m = 0; m < row; ++m) {
        T *dst = data() + m * stride + ptrdiff_t(c);
        std::copy_n(dst + 1, colsToCopy, dst);
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
  constexpr auto eachRow() -> SliceRange<T, false>
  requires(MatrixDimension<S>)
  {
    return {data(), ptrdiff_t(Col(this->sz)), ptrdiff_t(RowStride(this->sz)),
            ptrdiff_t(Row(this->sz))};
  }
  constexpr auto eachCol() -> SliceRange<T, true>
  requires(MatrixDimension<S>)
  {
    return {data(), ptrdiff_t(Row(this->sz)), ptrdiff_t(RowStride(this->sz)),
            ptrdiff_t(Col(this->sz))};
  }
};

template <typename T, typename S> MutArray(T *, S) -> MutArray<T, S>;

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
static_assert(AbstractVector<Array<int64_t, ptrdiff_t>>);
static_assert(!AbstractVector<Array<int64_t, StridedDims<>>>);
static_assert(AbstractMatrix<Array<int64_t, StridedDims<>>>);
static_assert(RowVector<Array<int64_t, ptrdiff_t>>);
static_assert(ColVector<Transpose<Array<int64_t, ptrdiff_t>>>);
static_assert(ColVector<Array<int64_t, StridedRange>>);
static_assert(RowVector<Transpose<Array<int64_t, StridedRange>>>);

template <typename T>
auto operator*(SliceIterator<T, false> it)
  -> SliceIterator<T, false>::value_type {
  return {it.data + it.rowStride * it.idx, it.len};
}
template <typename T>
auto operator*(SliceIterator<T, true> it)
  -> SliceIterator<T, true>::value_type {
  return {it.data + it.idx, StridedRange{it.len, it.rowStride}};
}

static_assert(std::weakly_incrementable<SliceIterator<int64_t, false>>);
static_assert(std::forward_iterator<SliceIterator<int64_t, false>>);
static_assert(std::ranges::forward_range<SliceRange<int64_t, false>>);
static_assert(std::ranges::range<SliceRange<int64_t, false>>);

/// Non-owning view of a managed array, capable of resizing,
/// but not of re-allocating in case the capacity is exceeded.
template <class T, Dimension S>
struct POLY_MATH_GSL_POINTER ResizeableView : MutArray<T, S> {
  using BaseT = MutArray<T, S>;
  using U = containers::default_capacity_type_t<S>;
  constexpr ResizeableView() noexcept : BaseT(nullptr, 0), capacity(0) {}
  constexpr ResizeableView(T *p, S s, U c) noexcept
    : BaseT(p, s), capacity(c) {}
  constexpr ResizeableView(alloc::Arena<> *a, S s, U c) noexcept
    : ResizeableView{a->template allocate<T>(c), s, c} {}

  [[nodiscard]] constexpr auto isFull() const -> bool {
    return U(this->sz) == capacity;
  }

  template <class... Args>
  constexpr auto emplace_back(Args &&...args) -> decltype(auto) {
    static_assert(std::is_integral_v<S>, "emplace_back requires integral size");
    invariant(U(this->sz) < capacity);
    return *std::construct_at(this->data() + this->sz++,
                              std::forward<Args>(args)...);
  }
  /// Allocates extra space if needed
  /// Has a different name to make sure we avoid ambiguities.
  template <class... Args>
  constexpr auto emplace_backa(alloc::Arena<> *alloc, Args &&...args)
    -> decltype(auto) {
    static_assert(std::is_integral_v<S>, "emplace_back requires integral size");
    if (isFull()) reserve(alloc, (capacity + 1) * 2);
    return *std::construct_at(this->data() + this->sz++,
                              std::forward<Args>(args)...);
  }
  constexpr void push_back(T value) {
    static_assert(std::is_integral_v<S>, "push_back requires integral size");
    invariant(U(this->sz) < capacity);
    std::construct_at<T>(this->data() + this->sz++, std::move(value));
  }
  constexpr void push_back(alloc::Arena<> *alloc, T value) {
    static_assert(std::is_integral_v<S>, "push_back requires integral size");
    if (isFull()) reserve(alloc, (capacity + 1) * 2);
    std::construct_at<T>(this->data() + this->sz++, std::move(value));
  }
  constexpr void pop_back() {
    static_assert(std::is_integral_v<S>, "pop_back requires integral size");
    invariant(this->sz > 0);
    if constexpr (std::is_trivially_destructible_v<T>) --this->sz;
    else this->data()[--this->sz].~T();
  }
  constexpr auto pop_back_val() -> T {
    static_assert(std::is_integral_v<S>, "pop_back requires integral size");
    invariant(this->sz > 0);
    return std::move(this->data()[--this->sz]);
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
    if constexpr (std::integral<S>) {
      invariant(U(nz) <= capacity);
      if (nz > oz) std::fill(this->data() + oz, this->data() + nz, T{});
    } else {
      static_assert(MatrixDimension<S>, "Can only resize 1 or 2d containers.");
      auto newX = ptrdiff_t{RowStride(nz)}, oldX = ptrdiff_t{RowStride(oz)},
           newN = ptrdiff_t{Col(nz)}, oldN = ptrdiff_t{Col(oz)},
           newM = ptrdiff_t{Row(nz)}, oldM = ptrdiff_t{Row(oz)};
      invariant(U(nz) <= capacity);
      U len = U(nz);
      T *npt = this->data();
      // we can copy forward so long as the new stride is smaller
      // so that the start of the dst range is outside of the src range
      // we can also safely forward copy if we allocated a new ptr
      bool forwardCopy = (newX <= oldX);
      ptrdiff_t colsToCopy = std::min(oldN, newN);
      // we only need to copy if memory shifts position
      bool copyCols = ((colsToCopy > 0) && (newX != oldX));
      // if we're in place, we have 1 less row to copy
      ptrdiff_t rowsToCopy = std::min(oldM, newM) - 1;
      ptrdiff_t fillCount = newN - colsToCopy;
      if ((rowsToCopy) && (copyCols || fillCount)) {
        if (forwardCopy) {
          // truncation, we need to copy rows to increase stride
          T *src = this->data() + oldX;
          T *dst = npt + newX;
          do {
            if (copyCols) std::copy_n(src, colsToCopy, dst);
            if (fillCount) std::fill_n(dst + colsToCopy, fillCount, T{});
            src += oldX;
            dst += newX;
          } while (--rowsToCopy);
        } else /* [[unlikely]] */ {
          // backwards copy, only needed when we increasing stride but not
          // reallocating, which should be comparatively uncommon.
          // Should probably benchmark or determine actual frequency
          // before adding `[[unlikely]]`.
          T *src = this->data() + (rowsToCopy + 1) * oldX;
          T *dst = npt + (rowsToCopy + 1) * newX;
          do {
            src -= oldX;
            dst -= newX;
            if (colsToCopy)
              std::copy_backward(src, src + colsToCopy, dst + colsToCopy);
            if (fillCount) std::fill_n(dst + colsToCopy, fillCount, T{});
          } while (--rowsToCopy);
        }
      }
      // zero init remaining rows
      for (ptrdiff_t m = oldM; m < newM; ++m)
        std::fill_n(npt + m * newX, newN, T{});
    }
  }

  constexpr void resize(Row<> r) {
    if constexpr (std::integral<S>) {
      return resize(S(r));
    } else if constexpr (MatrixDimension<S>) {
      S nz = this->sz;
      return resize(nz.set(r));
    }
  }
  constexpr void resize(Col<> c) {
    if constexpr (std::integral<S>) {
      return resize(S(c));
    } else if constexpr (MatrixDimension<S>) {
      S nz = this->sz;
      return resize(nz.set(c));
    }
  }
  constexpr void resizeForOverwrite(S M) {
    invariant(U(M) <= U(this->sz));
    this->sz = M;
  }
  constexpr void resizeForOverwrite(Row<> r) {
    if constexpr (std::integral<S>) {
      return resizeForOverwrite(S(r));
    } else if constexpr (MatrixDimension<S>) {
      S nz = this->sz;
      return resizeForOverwrite(nz.set(r));
    }
  }
  constexpr void resizeForOverwrite(Col<> c) {
    if constexpr (std::integral<S>) {
      return resizeForOverwrite(S(c));
    } else if constexpr (MatrixDimension<S>) {
      S nz = this->sz;
      return resizeForOverwrite(nz.set(c));
    }
  }
  [[nodiscard]] constexpr auto getCapacity() const -> U { return capacity; }

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

  constexpr auto insert(T *p, T x) -> T * {
    static_assert(std::is_same_v<S, ptrdiff_t>);
    invariant(p >= this->data());
    invariant(p <= this->data() + this->sz);
    invariant(this->sz < capacity);
    if (p < this->data() + this->sz)
      std::copy_backward(p, this->data() + this->sz,
                         this->data() + this->sz + 1);
    *p = x;
    ++this->sz;
    return p;
  }
  template <size_t SlabSize, bool BumpUp>
  constexpr void reserve(alloc::Arena<SlabSize, BumpUp> *alloc, U newCapacity) {
    if (newCapacity <= capacity) return;
    this->ptr = alloc->template reallocate<false, T>(
      const_cast<T *>(this->ptr), capacity, newCapacity, U{this->sz});
    // T *oldPtr =
    //   std::exchange(this->data(), alloc->template
    //   allocate<T>(newCapacity));
    // std::copy_n(oldPtr, U(this->sz), this->data());
    // alloc->deallocate(oldPtr, capacity);
    capacity = newCapacity;
  }

protected:
  // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
  [[no_unique_address]] U capacity{0};
};

// Figure out offset of first element
template <class T, class S, class A>
struct ArrayAlignmentAndSize : ResizeableView<T, S> {
  [[no_unique_address]] A a;
  alignas(T) char memory[sizeof(T)];
};

static_assert(std::is_copy_assignable_v<Array<void *, ptrdiff_t>>);
static_assert(!std::is_copy_assignable_v<MutArray<void *, ptrdiff_t>>);
static_assert(std::is_trivially_copyable_v<MutArray<void *, ptrdiff_t>>);
static_assert(std::is_trivially_move_assignable_v<MutArray<void *, ptrdiff_t>>);

/// Non-owning view of a managed array, capable of reallocating, etc.
/// It does not own memory. Mostly, it serves to drop the inlined
/// stack capacity of the `ManagedArray` from the type.
template <class T, Dimension S, class A = alloc::Mallocator<T>>
struct POLY_MATH_GSL_POINTER ReallocView : ResizeableView<T, S> {
  using BaseT = ResizeableView<T, S>;
  using U = containers::default_capacity_type_t<S>;
  constexpr ReallocView(T *p, S s, U c) noexcept : BaseT(p, s, c) {}
  constexpr ReallocView(T *p, S s, U c, A alloc) noexcept
    : BaseT(p, s, c), allocator(alloc) {}

  [[nodiscard]] constexpr auto newCapacity() const -> U {
    return this->capacity ? 2 * this->capacity : U{4};
  }
  template <class... Args>
  constexpr auto emplace_back(Args &&...args) -> decltype(auto) {
    static_assert(std::is_integral_v<S>, "emplace_back requires integral size");
    if (this->sz == this->capacity) [[unlikely]]
      reserve(newCapacity());
    return *std::construct_at<T>(this->data() + this->sz++,
                                 std::forward<Args>(args)...);
  }
  constexpr void push_back(T value) {
    static_assert(std::is_integral_v<S>, "push_back requires integral size");
    if (ptrdiff_t(this->sz) == this->capacity) [[unlikely]]
      reserve(newCapacity());
    std::construct_at<T>(this->data() + this->sz++, std::move(value));
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
    if constexpr (std::integral<S>) {
      if (nz <= oz) return;
      if (nz > this->capacity) {
        U newCapacity = U(nz);
        auto [newPtr, newCap] = alloc::alloc_at_least(allocator, newCapacity);
        if (oz) std::copy_n(this->data(), oz, newPtr);
        maybeDeallocate(newPtr, newCap);
        invariant(newCapacity > oz);
      }
      std::fill(this->data() + oz, this->data() + nz, T{});
    } else {
      static_assert(MatrixDimension<S>, "Can only resize 1 or 2d containers.");
      U len = U(nz);
      if (len == 0) return;
      auto newX = ptrdiff_t{RowStride(nz)}, oldX = ptrdiff_t{RowStride(oz)},
           newN = ptrdiff_t{Col(nz)}, oldN = ptrdiff_t{Col(oz)},
           newM = ptrdiff_t{Row(nz)}, oldM = ptrdiff_t{Row(oz)};
      bool newAlloc = len > this->capacity;
      bool inPlace = !newAlloc;
      T *npt = this->data();
      if (newAlloc) {
        alloc::AllocResult<T> res = alloc::alloc_at_least(allocator, len);
        npt = res.ptr;
        len = res.count;
      }
      // we can copy forward so long as the new stride is smaller
      // so that the start of the dst range is outside of the src range
      // we can also safely forward copy if we allocated a new ptr
      bool forwardCopy = (newX <= oldX) || newAlloc;
      ptrdiff_t colsToCopy = std::min(oldN, newN);
      // we only need to copy if memory shifts position
      bool copyCols = newAlloc || ((colsToCopy > 0) && (newX != oldX));
      // if we're in place, we have 1 less row to copy
      ptrdiff_t rowsToCopy = std::min(oldM, newM);
      ptrdiff_t fillCount = newN - colsToCopy;
      if ((rowsToCopy) && (copyCols || fillCount)) {
        if (forwardCopy) {
          // truncation, we need to copy rows to increase stride
          T *src = this->data();
          T *dst = npt;
          do {
            if (copyCols && (!inPlace)) std::copy_n(src, colsToCopy, dst);
            if (fillCount) std::fill_n(dst + colsToCopy, fillCount, T{});
            src += oldX;
            dst += newX;
            inPlace = false;
          } while (--rowsToCopy);
        } else /* [[unlikely]] */ {
          // backwards copy, only needed when we increasing stride but not
          // reallocating, which should be comparatively uncommon.
          // Should probably benchmark or determine actual frequency
          // before adding `[[unlikely]]`.
          invariant(inPlace);
          T *src = this->data() + (rowsToCopy + inPlace) * oldX;
          T *dst = npt + (rowsToCopy + inPlace) * newX;
          do {
            src -= oldX;
            dst -= newX;
            if (colsToCopy && (rowsToCopy > inPlace))
              std::copy_backward(src, src + colsToCopy, dst + colsToCopy);
            if (fillCount) std::fill_n(dst + colsToCopy, fillCount, T{});
          } while (--rowsToCopy);
        }
      }
      // zero init remaining rows
      for (ptrdiff_t m = oldM; m < newM; ++m)
        std::fill_n(npt + m * newX, newN, T{});
      if (newAlloc) maybeDeallocate(npt, len);
    }
  }
  constexpr void resize(Row<> r) {
    if constexpr (std::integral<S>) {
      return resize(S(r));
    } else if constexpr (MatrixDimension<S>) {
      S nz = this->sz;
      return resize(nz.set(r));
    }
  }
  constexpr void resize(Col<> c) {
    if constexpr (std::integral<S>) {
      return resize(S(c));
    } else if constexpr (MatrixDimension<S>) {
      S nz = this->sz;
      return resize(nz.set(c));
    }
  }
  constexpr void resizeForOverwrite(S M) {
    U L = U(M);
    if (L > U(this->sz)) growUndef(L);
    this->sz = M;
  }
  constexpr void resizeForOverwrite(Row<> r) {
    if constexpr (std::integral<S>) {
      return resizeForOverwrite(S(r));
    } else if constexpr (MatrixDimension<S>) {
      S nz = this->sz;
      return resizeForOverwrite(nz.set(r));
    }
  }
  constexpr void resizeForOverwrite(Col<> c) {
    if constexpr (std::integral<S>) {
      return resizeForOverwrite(S(c));
    } else if constexpr (MatrixDimension<S>) {
      S nz = this->sz;
      return resizeForOverwrite(nz.set(c));
    }
  }
  constexpr void reserve(S nz) {
    U newCapacity = U(nz);
    if (newCapacity <= this->capacity) return;
    // allocate new, copy, deallocate old
    auto [newPtr, newCap] = alloc::alloc_at_least(allocator, newCapacity);
    if (U oldLen = U(this->sz))
      std::uninitialized_copy_n(this->data(), oldLen, newPtr);
    maybeDeallocate(newPtr, newCap);
  }
  [[nodiscard]] constexpr auto get_allocator() const noexcept -> A {
    return allocator;
  }
  // set size and 0.
  constexpr void setSize(Row<> r, Col<> c) {
    resizeForOverwrite({r, c});
    this->fill(0);
  }
  constexpr void resize(Row<> MM, Col<> NN) { resize(DenseDims{MM, NN}); }
  constexpr void reserve(Row<> M, Col<> N) {
    if constexpr (std::is_same_v<S, StridedDims<>>)
      reserve(StridedDims{M, N, max(N, RowStride(this->dim()))});
    else if constexpr (std::is_same_v<S, SquareDims<>>)
      reserve(SquareDims{Row<>{std::max(ptrdiff_t(M), ptrdiff_t(N))}});
    else reserve(DenseDims{M, N});
  }
  constexpr void reserve(Row<> M, RowStride<> X) {
    if constexpr (std::is_same_v<S, StridedDims<>>)
      reserve(S{M, Col<>{ptrdiff_t(X)}, X});
    else if constexpr (std::is_same_v<S, SquareDims<>>)
      reserve(SquareDims{Row<>{std::max(ptrdiff_t(M), ptrdiff_t(X))}});
    else reserve(S{M, Col<>{ptrdiff_t(X)}});
  }
  constexpr void clearReserve(Row<> M, Col<> N) {
    this->clear();
    reserve(M, N);
  }
  constexpr void clearReserve(Row<> M, RowStride<> X) {
    this->clear();
    reserve(M, X);
  }
  constexpr void resizeForOverwrite(Row<> M, Col<> N, RowStride<> X) {
    invariant(X >= N);
    if constexpr (std::is_same_v<S, StridedDims<>>)
      resizeForOverwrite(S{M, N, X});
    else if constexpr (std::is_same_v<S, SquareDims<>>) {
      invariant(ptrdiff_t(M) == ptrdiff_t(N));
      resizeForOverwrite(S{M});
    } else resizeForOverwrite(S{M, N});
  }
  constexpr void resizeForOverwrite(Row<> M, Col<> N) {
    if constexpr (std::is_same_v<S, StridedDims<>>)
      resizeForOverwrite(S{M, N, {ptrdiff_t(N)}});
    else if constexpr (std::is_same_v<S, SquareDims<>>) {
      invariant(ptrdiff_t(M) == ptrdiff_t(N));
      resizeForOverwrite(S{M});
    } else resizeForOverwrite(S{M, N});
  }

protected:
  // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
  [[no_unique_address]] A allocator{};

  constexpr void allocateAtLeast(U len) {
    alloc::AllocResult<T> res = alloc::alloc_at_least(allocator, len);
    this->ptr = res.ptr;
    this->capacity = res.count;
  }
  [[nodiscard]] auto firstElt() const -> const void * {
    using AS = ArrayAlignmentAndSize<T, S, A>;
// AS is not a standard layout type, as more than one layer of the hierarchy
// contain non-static data members. Using `offsetof` is conditionally supported
// by some compilers as of c++17 (it was UB prior). Both Clang and GCC support
// it.
#if !defined(__clang__) && defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winvalid-offsetof"
#endif
    constexpr size_t offset = offsetof(AS, memory);
#if !defined(__clang__) && defined(__GNUC__)
#pragma GCC diagnostic pop
#else
#pragma clang diagnostic pop
#endif
    return reinterpret_cast<const char *>(this) + offset;
  }
  [[nodiscard]] auto isSmall() const -> bool { return this->ptr == firstElt(); }
  [[nodiscard]] auto wasAllocated() const -> bool {
    return this->ptr && this->ptr != firstElt();
  }
  // this method should only be called from the destructor
  // (and the implementation taking the new ptr and capacity)
  void maybeDeallocate() noexcept {
    if (wasAllocated()) allocator.deallocate(this->data(), this->capacity);
  }
  // this method should be called whenever the buffer lives
  void maybeDeallocate(T *newPtr, U newCapacity) noexcept {
    maybeDeallocate();
    this->ptr = newPtr;
    this->capacity = newCapacity;
  }
  // grow, discarding old data
  void growUndef(U M) {
    if (M <= this->capacity) return;
    maybeDeallocate();
    // because this doesn't care about the old data,
    // we can allocate after freeing, which may be faster
    this->ptr = this->allocator.allocate(M);
    this->capacity = M;
#ifndef NDEBUG
    if constexpr (std::numeric_limits<T>::has_signaling_NaN)
      std::fill_n(this->data(), M, std::numeric_limits<T>::signaling_NaN());
    else std::fill_n(this->data(), M, std::numeric_limits<T>::min());
#endif
  }
};

template <class T, class S>
concept AbstractSimilar =
  (MatrixDimension<S> && AbstractMatrix<T>) ||
  ((std::integral<S> || std::is_same_v<S, StridedRange> ||
    StaticInt<S>)&&AbstractVector<T>);

/// Stores memory, then pointer.
/// Thus struct's alignment determines initial alignment
/// of the stack memory.
/// Information related to size is then grouped next to the pointer.
///
/// The Intel compiler + OpenMP appears to memcpy data around,
/// or at least build ManagedArrays bypassing the constructors listed here.
/// This caused invalid frees, as the pointer still pointed to the old
/// stack memory.
template <class T, Dimension S, ptrdiff_t N, class A>
struct POLY_MATH_GSL_OWNER ManagedArray : ReallocView<T, S, A> {
  static_assert(std::is_trivially_destructible_v<T>);
  using BaseT = ReallocView<T, S, A>;
  using U = containers::default_capacity_type_t<S>;
  // We're deliberately not initializing storage.
#if !defined(__clang__) && defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wuninitialized"
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
#endif
  constexpr ManagedArray(A a) noexcept : BaseT{memory.data(), S{}, N, a} {
#ifndef NDEBUG
    if (!N) return;
    if constexpr (std::numeric_limits<T>::has_signaling_NaN)
      std::fill_n(this->data(), N, std::numeric_limits<T>::signaling_NaN());
    else if constexpr (std::numeric_limits<T>::is_specialized)
      std::fill_n(this->data(), N, std::numeric_limits<T>::min());
#endif
  }
  constexpr ManagedArray(S s, A a) noexcept : BaseT{memory.data(), s, N, a} {
    U len = U(this->sz);
    if (len > N) this->allocateAtLeast(len);
#ifndef NDEBUG
    if (!len) return;
    if constexpr (std::numeric_limits<T>::has_signaling_NaN)
      std::fill_n(this->data(), len, std::numeric_limits<T>::signaling_NaN());
    else if constexpr (std::numeric_limits<T>::is_specialized)
      std::fill_n(this->data(), len, std::numeric_limits<T>::min());
#endif
  }
  constexpr ManagedArray(S s, T x, A a) noexcept
    : BaseT{memory.data(), s, N, a} {
    U len = U(this->sz);
    if (len > N) this->allocateAtLeast(len);
    if (len) std::fill_n(this->data(), len, x);
  }
  constexpr ManagedArray() noexcept : ManagedArray(A{}){};
  constexpr ManagedArray(S s) noexcept : ManagedArray(s, A{}){};
  constexpr ManagedArray(ptrdiff_t s) noexcept
  requires(std::same_as<S, SquareDims<>>)
    : ManagedArray(SquareDims<>{Row<>{s}}, A{}){};
  constexpr ManagedArray(S s, T x) noexcept : ManagedArray(s, x, A{}){};

  // constexpr ManagedArray(std::type_identity<T>) noexcept :
  // ManagedArray(A{}){};
  constexpr ManagedArray(std::type_identity<T>, S s) noexcept
    : ManagedArray(s, A{}){};
  // constexpr ManagedArray(std::type_identity<T>, A a) noexcept :
  // ManagedArray(a){}; constexpr ManagedArray(std::type_identity<T>, S s, A
  // a) noexcept
  //   : ManagedArray(s, a){};

  template <class D>
  constexpr ManagedArray(const ManagedArray<T, D, N, A> &b) noexcept
    : BaseT{memory.data(), S(b.dim()), U(N), b.get_allocator()} {
    U len = U(this->sz);
    this->growUndef(len);
    std::copy_n(b.data(), len, this->data());
  }
  template <std::convertible_to<T> Y, class D, class AY>
  constexpr ManagedArray(const ManagedArray<Y, D, N, AY> &b) noexcept
    : BaseT{memory.data(), S(b.dim()), U(N), b.get_allocator()} {
    U len = U(this->sz);
    this->growUndef(len);
    (*this) << b;
  }
  template <std::convertible_to<T> Y, size_t M>
  constexpr ManagedArray(std::array<Y, M> il) noexcept
    : BaseT{memory.data(), S(il.size()), U(N)} {
    U len = U(this->sz);
    this->growUndef(len);
    std::copy_n(il.begin(), len, this->data());
  }
  template <std::convertible_to<T> Y, class D, class AY>
  constexpr ManagedArray(const ManagedArray<Y, D, N, AY> &b, S s) noexcept
    : BaseT{memory.data(), S(s), U(N), b.get_allocator()} {
    U len = U(this->sz);
    invariant(len == U(b.size()));
    this->growUndef(len);
    T *p = this->data();
    for (ptrdiff_t i = 0; i < len; ++i) p[i] = b[i];
  }
  constexpr ManagedArray(const ManagedArray &b) noexcept
    : BaseT{memory.data(), S(b.dim()), U(N), b.get_allocator()} {
    U len = U(this->sz);
    this->growUndef(len);
    std::copy_n(b.data(), len, this->data());
  }
  constexpr ManagedArray(const Array<T, S> &b) noexcept
    : BaseT{memory.data(), S(b.dim()), U(N)} {
    U len = U(this->sz);
    this->growUndef(len);
    std::copy_n(b.data(), len, this->data());
  }
  template <AbstractSimilar<S> V>
  constexpr ManagedArray(const V &b) noexcept
    : BaseT{memory.data(), S(shape(b)), U(N)} {
    U len = U(this->sz);
    this->growUndef(len);
    (*this) << b;
  }
  template <class D>
  constexpr ManagedArray(ManagedArray<T, D, N, A> &&b) noexcept
    : BaseT{memory.data(), b.dim(), U(N), b.get_allocator()} {
    if (b.isSmall()) { // copy
      std::copy_n(b.data(), ptrdiff_t(b.dim()), this->data());
    } else { // steal
      this->ptr = b.data();
      this->capacity = b.getCapacity();
    }
    b.resetNoFree();
  }
  constexpr ManagedArray(ManagedArray &&b) noexcept
    : BaseT{memory.data(), b.dim(), U(N), b.get_allocator()} {
    if constexpr (N > 0) {
      if (b.isSmall()) { // copy
        std::copy_n(b.data(), ptrdiff_t(b.dim()), this->data());
      } else { // steal
        this->ptr = b.data();
        this->capacity = b.getCapacity();
      }
    } else {
      this->capacity = b.getCapacity();
      if (this->capacity) this->ptr = b.data();
    }
    b.resetNoFree();
  }
  template <class D>
  constexpr ManagedArray(ManagedArray<T, D, N, A> &&b, S s) noexcept
    : BaseT{memory.data(), s, U(N), b.get_allocator()} {
    if (b.isSmall()) { // copy
      std::copy_n(b.data(), ptrdiff_t(b.dim()), this->data());
    } else { // steal
      this->ptr = b.data();
      this->capacity = b.getCapacity();
    }
    b.resetNoFree();
  }
  template <std::convertible_to<T> Y>
  constexpr ManagedArray(const SmallSparseMatrix<Y> &B)
    : BaseT{memory.data(), B.dim(), N} {
    U len = U(this->sz);
    this->growUndef(len);
    this->fill(0);
    ptrdiff_t k = 0;
    for (ptrdiff_t i = 0; i < this->numRow(); ++i) {
      uint32_t m = B.getRows()[i] & 0x00ffffff;
      ptrdiff_t j = 0;
      while (m) {
        uint32_t tz = std::countr_zero(m);
        m >>= tz + 1;
        j += tz;
        (*this)[i, j++] = T(B.getNonZeros()[k++]);
      }
    }
    invariant(k == B.getNonZeros().size());
  }
#if !defined(__clang__) && defined(__GNUC__)
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
#else
#pragma clang diagnostic pop
#endif

  template <class D>
  constexpr auto operator=(const ManagedArray<T, D, N, A> &b) noexcept
    -> ManagedArray & {
    if (this == &b) return *this;
    this->sz = b.dim();
    U len = U(this->sz);
    this->growUndef(len);
    std::copy_n(b.data(), len, this->data());
    return *this;
  }
  template <class D>
  constexpr auto operator=(ManagedArray<T, D, N, A> &&b) noexcept
    -> ManagedArray & {
    if (this->data() == b.data()) return *this;
    // here, we commandeer `b`'s memory
    this->sz = b.dim();
    this->allocator = std::move(b.get_allocator());
    // if `b` is small, we need to copy memory
    // no need to shrink our capacity
    if (b.isSmall()) std::copy_n(b.data(), ptrdiff_t(this->sz), this->data());
    else this->maybeDeallocate(b.data(), b.getCapacity());
    b.resetNoFree();
    return *this;
  }
  constexpr auto operator=(const ManagedArray &b) noexcept -> ManagedArray & {
    if (this == &b) return *this;
    this->sz = b.dim();
    U len = U(this->sz);
    this->growUndef(len);
    std::copy_n(b.data(), len, this->data());
    return *this;
  }
  constexpr auto operator=(ManagedArray &&b) noexcept -> ManagedArray & {
    if (this == &b) return *this;
    // here, we commandeer `b`'s memory
    this->sz = b.dim();
    this->allocator = std::move(b.get_allocator());
    if (b.isSmall()) {
      // if `b` is small, we need to copy memory
      // no need to shrink our capacity
      std::copy_n(b.data(), ptrdiff_t(this->sz), this->data());
    } else { // otherwise, we take its pointer
      this->maybeDeallocate(b.data(), b.getCapacity());
    }
    b.resetNoFree();
    return *this;
  }
  [[nodiscard]] constexpr auto isSmall() const -> bool {
    return this->data() == memory.data();
  }
  constexpr void resetNoFree() {
    this->ptr = memory.data();
    this->sz = S{};
    this->capacity = N;
  }
  constexpr ~ManagedArray() noexcept { this->maybeDeallocate(); }

  [[nodiscard]] static constexpr auto identity(ptrdiff_t M) -> ManagedArray {
    static_assert(MatrixDimension<S>);
    ManagedArray B(SquareDims<>{M}, T{0});
    B.diag() << 1;
    return B;
  }
  [[nodiscard]] static constexpr auto identity(Row<> R) -> ManagedArray {
    static_assert(MatrixDimension<S>);
    return identity(ptrdiff_t(R));
  }
  [[nodiscard]] static constexpr auto identity(Col<> C) -> ManagedArray {
    static_assert(MatrixDimension<S>);
    return identity(ptrdiff_t(C));
  }
  friend inline void PrintTo(const ManagedArray &x, ::std::ostream *os) {
    *os << x;
  }

private:
  [[no_unique_address]] containers::Storage<T, N> memory;
};

static_assert(std::move_constructible<ManagedArray<intptr_t, ptrdiff_t>>);
static_assert(std::copyable<ManagedArray<intptr_t, ptrdiff_t>>);
// Check that `[[no_unique_address]]` is working.
// sizes should be:
// [ptr, dims, capacity, allocator, array]
// 8 + 3*4 + 4 + 0 + 64*8 = 24 + 512 = 536
static_assert(sizeof(ManagedArray<int64_t, StridedDims<>, 64,
                                  alloc::Mallocator<int64_t>>) == 552);
// sizes should be:
// [ptr, dims, capacity, allocator, array]
// 8 + 2*4 + 8 + 0 + 64*8 = 24 + 512 = 536
static_assert(
  sizeof(ManagedArray<int64_t, DenseDims<>, 64, alloc::Mallocator<int64_t>>) ==
  544);
// sizes should be:
// [ptr, dims, capacity, allocator, array]
// 8 + 1*4 + 4 + 0 + 64*8 = 16 + 512 = 528
static_assert(
  sizeof(ManagedArray<int64_t, SquareDims<>, 64, alloc::Mallocator<int64_t>>) ==
  536);

template <class T, ptrdiff_t N = containers::PreAllocStorage<T, ptrdiff_t>()>
using Vector = ManagedArray<T, ptrdiff_t, N>;
template <class T> using PtrVector = Array<T, ptrdiff_t>;
template <class T> using MutPtrVector = MutArray<T, ptrdiff_t>;

static_assert(std::move_constructible<Vector<intptr_t>>);
static_assert(std::copy_constructible<Vector<intptr_t>>);
static_assert(std::copyable<Vector<intptr_t>>);
static_assert(AbstractVector<Array<int64_t, ptrdiff_t>>);
static_assert(AbstractVector<MutArray<int64_t, ptrdiff_t>>);
static_assert(AbstractVector<Vector<int64_t>>);
static_assert(!AbstractVector<int64_t>);
static_assert(!std::is_trivially_copyable_v<Vector<int64_t>>);
static_assert(!std::is_trivially_destructible_v<Vector<int64_t>>);

template <typename T> using StridedVector = Array<T, StridedRange>;
template <typename T> using MutStridedVector = MutArray<T, StridedRange>;

static_assert(AbstractVector<StridedVector<int64_t>>);
static_assert(AbstractVector<MutStridedVector<int64_t>>);
static_assert(std::is_trivially_copyable_v<StridedVector<int64_t>>);

template <class T> using PtrMatrix = Array<T, StridedDims<>>;
template <class T> using MutPtrMatrix = MutArray<T, StridedDims<>>;
template <class T, ptrdiff_t L = 64>
using Matrix = ManagedArray<T, StridedDims<>, L>;
template <class T> using DensePtrMatrix = Array<T, DenseDims<>>;
template <class T> using MutDensePtrMatrix = MutArray<T, DenseDims<>>;
template <class T, ptrdiff_t L = 64>
using DenseMatrix = ManagedArray<T, DenseDims<>, L>;
template <class T> using SquarePtrMatrix = Array<T, SquareDims<>>;
template <class T> using MutSquarePtrMatrix = MutArray<T, SquareDims<>>;
template <class T,
          ptrdiff_t L = containers::PreAllocSquareStorage<T, SquareDims<>>()>
using SquareMatrix = ManagedArray<T, SquareDims<>, L>;

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
static_assert(sizeof(ManagedArray<int32_t, DenseDims<3, 5>, 15>) ==
              sizeof(int32_t *) + 16 * sizeof(int32_t));
static_assert(sizeof(ManagedArray<int32_t, DenseDims<>, 15>) ==
              sizeof(int32_t *) + 3 * sizeof(ptrdiff_t) + 16 * sizeof(int32_t));
static_assert(sizeof(ReallocView<int32_t, DenseDims<>>) ==
              sizeof(int32_t *) + 3 * sizeof(ptrdiff_t));

static_assert(!AbstractVector<PtrMatrix<int64_t>>,
              "PtrMatrix<int64_t> isa AbstractVector succeeded");
static_assert(!AbstractVector<MutPtrMatrix<int64_t>>,
              "PtrMatrix<int64_t> isa AbstractVector succeeded");
static_assert(!AbstractVector<const PtrMatrix<int64_t>>,
              "PtrMatrix<int64_t> isa AbstractVector succeeded");

static_assert(AbstractMatrix<PtrMatrix<int64_t>>,
              "PtrMatrix<int64_t> isa AbstractMatrix failed");
static_assert(
  std::same_as<decltype(PtrMatrix<int64_t>(
                 nullptr, Row<>{0}, Col<>{0})[ptrdiff_t(0), ptrdiff_t(0)]),
               const int64_t &>);
static_assert(
  std::same_as<std::remove_reference_t<decltype(MutPtrMatrix<int64_t>(
                 nullptr, Row<>{0}, Col<>{0})[ptrdiff_t(0), ptrdiff_t(0)])>,
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

static_assert(AbstractVector<Vector<int64_t>>,
              "PtrVector<int64_t> isa AbstractVector failed");

static_assert(!AbstractMatrix<MutPtrVector<int64_t>>,
              "PtrVector<int64_t> isa AbstractMatrix succeeded");
static_assert(!AbstractMatrix<PtrVector<int64_t>>,
              "PtrVector<const int64_t> isa AbstractMatrix succeeded");
static_assert(!AbstractMatrix<const PtrVector<int64_t>>,
              "PtrVector<const int64_t> isa AbstractMatrix succeeded");
static_assert(!AbstractMatrix<const MutPtrVector<int64_t>>,
              "PtrVector<const int64_t> isa AbstractMatrix succeeded");
static_assert(std::is_convertible_v<DenseMatrix<int64_t>, Matrix<int64_t>>);
static_assert(
  std::is_convertible_v<DenseMatrix<int64_t>, DensePtrMatrix<int64_t>>);
static_assert(std::is_convertible_v<DenseMatrix<int64_t>, PtrMatrix<int64_t>>);
static_assert(std::is_convertible_v<SquareMatrix<int64_t>, Matrix<int64_t>>);
static_assert(
  std::is_convertible_v<SquareMatrix<int64_t>, MutPtrMatrix<int64_t>>);

template <class T, class S>
ManagedArray(std::type_identity<T>, S s) -> ManagedArray<T, S>;

template <class S> using IntArray = Array<int64_t, S>;
template <VectorDimension S = ptrdiff_t>
using IntVector = ManagedArray<int64_t, S>;
template <MatrixDimension S = DenseDims<>>
using IntMatrix = ManagedArray<int64_t, S>;

static_assert(std::same_as<IntMatrix<>::value_type, int64_t>);
static_assert(AbstractMatrix<IntMatrix<>>);
static_assert(std::copyable<IntMatrix<>>);
static_assert(std::same_as<utils::eltype_t<Matrix<int64_t>>, int64_t>);

static_assert(std::convertible_to<Array<int64_t, SquareDims<>>,
                                  Array<int64_t, StridedDims<>>>);

inline auto printVectorImpl(std::ostream &os, const AbstractVector auto &a)
  -> std::ostream & {
  os << "[ ";
  if (ptrdiff_t M = a.size()) {
    print_obj(os, a[0]);
    for (ptrdiff_t m = 1; m < M; m++) print_obj(os << ", ", a[m]);
  }
  os << " ]";
  return os;
}
template <typename T>
inline auto printVector(std::ostream &os, PtrVector<T> a) -> std::ostream & {
  return printVectorImpl(os, a);
}
template <typename T>
inline auto printVector(std::ostream &os, StridedVector<T> a)
  -> std::ostream & {
  return printVectorImpl(os, a);
}

template <typename T>
inline auto operator<<(std::ostream &os, PtrVector<T> const &A)
  -> std::ostream & {
  return printVector(os, A);
}
inline auto operator<<(std::ostream &os, const AbstractVector auto &A)
  -> std::ostream & {
  Vector<utils::eltype_t<decltype(A)>> B(A.size());
  B << A;
  return printVector(os, B);
}
template <std::integral T> static constexpr auto maxPow10() -> size_t {
  if constexpr (sizeof(T) == 1) return 3;
  else if constexpr (sizeof(T) == 2) return 5;
  else if constexpr (sizeof(T) == 4) return 10;
  else if constexpr (std::signed_integral<T>) return 19;
  else return 20;
}

template <std::unsigned_integral T> constexpr auto countDigits(T x) {
  std::array<T, maxPow10<T>() + 1> powers;
  powers[0] = 0;
  powers[1] = 10;
  for (ptrdiff_t i = 2; i < std::ssize(powers); i++)
    powers[i] = powers[i - 1] * 10;
  std::array<T, sizeof(T) * 8 + 1> bits;
  if constexpr (sizeof(T) == 8) {
    bits = {1,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4,  4,  5,  5,  5,
            6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  9,  9,  9,  10, 10, 10, 10,
            11, 11, 11, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16,
            16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 19, 20};
  } else if constexpr (sizeof(T) == 4) {
    bits = {1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4,  5,  5, 5,
            6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10};
  } else if constexpr (sizeof(T) == 2) {
    bits = {1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5};
  } else if constexpr (sizeof(T) == 1) {
    bits = {1, 1, 1, 1, 2, 2, 2, 3, 3};
  }
  T digits = bits[8 * sizeof(T) - std::countl_zero(x)];
  return std::make_signed_t<T>(digits - (x < powers[digits - 1]));
}
template <std::signed_integral T> constexpr auto countDigits(T x) {
  using U = std::make_unsigned_t<T>;
  if (x == std::numeric_limits<T>::min()) return T(sizeof(T) == 8 ? 20 : 11);
  return countDigits<U>(U(std::abs(x))) + T{x < 0};
}
constexpr auto countDigits(Rational x) -> ptrdiff_t {
  ptrdiff_t num = countDigits(x.numerator);
  return (x.denominator == 1) ? num : num + countDigits(x.denominator) + 2;
}
/// \brief Returns the maximum number of digits per column of a matrix.
constexpr auto getMaxDigits(PtrMatrix<Rational> A) -> Vector<ptrdiff_t> {
  ptrdiff_t M = ptrdiff_t(A.numRow());
  ptrdiff_t N = ptrdiff_t(A.numCol());
  Vector<ptrdiff_t> maxDigits{ptrdiff_t(N), 0};
  invariant(ptrdiff_t(maxDigits.size()), N);
  // this is slow, because we count the digits of every element
  // we could optimize this by reducing the number of calls to countDigits
  for (ptrdiff_t i = 0; i < M; i++) {
    for (ptrdiff_t j = 0; j < N; j++) {
      ptrdiff_t c = countDigits(A[i, j]);
      maxDigits[j] = std::max(maxDigits[j], c);
    }
  }
  return maxDigits;
}

/// Returns the number of digits of the largest number in the matrix.
template <std::integral T>
constexpr auto getMaxDigits(PtrMatrix<T> A) -> Vector<T> {
  ptrdiff_t M = ptrdiff_t(A.numRow());
  ptrdiff_t N = ptrdiff_t(A.numCol());
  Vector<T> maxDigits{ptrdiff_t(N), T{}};
  invariant(ptrdiff_t(maxDigits.size()), N);
  // first, we find the digits with the maximum value per column
  for (ptrdiff_t i = 0; i < M; i++) {
    for (ptrdiff_t j = 0; j < N; j++) {
      // negative numbers need one more digit
      // first, we find the maximum value per column,
      // dividing positive numbers by -10
      T Aij = A[i, j];
      if constexpr (std::signed_integral<T>)
        maxDigits[j] = std::min(maxDigits[j], Aij > 0 ? Aij / -10 : Aij);
      else maxDigits[j] = std::max(maxDigits[j], Aij);
    }
  }
  // then, we count the digits of the maximum value per column
  for (ptrdiff_t j = 0; j < maxDigits.size(); j++)
    maxDigits[j] = countDigits(maxDigits[j]);
  return maxDigits;
}

template <typename T>
inline auto printMatrix(std::ostream &os, PtrMatrix<T> A) -> std::ostream & {
  // std::ostream &printMatrix(std::ostream &os, T const &A) {
  auto [M, N] = shape(A);
  if ((!M) || (!N)) return os << "[ ]";
  // first, we determine the number of digits needed per column
  auto maxDigits{getMaxDigits(A)};
  using U = decltype(countDigits(std::declval<T>()));
  for (ptrdiff_t i = 0; i < M; i++) {
    if (i) os << "  ";
    else os << "\n[ ";
    for (ptrdiff_t j = 0; j < N; j++) {
      auto Aij = A[i, j];
      for (U k = 0; k < U(maxDigits[j]) - countDigits(Aij); k++) os << " ";
      os << Aij;
      if (j != ptrdiff_t(N) - 1) os << " ";
      else if (i != ptrdiff_t(M) - 1) os << "\n";
    }
  }
  return os << " ]";
}
// We mirror `A` with a matrix of integers indicating sizes, and a vectors of
// chars. We fill the matrix with the number of digits of each element, and
// the vector with the characters of each element. We could use a vector of
// vectors of chars to avoid needing to copy memory on reallocation, but this
// would yield more complicated management. We should also generally be able
// to avoid allocations. We can use a Vector with a lot of initial capacity,
// and then resize based on a conservative estimate of the number of chars per
// elements.
inline auto printMatrix(std::ostream &os, PtrMatrix<double> A)
  -> std::ostream & {
  // std::ostream &printMatrix(std::ostream &os, T const &A) {
  auto [M, N] = shape(A);
  if ((!M) || (!N)) return os << "[ ]";
  // first, we determine the number of digits needed per column
  Vector<char, 512> digits;
  digits.resizeForOverwrite(512);
  // we can't have more than 255 digits
  DenseMatrix<uint8_t> numDigits{DenseDims<>{{M}, {N}}};
  char *ptr = digits.begin();
  char *pEnd = digits.end();
  for (ptrdiff_t m = 0; m < M; m++) {
    for (ptrdiff_t n = 0; n < N; n++) {
      auto Aij = A[m, n];
      while (true) {
        auto [p, ec] = std::to_chars(ptr, pEnd, Aij);
        if (ec == std::errc()) [[likely]] {
          numDigits[m, n] = std::distance(ptr, p);
          ptr = p;
          break;
        }
        // we need more space
        ptrdiff_t elemSoFar = m * ptrdiff_t(N) + n;
        ptrdiff_t charSoFar = std::distance(digits.begin(), ptr);
        // cld
        ptrdiff_t charPerElem = (charSoFar + elemSoFar - 1) / elemSoFar;
        ptrdiff_t newCapacity =
          (1 + charPerElem) * M * N; // +1 for good measure
        digits.resize(newCapacity);
        ptr = digits.begin() + charSoFar;
        pEnd = digits.end();
      }
    }
  }
  Vector<uint8_t> maxDigits;
  maxDigits.resizeForOverwrite(N);
  maxDigits << numDigits[0, _];
  for (ptrdiff_t m = 0; m < M; m++)
    for (ptrdiff_t n = 0; n < N; n++)
      maxDigits[n] = std::max(maxDigits[n], numDigits[m, n]);

  ptr = digits.begin();
  // we will allocate 512 bytes at a time
  for (ptrdiff_t i = 0; i < M; i++) {
    if (i) os << "  ";
    else os << "\n[ ";
    for (ptrdiff_t j = 0; j < N; j++) {
      ptrdiff_t nD = numDigits[i, j];
      for (ptrdiff_t k = 0; k < maxDigits[j] - nD; k++) os << " ";
      os << std::string_view(ptr, nD);
      if (j != ptrdiff_t(N) - 1) os << " ";
      else if (i != ptrdiff_t(M) - 1) os << "\n";
      ptr += nD;
    }
  }
  return os << " ]";
}

template <typename T, ptrdiff_t R, ptrdiff_t C, ptrdiff_t X>
inline auto operator<<(std::ostream &os, Array<T, StridedDims<R, C, X>> A)
  -> std::ostream & {
  return printMatrix(os, A);
}
template <typename T, ptrdiff_t R>
inline auto operator<<(std::ostream &os, Array<T, SquareDims<R>> A)
  -> std::ostream & {
  return printMatrix(os, PtrMatrix<T>{A});
}
template <typename T, ptrdiff_t R, ptrdiff_t C>
inline auto operator<<(std::ostream &os, Array<T, DenseDims<R, C>> A)
  -> std::ostream & {
  return printMatrix(os, PtrMatrix<T>{A});
}

static_assert(std::same_as<const int64_t &,
                           decltype(std::declval<PtrMatrix<int64_t>>()[0, 0])>);
static_assert(std::is_trivially_copyable_v<MutArray<int64_t, ptrdiff_t>>);

} // namespace poly::math
