#pragma once

#include "Containers/Storage.hpp"
#include "Math/ArrayOps.hpp"
#include "Math/AxisTypes.hpp"
#include "Math/Indexing.hpp"
#include "Math/Iterators.hpp"
#include "Math/Matrix.hpp"
#include "Math/MatrixDimensions.hpp"
#include "Math/Rational.hpp"
#include "Math/Vector.hpp"
#include "Utilities/Allocators.hpp"
#include "Utilities/Invariant.hpp"
#include "Utilities/Optional.hpp"
#include "Utilities/TypePromotion.hpp"
#include "Utilities/Valid.hpp"
#include <algorithm>
#include <charconv>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>

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
template <class T, class S, size_t N = PreAllocStorage<T>(),
          class A = std::allocator<T>,
          std::unsigned_integral U = default_capacity_type_t<S>>
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
using utils::NotNull, utils::Optional;

/// Constant Array
template <class T, class S> struct POLY_MATH_GSL_POINTER Array {
  static_assert(!std::is_const_v<T>, "T shouldn't be const");
  static_assert(std::is_trivially_destructible_v<T>,
                "maybe should add support for destroying");
  using value_type = T;
  using reference = T &;
  using const_reference = const T &;
  using size_type = unsigned;
  using difference_type = int;
  using iterator = T *;
  using const_iterator = const T *;
  using pointer = T *;
  using const_pointer = const T *;
  using concrete = std::true_type;

  constexpr Array() = default;
  constexpr Array(const Array &) = default;
  constexpr Array(Array &&) noexcept = default;
  constexpr auto operator=(const Array &) -> Array & = default;
  constexpr auto operator=(Array &&) noexcept -> Array & = default;
  constexpr Array(const T *p, S s) : ptr(p), sz(s) {}
  constexpr Array(NotNull<const T> p, S s) : ptr(p), sz(s) {}
  constexpr Array(const T *p, Row r, Col c) : ptr(p), sz(S{r, c}) {}
  constexpr Array(NotNull<const T> p, Row r, Col c)
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
  [[nodiscard]] constexpr auto wrappedPtr() noexcept -> NotNull<T> {
    return ptr;
  }

  [[nodiscard]] constexpr auto begin() const noexcept {
    const T *p = ptr;
    if constexpr (std::is_same_v<S, StridedRange>)
      return StridedIterator{p, sz.stride};
    else return p;
  }
  [[nodiscard]] constexpr auto end() const noexcept {
    return begin() + ptrdiff_t(sz);
  }
  [[nodiscard]] constexpr auto rbegin() const noexcept {
    return std::reverse_iterator(end());
  }
  [[nodiscard]] constexpr auto rend() const noexcept {
    return std::reverse_iterator(begin());
  }
  [[nodiscard]] constexpr auto front() const noexcept -> const T & {
    return *begin();
  }
  [[nodiscard]] constexpr auto back() const noexcept -> const T & {
    return *(end() - 1);
  }
  // indexing has two components:
  // 1. offsetting the pointer
  // 2. calculating new dim
  // static constexpr auto slice(NotNull<T>, Index<S> auto i){
  //   auto
  // }
  constexpr auto operator[](Index<S> auto i) const noexcept -> decltype(auto) {
    auto offset = calcOffset(sz, i);
    auto newDim = calcNewDim(sz, i);
    invariant(ptr != nullptr);
    if constexpr (std::is_same_v<decltype(newDim), Empty>)
      return static_cast<const T *>(ptr)[offset];
    else return Array<T, decltype(newDim)>{ptr + offset, newDim};
  }
  // TODO: switch to operator[] when we enable c++23
  // for vectors, we just drop the column, essentially broadcasting
  template <class R, class C>
  constexpr auto operator()(R r, C c) const noexcept -> decltype(auto) {
    if constexpr (MatrixDimension<S>)
      return (*this)[CartesianIndex<R, C>{r, c}];
    else return (*this)[ptrdiff_t(r)];
  }
  [[nodiscard]] constexpr auto minRowCol() const -> ptrdiff_t {
    return std::min(ptrdiff_t(numRow()), ptrdiff_t(numCol()));
  }

  [[nodiscard]] constexpr auto diag() const noexcept {
    StridedRange r{minRowCol(), unsigned(RowStride{sz}) + 1};
    invariant(ptr != nullptr);
    return Array<T, StridedRange>{ptr, r};
  }
  [[nodiscard]] constexpr auto antiDiag() const noexcept {
    StridedRange r{minRowCol(), unsigned(RowStride{sz}) - 1};
    invariant(ptr != nullptr);
    return Array<T, StridedRange>{ptr + ptrdiff_t(Col{sz}) - 1, r};
  }
  [[nodiscard]] constexpr auto isSquare() const noexcept -> bool {
    return Row{sz} == Col{sz};
  }
  [[nodiscard]] constexpr auto checkSquare() const -> Optional<ptrdiff_t> {
    ptrdiff_t N = ptrdiff_t(numRow());
    if (N != ptrdiff_t(numCol())) return {};
    return N;
  }

  [[nodiscard]] constexpr auto numRow() const noexcept -> Row {
    return Row{sz};
  }
  [[nodiscard]] constexpr auto numCol() const noexcept -> Col {
    return Col{sz};
  }
  [[nodiscard]] constexpr auto rowStride() const noexcept -> RowStride {
    return RowStride{sz};
  }
  [[nodiscard]] constexpr auto empty() const -> bool { return sz == S{}; }
  [[nodiscard]] constexpr auto size() const noexcept {
    if constexpr (StaticInt<S>) return S{};
    else if constexpr (std::integral<S>) return sz;
    else if constexpr (std::is_same_v<S, StridedRange>) return ptrdiff_t(sz);
    else return CartesianIndex{Row{sz}, Col{sz}};
  }
  [[nodiscard]] constexpr auto dim() const noexcept -> S { return sz; }
  constexpr void clear() { sz = S{}; }
  [[nodiscard]] constexpr auto transpose() const { return Transpose{*this}; }
  [[nodiscard]] constexpr auto isExchangeMatrix() const -> bool {
    ptrdiff_t N = ptrdiff_t(numRow());
    if (N != ptrdiff_t(numCol())) return false;
    for (ptrdiff_t i = 0; i < N; ++i) {
      for (ptrdiff_t j = 0; j < N; ++j)
        if ((*this)(i, j) != (i + j == N - 1)) return false;
    }
  }
  [[nodiscard]] constexpr auto isDiagonal() const -> bool {
    for (Row r = 0; r < numRow(); ++r)
      for (Col c = 0; c < numCol(); ++c)
        if (r != c && (*this)(r, c) != 0) return false;
    return true;
  }
  [[nodiscard]] constexpr auto view() const noexcept -> Array<T, S> {
    invariant(ptr != nullptr);
    return Array<T, S>{ptr, this->sz};
  }
#ifndef NDEBUG
  constexpr void extendOrAssertSize(Row MM, Col NN) const {
    ASSERT((MM == numRow() && NN == numCol()));
  }
#else
  static constexpr void extendOrAssertSize(Row, Col) {}
#endif

  [[nodiscard]] constexpr auto deleteCol(ptrdiff_t c) const
    -> ManagedArray<T, S> {
    static_assert(MatrixDimension<S>);
    auto newDim = dim().similar(numRow() - 1);
    ManagedArray<T, decltype(newDim)> A(newDim);
    for (ptrdiff_t m = 0; m < numRow(); ++m) {
      A(m, _(0, c)) = (*this)(m, _(0, c));
      A(m, _(c, math::end)) = (*this)(m, _(c + 1, math::end));
    }
    return A;
  }
  [[nodiscard]] constexpr auto operator==(const Array &other) const noexcept
    -> bool {
    if (size() != other.size()) return false;
    if constexpr (std::is_same_v<S, StridedDims>) {
      // may not be dense, iterate over rows
      for (Row i = 0; i < numRow(); ++i)
        if ((*this)(i, _) != other(i, _)) return false;
      return true;
    }
    return std::equal(begin(), end(), other.begin());
  }
  [[nodiscard]] constexpr auto operator<(const Array &other) const noexcept
    -> bool {
    static_assert(std::integral<S>);
    return std::lexicographical_compare(begin(), end(), other.begin(),
                                        other.end());
  }
  [[nodiscard]] constexpr auto operator>(const Array &other) const noexcept
    -> bool {
    return other < *this;
  }
  [[nodiscard]] constexpr auto operator>=(const Array &other) const noexcept
    -> bool {
    return !(*this < other);
  }
  [[nodiscard]] constexpr auto operator<=(const Array &other) const noexcept
    -> bool {
    return !(*this > other);
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
        for (ptrdiff_t i = 0; i < Row{sz}; ++i) {
          if (i) (void)std::fprintf(f, "\n");
          (void)std::fprintf(f, "%ld", int64_t((*this)(i, 0)));
          for (ptrdiff_t j = 1; j < Col{sz}; ++j)
            (void)std::fprintf(f, " %ld", int64_t((*this)(i, j)));
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

template <class T, class S>
struct POLY_MATH_GSL_POINTER MutArray : Array<T, S>,
                                        ArrayOps<T, S, MutArray<T, S>> {
  using BaseT = Array<T, S>;
  // using BaseT::BaseT;
  using BaseT::operator[], BaseT::operator(), BaseT::data, BaseT::begin,
    BaseT::end, BaseT::rbegin, BaseT::rend, BaseT::front, BaseT::back;

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
    } else if constexpr (std::is_same_v<S, StridedDims>) {
      invariant(nz.row() <= oz.row());
      invariant(nz.col() <= oz.col());
    } else {
      static_assert(MatrixDimension<S>, "Can only resize 1 or 2d containers.");
      auto newX = unsigned{RowStride{nz}}, oldX = unsigned{RowStride{oz}},
           newN = unsigned{Col{nz}}, oldN = unsigned{Col{oz}},
           newM = unsigned{Row{nz}}, oldM = unsigned{Row{oz}};
      invariant(newM <= oldM);
      invariant(newN <= oldN);
      invariant(newX <= oldX);
      unsigned colsToCopy = newN;
      // we only need to copy if memory shifts position
      bool copyCols = ((colsToCopy > 0) && (newX != oldX));
      // if we're in place, we have 1 less row to copy
      unsigned rowsToCopy = newM;
      if (rowsToCopy && (--rowsToCopy) && (copyCols)) {
        // truncation, we need to copy rows to increase stride
        T *src = data(), *dst = src;
        do {
          src += oldX;
          dst += newX;
          std::copy_n(src, colsToCopy, dst);
        } while (--rowsToCopy);
      }
    }
  }

  constexpr void truncate(Row r) {
    if constexpr (std::integral<S>) {
      return truncate(S(r));
    } else if constexpr (std::is_same_v<S, StridedDims>) {
      invariant(r <= Row{this->sz});
      this->sz.set(r);
    } else { // if constexpr (std::is_same_v<S, DenseDims>) {
      static_assert(std::is_same_v<S, DenseDims>,
                    "if truncating a row, matrix must be strided or dense.");
      invariant(r <= Row{this->sz});
      DenseDims newSz = this->sz;
      truncate(newSz.set(r));
    }
  }
  constexpr void truncate(Col c) {
    if constexpr (std::integral<S>) {
      return truncate(S(c));
    } else if constexpr (std::is_same_v<S, StridedDims>) {
      invariant(c <= Col{this->sz});
      this->sz.set(c);
    } else { // if constexpr (std::is_same_v<S, DenseDims>) {
      static_assert(std::is_same_v<S, DenseDims>,
                    "if truncating a col, matrix must be strided or dense.");
      invariant(c <= Col{this->sz});
      DenseDims newSz = this->sz;
      truncate(newSz.set(c));
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
  [[nodiscard]] constexpr auto wrappedPtr() noexcept -> NotNull<T> {
    return data();
  }

  [[nodiscard]] constexpr auto begin() noexcept {
    T *p = const_cast<T *>(this->ptr);
    if constexpr (std::is_same_v<S, StridedRange>)
      return StridedIterator{p, this->sz.stride};
    else return p;
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
  constexpr auto operator[](Index<S> auto i) noexcept -> decltype(auto) {
    auto offset = calcOffset(this->sz, i);
    auto newDim = calcNewDim(this->sz, i);
    if constexpr (std::is_same_v<decltype(newDim), Empty>)
      return data()[offset];
    else return MutArray<T, decltype(newDim)>{data() + offset, newDim};
  }
  // TODO: switch to operator[] when we enable c++23
  template <class R, class C>
  constexpr auto operator()(R r, C c) noexcept -> decltype(auto) {
    if constexpr (MatrixDimension<S>)
      return (*this)[CartesianIndex<R, C>{r, c}];
    else return (*this)[ptrdiff_t(r)];
  }
  constexpr void fill(T value) {
    std::fill_n(this->data(), ptrdiff_t(this->dim()), value);
  }
  [[nodiscard]] constexpr auto diag() noexcept {
    StridedRange r{unsigned(min(Row{this->sz}, Col{this->sz})),
                   unsigned(RowStride{this->sz}) + 1};
    return MutArray<T, StridedRange>{data(), r};
  }
  [[nodiscard]] constexpr auto antiDiag() noexcept {
    Col c = Col{this->sz};
    StridedRange r{unsigned(min(Row{this->sz}, c)),
                   unsigned(RowStride{this->sz}) - 1};
    return MutArray<T, StridedRange>{data() + ptrdiff_t(c) - 1, r};
  }
  constexpr void erase(S i) {
    static_assert(std::integral<S>, "erase requires integral size");
    S oldLen = this->sz--;
    if (i < this->sz)
      std::copy(this->data() + i + 1, data() + oldLen, this->data() + i);
  }
  constexpr void erase(Row r) {
    if constexpr (std::integral<S>) {
      return erase(S(r));
    } else if constexpr (std::is_same_v<S, StridedDims>) {

      auto stride = unsigned{RowStride{this->sz}},
           col = unsigned{Col{this->sz}}, newRow = unsigned{Row{this->sz}} - 1;
      this->sz.set(Row{newRow});
      if ((col == 0) || (r == newRow)) return;
      invariant(col <= stride);
      if ((col + (512 / (sizeof(T)))) <= stride) {
        T *dst = data() + r * stride;
        for (ptrdiff_t m = *r; m < newRow; ++m) {
          T *src = dst + stride;
          std::copy_n(src, col, dst);
          dst = src;
        }
      } else {
        T *dst = data() + r * stride;
        std::copy_n(dst + stride, (newRow - unsigned(r)) * stride, dst);
      }
    } else { // if constexpr (std::is_same_v<S, DenseDims>) {
      static_assert(std::is_same_v<S, DenseDims>,
                    "if erasing a row, matrix must be strided or dense.");
      auto col = unsigned{Col{this->sz}}, newRow = unsigned{Row{this->sz}} - 1;
      this->sz.set(Row{newRow});
      if ((col == 0) || (r == newRow)) return;
      T *dst = data() + r * col;
      std::copy_n(dst + col, (newRow - unsigned(r)) * col, dst);
    }
  }
  constexpr void erase(Col c) {
    if constexpr (std::integral<S>) {
      return erase(S(c));
    } else if constexpr (std::is_same_v<S, StridedDims>) {
      auto stride = unsigned{RowStride{this->sz}},
           newCol = unsigned{Col{this->sz}} - 1, row = unsigned{Row{this->sz}};
      this->sz.set(Col{newCol});
      unsigned colsToCopy = newCol - unsigned(c);
      if ((colsToCopy == 0) || (row == 0)) return;
      // we only need to copy if memory shifts position
      for (ptrdiff_t m = 0; m < row; ++m) {
        T *dst = data() + m * stride + unsigned(c);
        std::copy_n(dst + 1, colsToCopy, dst);
      }
    } else { // if constexpr (std::is_same_v<S, DenseDims>) {
      static_assert(std::is_same_v<S, DenseDims>,
                    "if erasing a col, matrix must be strided or dense.");
      auto newCol = unsigned{Col{this->sz}}, oldCol = newCol--,
           row = unsigned{Row{this->sz}};
      this->sz.set(Col{newCol});
      unsigned colsToCopy = newCol - unsigned(c);
      if ((colsToCopy == 0) || (row == 0)) return;
      // we only need to copy if memory shifts position
      for (ptrdiff_t m = 0; m < row; ++m) {
        T *dst = data() + m * newCol + unsigned(c);
        T *src = data() + m * oldCol + unsigned(c) + 1;
        std::copy_n(src, colsToCopy, dst);
      }
    }
  }
  constexpr void moveLast(Col j) {
    static_assert(MatrixDimension<S>);
    if (j == this->numCol()) return;
    Col Nd = this->numCol() - 1;
    for (ptrdiff_t m = 0; m < this->numRow(); ++m) {
      auto x = (*this)(m, j);
      for (Col n = j; n < Nd;) {
        Col o = n++;
        (*this)(m, o) = (*this)(m, n);
      }
      (*this)(m, Nd) = x;
    }
  }
};

template <typename T, typename S> MutArray(T *, S) -> MutArray<T, S>;

template <typename T, typename S> MutArray(MutArray<T, S>) -> MutArray<T, S>;

static_assert(std::convertible_to<MutArray<int64_t, SquareDims>,
                                  MutArray<int64_t, DenseDims>>);
static_assert(std::convertible_to<MutArray<int64_t, SquareDims>,
                                  MutArray<int64_t, StridedDims>>);
static_assert(std::convertible_to<MutArray<int64_t, DenseDims>,
                                  MutArray<int64_t, StridedDims>>);

/// Non-owning view of a managed array, capable of resizing,
/// but not of re-allocating in case the capacity is exceeded.
template <class T, class S,
          std::unsigned_integral U = default_capacity_type_t<S>>
struct POLY_MATH_GSL_POINTER ResizeableView : MutArray<T, S> {
  using BaseT = MutArray<T, S>;

  constexpr ResizeableView() noexcept : BaseT(nullptr, 0), capacity(0) {}
  constexpr ResizeableView(T *p, S s, U c) noexcept
    : BaseT(p, s), capacity(c) {}
  constexpr ResizeableView(utils::Arena<> *a, S s, U c) noexcept
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
  constexpr auto emplace_backa(utils::Arena<> *alloc, Args &&...args)
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
  constexpr void push_back(utils::Arena<> *alloc, T value) {
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
      auto newX = unsigned{RowStride{nz}}, oldX = unsigned{RowStride{oz}},
           newN = unsigned{Col{nz}}, oldN = unsigned{Col{oz}},
           newM = unsigned{Row{nz}}, oldM = unsigned{Row{oz}};
      invariant(U(nz) <= capacity);
      U len = U(nz);
      T *npt = this->data();
      // we can copy forward so long as the new stride is smaller
      // so that the start of the dst range is outside of the src range
      // we can also safely forward copy if we allocated a new ptr
      bool forwardCopy = (newX <= oldX);
      unsigned colsToCopy = std::min(oldN, newN);
      // we only need to copy if memory shifts position
      bool copyCols = ((colsToCopy > 0) && (newX != oldX));
      // if we're in place, we have 1 less row to copy
      unsigned rowsToCopy = std::min(oldM, newM) - 1;
      unsigned fillCount = newN - colsToCopy;
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

  constexpr void resize(Row r) {
    if constexpr (std::integral<S>) {
      return resize(S(r));
    } else if constexpr (MatrixDimension<S>) {
      S nz = this->sz;
      return resize(nz.set(r));
    }
  }
  constexpr void resize(Col c) {
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
  constexpr void resizeForOverwrite(Row r) {
    if constexpr (std::integral<S>) {
      return resizeForOverwrite(S(r));
    } else if constexpr (MatrixDimension<S>) {
      S nz = this->sz;
      return resizeForOverwrite(nz.set(r));
    }
  }
  constexpr void resizeForOverwrite(Col c) {
    if constexpr (std::integral<S>) {
      return resizeForOverwrite(S(c));
    } else if constexpr (MatrixDimension<S>) {
      S nz = this->sz;
      return resizeForOverwrite(nz.set(c));
    }
  }
  [[nodiscard]] constexpr auto getCapacity() const -> U { return capacity; }

  // set size and 0.
  constexpr void setSize(Row r, Col c) {
    resizeForOverwrite({r, c});
    this->fill(0);
  }
  constexpr void resize(Row MM, Col NN) { resize(DenseDims{MM, NN}); }
  constexpr void resizeForOverwrite(Row M, Col N, RowStride X) {
    invariant(X >= N);
    if constexpr (std::is_same_v<S, StridedDims>) resizeForOverwrite({M, N, X});
    else if constexpr (std::is_same_v<S, SquareDims>) {
      invariant(*M == *N);
      resizeForOverwrite({*M});
    } else resizeForOverwrite({*M, *N});
  }
  constexpr void resizeForOverwrite(Row M, Col N) {
    if constexpr (std::is_same_v<S, StridedDims>)
      resizeForOverwrite({M, N, *N});
    else if constexpr (std::is_same_v<S, SquareDims>) {
      invariant(*M == *N);
      resizeForOverwrite({*M});
    } else resizeForOverwrite({*M, *N});
  }

  constexpr void extendOrAssertSize(Row R, Col C) {
    resizeForOverwrite(DenseDims{R, C});
  }
  constexpr auto insert(T *p, T x) -> T * {
    static_assert(std::is_same_v<S, unsigned>);
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
  constexpr void reserve(utils::Arena<SlabSize, BumpUp> *alloc, U newCapacity) {
    if (newCapacity <= capacity) return;
    this->ptr = alloc->template reallocate<false, T>(
      const_cast<T *>(this->ptr), capacity, newCapacity, U{this->sz});
    // T *oldPtr =
    //   std::exchange(this->data(), alloc->template allocate<T>(newCapacity));
    // std::copy_n(oldPtr, U(this->sz), this->data());
    // alloc->deallocate(oldPtr, capacity);
    capacity = newCapacity;
  }

protected:
  // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
  [[no_unique_address]] U capacity{0};
};

/// Non-owning view of a managed array, capable of reallocating, etc.
/// It does not own memory. Mostly, it serves to drop the inlined
/// stack capacity of the `ManagedArray` from the type.
template <class T, class S, class P, class A = std::allocator<T>,

          std::unsigned_integral U = default_capacity_type_t<S>>
struct POLY_MATH_GSL_POINTER ReallocView : ResizeableView<T, S, U> {
  using BaseT = ResizeableView<T, S, U>;

  constexpr ReallocView(T *p, S s, U c) noexcept : BaseT(p, s, c) {}
  constexpr ReallocView(T *p, S s, U c, A alloc) noexcept
    : BaseT(p, s, c), allocator(alloc) {}

  [[nodiscard]] constexpr auto newCapacity() const -> U {
    return static_cast<const P *>(this)->newCapacity();
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
    if (this->sz == this->capacity) [[unlikely]]
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
#ifdef __cpp_lib_allocate_at_least
        std::allocation_result res = allocator.allocate_at_least(newCapacity);
        T *newPtr = res.ptr;
        newCapacity = U(res.count);
#else
        T *newPtr = allocator.allocate(newCapacity);
#endif
        if (oz) std::copy_n(this->data(), oz, newPtr);
        maybeDeallocate(newPtr, newCapacity);
        invariant(newCapacity > oz);
      }
      std::fill(this->data() + oz, this->data() + nz, T{});
    } else {
      static_assert(MatrixDimension<S>, "Can only resize 1 or 2d containers.");
      U len = U(nz);
      if (len == 0) return;
      auto newX = unsigned{RowStride{nz}}, oldX = unsigned{RowStride{oz}},
           newN = unsigned{Col{nz}}, oldN = unsigned{Col{oz}},
           newM = unsigned{Row{nz}}, oldM = unsigned{Row{oz}};
      bool newAlloc = len > this->capacity;
      bool inPlace = !newAlloc;
#ifdef __cpp_lib_allocate_at_least
      T *npt = this->data();
      if (newAlloc) {
        std::allocation_result res = allocator.allocate_at_least(len);
        npt = res.ptr;
        len = U(res.count);
      }
#else
      T *npt = newAlloc ? allocator.allocate(len) : this->data();
#endif
      // we can copy forward so long as the new stride is smaller
      // so that the start of the dst range is outside of the src range
      // we can also safely forward copy if we allocated a new ptr
      bool forwardCopy = (newX <= oldX) || newAlloc;
      unsigned colsToCopy = std::min(oldN, newN);
      // we only need to copy if memory shifts position
      bool copyCols = newAlloc || ((colsToCopy > 0) && (newX != oldX));
      // if we're in place, we have 1 less row to copy
      unsigned rowsToCopy = std::min(oldM, newM);
      unsigned fillCount = newN - colsToCopy;
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
  constexpr void resize(Row r) {
    if constexpr (std::integral<S>) {
      return resize(S(r));
    } else if constexpr (MatrixDimension<S>) {
      S nz = this->sz;
      return resize(nz.set(r));
    }
  }
  constexpr void resize(Col c) {
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
  constexpr void resizeForOverwrite(Row r) {
    if constexpr (std::integral<S>) {
      return resizeForOverwrite(S(r));
    } else if constexpr (MatrixDimension<S>) {
      S nz = this->sz;
      return resizeForOverwrite(nz.set(r));
    }
  }
  constexpr void resizeForOverwrite(Col c) {
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
#ifdef __cpp_lib_allocate_at_least
    std::allocation_result res = allocator.allocate_at_least(newCapacity);
    T *newPtr = res.ptr;
    newCapacity = U(res.size);
#else
    T *newPtr = allocator.allocate(newCapacity);
#endif
    if (U oldLen = U(this->sz))
      std::uninitialized_copy_n(this->data(), oldLen, newPtr);
    maybeDeallocate(newPtr, newCapacity);
  }
  [[nodiscard]] constexpr auto get_allocator() const noexcept -> A {
    return allocator;
  }
  // set size and 0.
  constexpr void setSize(Row r, Col c) {
    resizeForOverwrite({r, c});
    this->fill(0);
  }
  constexpr void resize(Row MM, Col NN) { resize(DenseDims{MM, NN}); }
  constexpr void reserve(Row M, Col N) {
    if constexpr (std::is_same_v<S, StridedDims>)
      reserve(StridedDims{M, N, max(N, RowStride{this->dim()})});
    else if constexpr (std::is_same_v<S, SquareDims>)
      reserve(SquareDims{unsigned(std::max(*M, *N))});
    else reserve(DenseDims{M, N});
  }
  constexpr void reserve(Row M, RowStride X) {
    if constexpr (std::is_same_v<S, StridedDims>)
      reserve(StridedDims{*M, *X, *X});
    else if constexpr (std::is_same_v<S, SquareDims>)
      reserve(SquareDims{unsigned(std::max(*M, *X))});
    else reserve(DenseDims{*M, *X});
  }
  constexpr void clearReserve(Row M, Col N) {
    this->clear();
    reserve(M, N);
  }
  constexpr void clearReserve(Row M, RowStride X) {
    this->clear();
    reserve(M, X);
  }
  constexpr void resizeForOverwrite(Row M, Col N, RowStride X) {
    invariant(X >= N);
    if constexpr (std::is_same_v<S, StridedDims>) resizeForOverwrite({M, N, X});
    else if constexpr (std::is_same_v<S, SquareDims>) {
      invariant(*M == *N);
      resizeForOverwrite({*M});
    } else resizeForOverwrite({*M, *N});
  }
  constexpr void resizeForOverwrite(Row M, Col N) {
    if constexpr (std::is_same_v<S, StridedDims>)
      resizeForOverwrite({M, N, *N});
    else if constexpr (std::is_same_v<S, SquareDims>) {
      invariant(*M == *N);
      resizeForOverwrite({*M});
    } else resizeForOverwrite({*M, *N});
  }

  constexpr void extendOrAssertSize(Row R, Col C) {
    resizeForOverwrite(DenseDims{R, C});
  }

protected:
  // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
  [[no_unique_address]] A allocator{};

  constexpr void allocateAtLeast(U len) {
#ifdef __cpp_lib_allocate_at_least
    std::allocation_result res = allocator.allocate_at_least(len);
    this->ptr = res.ptr;
    this->capacity = res.count;
#else
    this->ptr = allocator.allocate(len);
    this->capacity = len;
#endif
  }
  [[nodiscard]] constexpr auto isSmall() const -> bool {
    return static_cast<const P *>(this)->isSmall();
  }
  [[nodiscard]] constexpr auto getmemptr() const -> const T * {
    return static_cast<const P *>(this)->getmemptr();
  }
  [[nodiscard]] constexpr auto wasAllocated() const -> bool {
    return static_cast<const P *>(this)->wasAllocated();
  }
  // this method should only be called from the destructor
  // (and the implementation taking the new ptr and capacity)
  constexpr void maybeDeallocate() noexcept {
    if (wasAllocated()) allocator.deallocate(this->data(), this->capacity);
  }
  // this method should be called whenever the buffer lives
  constexpr void maybeDeallocate(T *newPtr, U newCapacity) noexcept {
    maybeDeallocate();
    this->ptr = newPtr;
    this->capacity = newCapacity;
  }
  // grow, discarding old data
  constexpr void growUndef(U M) {
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
template <class T, class S, size_t N, class A, std::unsigned_integral U>
struct POLY_MATH_GSL_OWNER ManagedArray
  : ReallocView<T, S, ManagedArray<T, S, N, A, U>, A, U> {
  static_assert(std::is_trivially_destructible_v<T>);
  using BaseT = ReallocView<T, S, ManagedArray<T, S, N, A, U>, A, U>;
  // We're deliberately not initializing storage.
#if !defined(__clang__) && defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
#endif
  constexpr ManagedArray() noexcept : BaseT{memory.data(), S{}, N} {
#ifndef NDEBUG
    if (!N) return;
    if constexpr (std::numeric_limits<T>::has_signaling_NaN)
      std::fill_n(this->data(), N, std::numeric_limits<T>::signaling_NaN());
    else if constexpr (std::numeric_limits<T>::is_specialized)
      std::fill_n(this->data(), N, std::numeric_limits<T>::min());
#endif
  }
  constexpr ManagedArray(S s) noexcept : BaseT{memory.data(), s, N} {
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
  constexpr ManagedArray(S s, T x) noexcept : BaseT{memory.data(), s, N} {
    U len = U(this->sz);
    if (len > N) this->allocateAtLeast(len);
    if (len) std::fill_n(this->data(), len, x);
  }
  template <class D, std::unsigned_integral I>
  constexpr ManagedArray(const ManagedArray<T, D, N, A, I> &b) noexcept
    : BaseT{memory.data(), S(b.dim()), U(N), b.get_allocator()} {
    U len = U(this->sz);
    this->growUndef(len);
    std::copy_n(b.data(), len, this->data());
  }
  template <std::convertible_to<T> Y, class D, class AY,
            std::unsigned_integral I>
  constexpr ManagedArray(const ManagedArray<Y, D, N, AY, I> &b) noexcept
    : BaseT{memory.data(), S(b.dim()), U(N), b.get_allocator()} {
    U len = U(this->sz);
    this->growUndef(len);
    if constexpr (DenseLayout<D> && DenseLayout<S>) {
      std::copy_n(b.data(), len, this->data());
    } else if constexpr (MatrixDimension<D> && MatrixDimension<S>) {
      invariant(b.numRow() == this->numRow());
      invariant(b.numCol() == this->numCol());
      for (ptrdiff_t m = 0; m < this->numRow(); ++m) {
        POLYMATHVECTORIZE
        for (ptrdiff_t n = 0; n < this->numCol(); ++n) (*this)(m, n) = b(m, n);
      }
    } else if constexpr (MatrixDimension<D>) {
      ptrdiff_t j = 0;
      for (ptrdiff_t m = 0; m < b.numRow(); ++m) {
        POLYMATHVECTORIZE
        for (ptrdiff_t n = 0; n < b.numCol(); ++n) (*this)(j++) = b(m, n);
      }
    } else if constexpr (MatrixDimension<S>) {
      ptrdiff_t j = 0;
      for (ptrdiff_t m = 0; m < this->numRow(); ++m) {
        POLYMATHVECTORIZE
        for (ptrdiff_t n = 0; n < this->numCol(); ++n) (*this)(m, n) = b(j++);
      }
    } else {
      T *p = this->data();
      POLYMATHVECTORIZE
      for (ptrdiff_t i = 0; i < len; ++i) p[i] = b[i];
    }
  }
  template <std::convertible_to<T> Y>
  constexpr ManagedArray(std::initializer_list<Y> il) noexcept
    : BaseT{memory.data(), S(il.size()), U(N)} {
    U len = U(this->sz);
    this->growUndef(len);
    std::copy_n(il.begin(), len, this->data());
  }
  template <std::convertible_to<T> Y, class D, class AY,
            std::unsigned_integral I>
  constexpr ManagedArray(const ManagedArray<Y, D, N, AY, I> &b, S s) noexcept
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
    : BaseT{memory.data(), S(b.size()), U(N)} {
    U len = U(this->sz);
    this->growUndef(len);
    (*this) << b;
  }
  template <class D, std::unsigned_integral I>
  constexpr ManagedArray(ManagedArray<T, D, N, A, I> &&b) noexcept
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
  template <class D, std::unsigned_integral I>
  constexpr ManagedArray(ManagedArray<T, D, N, A, I> &&b, S s) noexcept
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
        (*this)(i, j++) = T(B.getNonZeros()[k++]);
      }
    }
    invariant(k == B.getNonZeros().size());
  }
#if !defined(__clang__) && defined(__GNUC__)
#pragma GCC diagnostic pop
#else
#pragma clang diagnostic pop
#endif

  template <class D, std::unsigned_integral I>
  constexpr auto operator=(const ManagedArray<T, D, N, A, I> &b) noexcept
    -> ManagedArray & {
    if (this == &b) return *this;
    this->sz = b.dim();
    U len = U(this->sz);
    this->growUndef(len);
    std::copy_n(b.data(), len, this->data());
    return *this;
  }
  template <class D, std::unsigned_integral I>
  constexpr auto operator=(ManagedArray<T, D, N, A, I> &&b) noexcept
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

  [[nodiscard]] static constexpr auto identity(unsigned M) -> ManagedArray {
    static_assert(MatrixDimension<S>);
    ManagedArray B(SquareDims{M}, T{0});
    B.diag() << 1;
    return B;
  }
  [[nodiscard]] static constexpr auto identity(Row R) -> ManagedArray {
    static_assert(MatrixDimension<S>);
    return identity(unsigned(R));
  }
  [[nodiscard]] static constexpr auto identity(Col C) -> ManagedArray {
    static_assert(MatrixDimension<S>);
    return identity(unsigned(C));
  }
  friend inline void PrintTo(const ManagedArray &x, ::std::ostream *os) {
    *os << x;
  }
  [[nodiscard]] auto getmemptr() const -> const T * { return memory.data(); }
  [[nodiscard]] constexpr auto newCapacity() const -> U {
    if constexpr (N == 0)
      return this->capacity == 0 ? U{4} : 2 * this->capacity;
    else return 2 * this->capacity;
  }
  [[nodiscard]] constexpr auto wasAllocated() const -> bool {
    if constexpr (N == 0) return this->ptr != nullptr;
    else return !isSmall();
  }

private:
  [[no_unique_address]] containers::Storage<T, N> memory;
};

static_assert(std::move_constructible<ManagedArray<intptr_t, unsigned>>);
static_assert(std::copyable<ManagedArray<intptr_t, unsigned>>);
// Check that `[[no_unique_address]]` is working.
// sizes should be:
// [ptr, dims, capacity, allocator, array]
// 8 + 3*4 + 4 + 0 + 64*8 = 24 + 512 = 536
static_assert(
  sizeof(ManagedArray<int64_t, StridedDims, 64, std::allocator<int64_t>>) ==
  536);
// sizes should be:
// [ptr, dims, capacity, allocator, array]
// 8 + 2*4 + 8 + 0 + 64*8 = 24 + 512 = 536
static_assert(
  sizeof(ManagedArray<int64_t, DenseDims, 64, std::allocator<int64_t>>) == 536);
// sizes should be:
// [ptr, dims, capacity, allocator, array]
// 8 + 1*4 + 4 + 0 + 64*8 = 16 + 512 = 528
static_assert(
  sizeof(ManagedArray<int64_t, SquareDims, 64, std::allocator<int64_t>>) ==
  528);

template <class T, size_t N = PreAllocStorage<T>()>
using Vector = ManagedArray<T, unsigned, N>;
template <class T> using PtrVector = Array<T, unsigned>;
template <class T> using MutPtrVector = MutArray<T, unsigned>;

static_assert(std::move_constructible<Vector<intptr_t>>);
static_assert(std::copy_constructible<Vector<intptr_t>>);
static_assert(std::copyable<Vector<intptr_t>>);
static_assert(AbstractVector<Array<int64_t, unsigned>>);
static_assert(AbstractVector<MutArray<int64_t, unsigned>>);
static_assert(AbstractVector<Vector<int64_t>>);
static_assert(!AbstractVector<int64_t>);
static_assert(!std::is_trivially_copyable_v<Vector<int64_t>>);
static_assert(!std::is_trivially_destructible_v<Vector<int64_t>>);

template <typename T> using StridedVector = Array<T, StridedRange>;
template <typename T> using MutStridedVector = MutArray<T, StridedRange>;

static_assert(AbstractVector<StridedVector<int64_t>>);
static_assert(AbstractVector<MutStridedVector<int64_t>>);
static_assert(std::is_trivially_copyable_v<StridedVector<int64_t>>);

template <class T> using PtrMatrix = Array<T, StridedDims>;
template <class T> using MutPtrMatrix = MutArray<T, StridedDims>;
template <class T, size_t L = 64>
using Matrix = ManagedArray<T, StridedDims, L>;
template <class T> using DensePtrMatrix = Array<T, DenseDims>;
template <class T> using MutDensePtrMatrix = MutArray<T, DenseDims>;
template <class T, size_t L = 64>
using DenseMatrix = ManagedArray<T, DenseDims, L>;
template <class T> using SquarePtrMatrix = Array<T, SquareDims>;
template <class T> using MutSquarePtrMatrix = MutArray<T, SquareDims>;
template <class T, size_t L = PreAllocSquareStorage<T>()>
using SquareMatrix = ManagedArray<T, SquareDims, L>;

static_assert(sizeof(PtrMatrix<int64_t>) ==
              4 * sizeof(unsigned int) + sizeof(int64_t *));
static_assert(sizeof(MutPtrMatrix<int64_t>) ==
              4 * sizeof(unsigned int) + sizeof(int64_t *));
static_assert(sizeof(DensePtrMatrix<int64_t>) ==
              2 * sizeof(unsigned int) + sizeof(int64_t *));
static_assert(sizeof(MutDensePtrMatrix<int64_t>) ==
              2 * sizeof(unsigned int) + sizeof(int64_t *));
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
static_assert(sizeof(ManagedArray<int32_t, DenseDims, 15>) ==
              sizeof(int32_t *) + 4 * sizeof(unsigned int) +
                16 * sizeof(int32_t));
static_assert(sizeof(ReallocView<int32_t, DenseDims, DenseMatrix<int32_t>>) ==
              sizeof(int32_t *) + 4 * sizeof(unsigned int));

static_assert(!AbstractVector<PtrMatrix<int64_t>>,
              "PtrMatrix<int64_t> isa AbstractVector succeeded");
static_assert(!AbstractVector<MutPtrMatrix<int64_t>>,
              "PtrMatrix<int64_t> isa AbstractVector succeeded");
static_assert(!AbstractVector<const PtrMatrix<int64_t>>,
              "PtrMatrix<int64_t> isa AbstractVector succeeded");

static_assert(AbstractMatrix<PtrMatrix<int64_t>>,
              "PtrMatrix<int64_t> isa AbstractMatrix failed");
static_assert(std::same_as<decltype(PtrMatrix<int64_t>(nullptr, Row{0}, Col{0})(
                             ptrdiff_t(0), ptrdiff_t(0))),
                           const int64_t &>);
static_assert(
  std::same_as<std::remove_reference_t<decltype(MutPtrMatrix<int64_t>(
                 nullptr, Row{0}, Col{0})(ptrdiff_t(0), ptrdiff_t(0)))>,
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

using IntMatrix = Matrix<int64_t>;
static_assert(std::same_as<IntMatrix::value_type, int64_t>);
static_assert(AbstractMatrix<IntMatrix>);
static_assert(std::copyable<IntMatrix>);
static_assert(std::same_as<utils::eltype_t<Matrix<int64_t>>, int64_t>);

static_assert(
  std::convertible_to<Array<int64_t, SquareDims>, Array<int64_t, StridedDims>>);

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
  Vector<ptrdiff_t> maxDigits{unsigned(N), 0};
  invariant(ptrdiff_t(maxDigits.size()), N);
  // this is slow, because we count the digits of every element
  // we could optimize this by reducing the number of calls to countDigits
  for (Row i = 0; i < M; i++) {
    for (ptrdiff_t j = 0; j < N; j++) {
      ptrdiff_t c = countDigits(A(i, j));
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
  Vector<T> maxDigits{unsigned(N), T{}};
  invariant(ptrdiff_t(maxDigits.size()), N);
  // first, we find the digits with the maximum value per column
  for (Row i = 0; i < M; i++) {
    for (ptrdiff_t j = 0; j < N; j++) {
      // negative numbers need one more digit
      // first, we find the maximum value per column,
      // dividing positive numbers by -10
      T Aij = A(i, j);
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
  auto [M, N] = A.size();
  if ((!M) || (!N)) return os << "[ ]";
  // first, we determine the number of digits needed per column
  auto maxDigits{getMaxDigits(A)};
  using U = decltype(countDigits(std::declval<T>()));
  for (Row i = 0; i < M; i++) {
    if (i) os << "  ";
    else os << "\n[ ";
    for (ptrdiff_t j = 0; j < N; j++) {
      auto Aij = A(i, j);
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
  auto [M, N] = A.size();
  if ((!M) || (!N)) return os << "[ ]";
  // first, we determine the number of digits needed per column
  Vector<char, 512> digits;
  digits.resizeForOverwrite(512);
  // we can't have more than 255 digits
  DenseMatrix<uint8_t> numDigits{DenseDims{M, N}};
  char *ptr = digits.begin();
  char *pEnd = digits.end();
  for (ptrdiff_t m = 0; m < M; m++) {
    for (ptrdiff_t n = 0; n < N; n++) {
      auto Aij = A(m, n);
      while (true) {
        auto [p, ec] = std::to_chars(ptr, pEnd, Aij);
        if (ec == std::errc()) [[likely]] {
          numDigits(m, n) = std::distance(ptr, p);
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
  maxDigits << numDigits(0, _);
  for (ptrdiff_t m = 0; m < M; m++)
    for (ptrdiff_t n = 0; n < N; n++)
      maxDigits[n] = std::max(maxDigits[n], numDigits(m, n));

  ptr = digits.begin();
  // we will allocate 512 bytes at a time
  for (Row i = 0; i < M; i++) {
    if (i) os << "  ";
    else os << "\n[ ";
    for (ptrdiff_t j = 0; j < N; j++) {
      ptrdiff_t nD = numDigits(i, j);
      for (ptrdiff_t k = 0; k < maxDigits[j] - nD; k++) os << " ";
      os << std::string_view(ptr, nD);
      if (j != ptrdiff_t(N) - 1) os << " ";
      else if (i != ptrdiff_t(M) - 1) os << "\n";
      ptr += nD;
    }
  }
  return os << " ]";
}

template <typename T>
inline auto operator<<(std::ostream &os, PtrMatrix<T> A) -> std::ostream & {
  return printMatrix(os, A);
}
template <typename T>
inline auto operator<<(std::ostream &os, Array<T, SquareDims> A)
  -> std::ostream & {
  return printMatrix(os, PtrMatrix<T>{A});
}
template <typename T>
inline auto operator<<(std::ostream &os, Array<T, DenseDims> A)
  -> std::ostream & {
  return printMatrix(os, PtrMatrix<T>{A});
}
} // namespace poly::math
