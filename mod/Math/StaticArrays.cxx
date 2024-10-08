#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#include "LoopMacros.hxx"
#include "Owner.hxx"

#ifndef USE_MODULE
#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <utility>

#include "Containers/Pair.cxx"
#include "Math/Array.cxx"
#include "Math/ArrayConcepts.cxx"
#include "Math/AxisTypes.cxx"
#include "Math/ExpressionTemplates.cxx"
#include "Math/MatrixDimensions.cxx"
#include "Math/Ranges.cxx"
#include "SIMD/SIMD.cxx"
#include "Utilities/ArrayPrint.cxx"
#include "Utilities/Reference.cxx"
#include "Utilities/TypeCompression.cxx"
#else
export module StaticArray;

import Array;
import ArrayConcepts;
import ArrayPrint;
import AxisTypes;
import CompressReference;
import ExprTemplates;
import MatDim;
import Pair;
import Range;
import SIMD;
import STL;
import TypeCompression;
#endif

template <typename T, ptrdiff_t L>
consteval auto paddedSize() -> std::array<ptrdiff_t, 2> {
  constexpr ptrdiff_t WF = simd::Width<T>;
  constexpr ptrdiff_t W = L < WF ? ptrdiff_t(std::bit_ceil(uint64_t(L))) : WF;
  constexpr ptrdiff_t N = ((L + W - 1) / W);
  return {N, W};
}
template <typename T, ptrdiff_t L, ptrdiff_t Align>
consteval auto calcPaddedCols() -> ptrdiff_t {
  constexpr ptrdiff_t a = Align / sizeof(T);
  if constexpr (a <= 1) return L;
  else return ((L + a - 1) / a) * a;
}
template <typename T, ptrdiff_t L> consteval auto alignSIMD() -> size_t {
  if constexpr (!simd::SIMDSupported<T>) return alignof(T);
  else return alignof(simd::Vec<simd::VecLen<L, T>, T>);
}

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif

template <class T, ptrdiff_t M, ptrdiff_t N, bool Compress>
using StaticDims = std::conditional_t<
  M == 1, math::Length<N>,
  std::conditional_t<
    Compress || ((N % simd::VecLen<N, T>) == 0), math::DenseDims<M, N>,
    math::StridedDims<M, N, calcPaddedCols<T, N, alignSIMD<T, N>()>()>>>;

static_assert(ptrdiff_t(StridedDims<1, 1, 1>{
                math::StaticDims<int64_t, 1, 1, true>{}}) == 1);
static_assert(ptrdiff_t(StridedDims<>{
                math::StaticDims<int64_t, 1, 1, true>{}}) == 1);

static_assert(AbstractSimilar<PtrVector<int64_t>, Length<4>>);

template <class T, ptrdiff_t M, ptrdiff_t N, bool Compress = false>
struct MATH_GSL_OWNER StaticArray
  : public ArrayOps<T, StaticDims<T, M, N, Compress>,
                    StaticArray<T, M, N, Compress>>,
    Expr<T, StaticArray<T, M, N, Compress>> {
  using storage_type = std::conditional_t<Compress, utils::compressed_t<T>, T>;
  static constexpr ptrdiff_t Align =
    Compress ? alignof(storage_type) : alignSIMD<T, N>();
  static constexpr ptrdiff_t PaddedCols = calcPaddedCols<T, N, Align>();
  static constexpr ptrdiff_t capacity = M * PaddedCols;

  static_assert(alignof(storage_type) <= Align);

  alignas(Align) storage_type memory_[capacity]{};

  using value_type = utils::decompressed_t<T>;
  using reference = decltype(utils::ref((storage_type *)nullptr, 0));
  using const_reference =
    decltype(utils::ref((const storage_type *)nullptr, 0));
  using size_type = ptrdiff_t;
  using difference_type = ptrdiff_t;
  using iterator = storage_type *;
  using const_iterator = const storage_type *;
  using pointer = storage_type *;
  using const_pointer = const storage_type *;
  using concrete = std::true_type;

  // if Compress=false, we have a compressed_t
  // if Compress=true, we should already be compressed
  using compressed_type = StaticArray<T, M, N, true>;
  using decompressed_type = StaticArray<value_type, M, N, false>;
  using S = StaticDims<T, M, N, Compress>;
  using Expr<T, StaticArray<T, M, N, Compress>>::operator==;
  constexpr StaticArray() = default;
  constexpr explicit StaticArray(const T &x) noexcept {
    if consteval {
      for (ptrdiff_t i = 0; i < M * N; ++i) memory_[i] = x;
    } else {
      (*this) << x;
    }
  }
  constexpr explicit StaticArray(
    const std::convertible_to<T> auto &x) noexcept {
    (*this) << x;
  }
  constexpr StaticArray(const StaticArray &) = default;
  constexpr explicit StaticArray(StaticArray &&) noexcept = default;
  constexpr explicit StaticArray(const std::initializer_list<T> &list) {
    if (list.size() == 1) {

      (*this) << *list.begin();
      return;
    }
    invariant(list.size() <= size_t(capacity));
    std::copy_n(list.begin(), list.size(), data());
  }
  template <AbstractSimilar<S> V> constexpr StaticArray(const V &b) noexcept {
    this->vcopyTo(b, detail::CopyAssign{});
  }

  constexpr void compress(compressed_type *p) const
  requires(std::same_as<StaticArray, decompressed_type>)
  {
    *p << *this;
  }
  static constexpr auto decompress(const compressed_type *p) -> StaticArray
  requires(std::same_as<StaticArray, decompressed_type>)
  {
    return StaticArray{*p};
  }
  [[nodiscard]] constexpr auto data() const noexcept -> const storage_type * {
    return static_cast<const storage_type *>(memory_);
  }
  constexpr auto data() noexcept -> storage_type * {
    return static_cast<storage_type *>(memory_);
  }

  constexpr auto operator=(StaticArray const &) -> StaticArray & = default;
  constexpr auto operator=(StaticArray &&) noexcept -> StaticArray & = default;

  [[nodiscard]] constexpr auto begin() const noexcept { return data(); }
  [[nodiscard]] constexpr auto end() const noexcept {
    return begin() + capacity;
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
  // static constexpr auto slice(Valid<T>, Index<S> auto i){
  //   auto
  // }
  constexpr auto operator[](Index<S> auto i) const noexcept -> decltype(auto) {
    return index<T>(data(), S{}, i);
  }
  // for vectors, we just drop the column, essentially broadcasting
  template <class R, class C>
  constexpr auto operator[](R r, C c) const noexcept -> decltype(auto) {
    return index<T>(data(), S{}, r, c);
  }
  [[nodiscard]] static constexpr auto minRowCol() -> ptrdiff_t {
    return std::min(ptrdiff_t(numRow()), ptrdiff_t(numCol()));
  }

  [[nodiscard]] constexpr auto diag() const noexcept {
    StridedRange<> r{length(minRowCol()), RowStride(S{}) + 1};
    auto ptr = data();
    invariant(ptr != nullptr);
    return Array<T, StridedRange<>>{ptr, r};
  }
  [[nodiscard]] constexpr auto antiDiag() const noexcept {
    StridedRange<> r{length(minRowCol()), RowStride(S{}) - 1};
    auto ptr = data();
    invariant(ptr != nullptr);
    return Array<T, StridedRange<>>{ptr + ptrdiff_t(Col(S{})) - 1, r};
  }
  [[nodiscard]] static constexpr auto isSquare() noexcept -> bool {
    return Row(S{}) == Col(S{});
  }

  [[nodiscard]] static constexpr auto checkSquare() -> Optional<ptrdiff_t> {
    if constexpr (M == N) return M;
    else return {};
  }
  [[nodiscard]] static constexpr auto numRow() noexcept -> Row<M> { return {}; }
  [[nodiscard]] static constexpr auto numCol() noexcept -> Col<N> { return {}; }
  static constexpr auto safeRow() -> Row<M> { return {}; }
  static constexpr auto safeCol() -> Col<PaddedCols> { return {}; }
  [[nodiscard]] static constexpr auto
  rowStride() noexcept -> RowStride<PaddedCols> {
    return {};
  }
  [[nodiscard]] static constexpr auto empty() -> bool { return capacity == 0; }
  [[nodiscard]] static constexpr auto
  size() noexcept -> std::integral_constant<ptrdiff_t, M * N> {
    return {};
  }
  [[nodiscard]] static constexpr auto dim() noexcept -> S { return S{}; }
  [[nodiscard]] constexpr auto
  t() const -> Transpose<T, Array<T, S, Compress>> {
    return {*this};
  }
  [[nodiscard]] constexpr auto isExchangeMatrix() const -> bool {
    if constexpr (M == N) {
      for (ptrdiff_t i = 0; i < M; ++i) {
        for (ptrdiff_t j = 0; j < M; ++j)
          if ((*this)(i, j) != (i + j == M - 1)) return false;
      }
    } else return false;
  }
  [[nodiscard]] constexpr auto isDiagonal() const -> bool {
    for (ptrdiff_t r = 0; r < numRow(); ++r)
      for (ptrdiff_t c = 0; c < numCol(); ++c)
        if (r != c && (*this)(r, c) != 0) return false;
    return true;
  }
  [[nodiscard, gnu::always_inline]] constexpr auto
  view() const noexcept -> Array<T, S, Compress && utils::Compressible<T>> {
    return {data(), S{}};
  }
  [[nodiscard, gnu::always_inline]] constexpr auto
  mview() noexcept -> MutArray<T, S, Compress && utils::Compressible<T>> {
    return {data(), S{}};
  }

  [[nodiscard]] constexpr auto begin() noexcept {
    if constexpr (std::is_same_v<S, StridedRange<>>)
      return StridedIterator{data(), S{}.stride};
    else return data();
  }
  [[nodiscard]] constexpr auto end() noexcept { return begin() + capacity; }
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
    return index<T>(data(), S{}, i);
  }
  template <class R, class C>
  [[gnu::flatten, gnu::always_inline]] constexpr auto
  operator[](R r, C c) noexcept -> decltype(auto) {
    return index<T>(data(), S{}, r, c);
  }
  constexpr void fill(T value) {
    std::fill_n(data(), ptrdiff_t(this->dim()), value);
  }
  [[nodiscard]] constexpr auto diag() noexcept {
    StridedRange<> r{length(min(Row(S{}), Col(S{}))), RowStride(S{}) + 1};
    return MutArray<T, StridedRange<>>{data(), r};
  }
  [[nodiscard]] constexpr auto antiDiag() noexcept {
    Col c = Col(S{});
    StridedRange<> r{length(min(Row(S{}), c)), RowStride(S{}) - 1};
    return MutArray<T, StridedRange<>>{data() + ptrdiff_t(c) - 1, r};
  }
  template <typename SHAPE>
  constexpr operator Array<T, SHAPE, Compress && utils::Compressible<T>>() const
  requires(std::convertible_to<S, SHAPE>)
  {
    return {const_cast<T *>(data()), SHAPE(dim())};
  }
  template <typename SHAPE>
  constexpr operator MutArray<T, SHAPE, Compress && utils::Compressible<T>>()
  requires(std::convertible_to<S, SHAPE>)
  {
    return {data(), SHAPE(dim())};
  }
  // template<containers::ConvertibleFrom<S> SHAPE>
  // constexpr auto operator==(Array<T,SHAPE> rhs) const noexcept -> bool {
  //   return std::equal(begin(), end(), rhs.begin());
  // }
  // template <MatrixDimension S>
  // constexpr auto operator==(Array<T,S> rhs) const ->bool
  // (requires((M==1)||(N==1))){
  //   if ((rhs.numRow() != M)||(rhs.numCol() != N)) return false;
  // }
  template <size_t I> constexpr auto get() -> T & { return memory_[I]; }
  template <size_t I> [[nodiscard]] constexpr auto get() const -> const T & {
    return memory_[I];
  }
  constexpr void set(T x, ptrdiff_t r, ptrdiff_t c) { memory_[r * N + c] = x; }
  friend void PrintTo(const StaticArray &x, ::std::ostream *os) {
    *os << x.view();
  }

private:
  friend auto operator<<(std::ostream &os, const StaticArray &x)
    -> std::ostream &requires(utils::Printable<T>) {
    if constexpr (MatrixDimension<S>)
      return utils::printMatrix(os, x.data(), M, N, N);
    else return utils::printVector(os, x.begin(), x.end());
  }
};

template <simd::SIMDSupported T, ptrdiff_t M, ptrdiff_t N>
requires((M * (N + simd::VecLen<N, T> - 1) / simd::VecLen<N, T>) > 1)
struct MATH_GSL_OWNER StaticArray<T, M, N, false>
  : ArrayOps<T, StaticDims<T, M, N, false>, StaticArray<T, M, N, false>>,
    Expr<T, StaticArray<T, M, N, false>> {
  // struct StaticArray<T, M, N, alignof(simd::Vec<simd::VecLen<N, T>, T>)>
  //   : ArrayOps<T, StaticDims<T, M, N, alignof(simd::Vec<simd::VecLen<N, T>,
  //   T>)>,
  //              StaticArray<T, M, N, alignof(simd::Vec<simd::VecLen<N, T>,
  //              T>)>> {

  using value_type = T;
  using reference = T &;
  using const_reference = const T &;
  using size_type = ptrdiff_t;
  using difference_type = ptrdiff_t;
  using iterator = T *;
  using const_iterator = const T *;
  using pointer = T *;
  using const_pointer = const T *;
  using concrete = std::true_type;

  using compressed_type = StaticArray<T, M, N, true>;

  constexpr void compress(compressed_type *p) const { *p << *this; }
  static constexpr auto decompress(const compressed_type *p) -> StaticArray {
    return StaticArray{*p};
  }
  static constexpr ptrdiff_t W = simd::VecLen<N, T>;
  static constexpr ptrdiff_t Align = alignof(simd::Vec<W, T>);
  using S = StaticDims<T, M, N, false>;

  [[nodiscard]] static constexpr auto dim() noexcept -> S { return S{}; }

  static constexpr ptrdiff_t L = (N + W - 1) / W;
  static_assert(L * W == calcPaddedCols<T, N, Align>());
  using V = simd::Vec<W, T>;
  // simd::Vec<W, T> data[M][L];
  V memory_[M * L]{};
  // std::array<std::array<simd::Vec<W, T>, L>, M> data;
  // constexpr operator compressed_type() { return compressed_type{*this}; }
  [[nodiscard, gnu::always_inline]] constexpr auto view() const -> StaticArray
  requires(M *L <= 4)
  {
    return *this;
  }
  [[nodiscard, gnu::always_inline]] constexpr auto
  view() const noexcept -> Array<T, S, false>
  requires(M *L > 4)
  {
    return {data(), S{}};
  }
  [[nodiscard, gnu::always_inline]] constexpr auto
  mview() noexcept -> MutArray<T, S, false> {
    return {data(), S{}};
  }

  constexpr StaticArray() = default;
  constexpr StaticArray(StaticArray const &) = default;
  constexpr StaticArray(StaticArray &&) noexcept = default;
  constexpr explicit StaticArray(const std::initializer_list<T> &list) {
    if (list.size() == 1) {
      (*this) << *list.begin();
      return;
    }
    invariant(list.size() <= L * W);
    size_t count = list.size() * sizeof(T);
    std::memcpy(memory_, list.begin(), count);
    std::memset((char *)memory_ + count, 0, (L * W - list.size()) * sizeof(T));
  }
  constexpr auto data() -> T * { return reinterpret_cast<T *>(memory_); }
  [[nodiscard]] constexpr auto data() const -> const T * {
    return reinterpret_cast<const T *>(memory_);
  }
  [[nodiscard]] constexpr auto begin() -> T *requires(((M == 1) || (N == 1))) {
    return data();
  }

  [[nodiscard]] constexpr auto end() -> T *requires(((M == 1) || (N == 1))) {
    return data() + (M * N);
  }

  [[nodiscard]] constexpr auto begin() const
    -> const T *requires(((M == 1) || (N == 1))) { return data(); }

  [[nodiscard]] constexpr auto end() const
    -> const T *requires(((M == 1) || (N == 1))) { return data() + (M * N); }

  [[nodiscard]] constexpr auto t() const -> Transpose<T, Array<T, S, false>> {
    return {*this};
  }
  template <AbstractSimilar<S> V> constexpr StaticArray(const V &b) noexcept {
    this->vcopyTo(b, detail::CopyAssign{});
  }
  constexpr explicit StaticArray(T x) {
    simd::Vec<W, T> v = simd::vbroadcast<W, T>(x);
    for (ptrdiff_t i = 0; i < M * L; ++i) memory_[i] = v;
  }
  constexpr auto operator=(StaticArray const &) -> StaticArray & = default;
  constexpr auto operator=(StaticArray &&) noexcept -> StaticArray & = default;
  static constexpr auto numRow() -> Row<M> { return {}; }
  static constexpr auto numCol() -> Col<N> { return {}; }
  static constexpr auto safeRow() -> Row<M> { return {}; }
  static constexpr auto safeCol() -> Col<L * W> { return {}; }
  [[nodiscard]] static constexpr auto rowStride() noexcept -> RowStride<L * W> {
    return {};
  }
  [[nodiscard]] static constexpr auto
  size() noexcept -> std::integral_constant<ptrdiff_t, M * N> {
    return {};
  }
  template <typename SHAPE>
  constexpr operator Array<T, SHAPE, false>() const
  requires(std::convertible_to<S, SHAPE>)
  {
    return {const_cast<T *>(data()), SHAPE(dim())};
  }
  template <typename SHAPE>
  constexpr operator MutArray<T, SHAPE, false>()
  requires(std::convertible_to<S, SHAPE>)
  {
    return {data(), SHAPE(dim())};
  }
  template <ptrdiff_t U, typename Mask>
  [[gnu::always_inline]] auto
  operator[](ptrdiff_t i, simd::index::Unroll<U, W, Mask> j) const
    -> simd::Unroll<1, U, W, T> {
    return (*this)[simd::index::Unroll<1>{i}, j];
  }
  template <ptrdiff_t R = 1>
  [[gnu::always_inline]] static constexpr void checkinds(ptrdiff_t i,
                                                         ptrdiff_t j) {
    invariant(i >= 0);
    invariant(i + (R - 1) < M);
    invariant(j >= 0);
    invariant(j < N);
    invariant((j % W) == 0);
  }
  template <ptrdiff_t R, ptrdiff_t C, typename Mask>
  [[gnu::flatten, gnu::always_inline]] auto
  operator[](simd::index::Unroll<R> i, simd::index::Unroll<C, W, Mask> j) const
    -> simd::Unroll<R, C, W, T> {
    checkinds<R>(i.index_, j.index_);
    simd::Unroll<R, C, W, T> ret;
    ptrdiff_t k = j.index_ / W;
    POLYMATHFULLUNROLL
    for (ptrdiff_t r = 0; r < R; ++r) {
      POLYMATHFULLUNROLL
      for (ptrdiff_t u = 0; u < C; ++u)
        ret[r, u] = memory_[(i.index_ + r) * L + k + u];
    }
    return ret;
  }
  template <class R, class C>
  [[gnu::flatten, gnu::always_inline]] constexpr auto
  operator[](R r, C c) noexcept -> decltype(auto) {
    if constexpr (std::integral<R> && std::integral<C>)
      return reinterpret_cast<T *>(memory_ + ptrdiff_t(r) * L)[c];
    else return index<T>(data(), S{}, r, c);
  }
  template <class R, class C>
  [[gnu::flatten, gnu::always_inline]] constexpr auto
  operator[](R r, C c) const noexcept -> decltype(auto) {
    if constexpr (std::integral<R> && std::integral<C>)
      return reinterpret_cast<const T *>(memory_ + ptrdiff_t(r) * L)[c];
    else return index<T>(data(), S{}, r, c);
  }
  constexpr void set(T x, ptrdiff_t r, ptrdiff_t c) {
    if constexpr (W == 1) {
      memory_[L * r + c] = x;
    } else {
      V v = memory_[L * r + c / W];
      using IT = std::conditional_t<sizeof(T) == 8, int64_t, int32_t>;
      v = simd::range<W, IT>() == simd::vbroadcast<W, IT>(c % W)
            ? simd::vbroadcast<W, T>(x)
            : v;
      // v[c % W] = x;
      memory_[L * r + c / W] = v;
    }
  }
  template <ptrdiff_t R, ptrdiff_t C> struct Ref {
    StaticArray *parent_;
    ptrdiff_t i_, j_;
    constexpr auto operator=(simd::Unroll<R, C, W, T> x) -> Ref & {
      checkinds<R>(i_, j_);
      ptrdiff_t k = j_ / W;
      POLYMATHFULLUNROLL
      for (ptrdiff_t r = 0; r < R; ++r) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t u = 0; u < C; ++u)
          parent_->memory_[(i_ + r) * L + k + u] = x[r, u];
      }
      return *this;
    }
    constexpr auto operator=(simd::Vec<W, T> x) -> Ref & {
      checkinds<R>(i_, j_);
      ptrdiff_t k = j_ / W;
      POLYMATHFULLUNROLL
      for (ptrdiff_t r = 0; r < R; ++r) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t u = 0; u < C; ++u)
          parent_->memory_[(i_ + r) * L + k + u] = x;
      }
      return *this;
    }
    constexpr auto operator=(std::convertible_to<T> auto x) -> Ref & {
      *this = simd::Vec<W, T>{} + T(x);
      return *this;
    }
    constexpr operator simd::Unroll<R, C, W, T>() {
      return (*const_cast<const StaticArray *>(
        parent_))[simd::index::Unroll<R>{i_}, simd::index::Unroll<C, W>{j_}];
    }
    constexpr auto operator+=(const auto &x) -> Ref & {
      return (*this) = simd::Unroll<R, C, W, T>(*this) + x;
    }
    constexpr auto operator-=(const auto &x) -> Ref & {
      return (*this) = simd::Unroll<R, C, W, T>(*this) - x;
    }
    constexpr auto operator*=(const auto &x) -> Ref & {
      return (*this) = simd::Unroll<R, C, W, T>(*this) * x;
    }
    constexpr auto operator/=(const auto &x) -> Ref & {
      return (*this) = simd::Unroll<R, C, W, T>(*this) / x;
    }
  };
  template <ptrdiff_t U, typename Mask>
  [[gnu::always_inline]] auto
  operator[](ptrdiff_t i, simd::index::Unroll<U, W, Mask> j) -> Ref<1, U> {
    return Ref<1, U>{this, i, j.index_};
  }
  template <ptrdiff_t R, ptrdiff_t C, typename Mask>
  [[gnu::always_inline]] auto
  operator[](simd::index::Unroll<R> i,
             simd::index::Unroll<C, W, Mask> j) -> Ref<R, C> {
    return Ref<R, C>{this, i.index_, j.index_};
  }
  [[gnu::always_inline]] constexpr auto
  operator[](auto i) noexcept -> decltype(auto)
  requires((N == 1) || (M == 1))
  {
    if constexpr (M == 1) return (*this)[0z, i];
    else return (*this)[i, 0z];
  }
  [[gnu::always_inline]] constexpr auto
  operator[](auto i) const noexcept -> decltype(auto)
  requires((N == 1) || (M == 1))
  {
    if constexpr (M == 1) return (*this)[0z, i];
    else return (*this)[i, 0z];
  }
  constexpr auto operator==(const StaticArray &other) const -> bool {
    // masks return `true` if `any` are on
    for (ptrdiff_t i = 0z; i < M * L; ++i)
      if (simd::cmp::ne<W, T>(memory_[i], other.memory_[i])) return false;
    return true;
  }
  template <size_t I> [[nodiscard]] constexpr auto get() const -> T {
    return memory_[I / W][I % W];
  }
  friend void PrintTo(const StaticArray &x, ::std::ostream *os) {
    *os << x.view();
  }

private:
  friend auto operator<<(std::ostream &os, const StaticArray &x)
    -> std::ostream &requires(utils::Printable<T>) {
    if constexpr (MatrixDimension<S>)
      return printMatrix(os, Array<T, StridedDims<>>{x});
    else return utils::printVector(os, x.begin(), x.end());
  }
};

template <simd::SIMDSupported T, ptrdiff_t N>
requires((N > 1) && ((N + simd::VecLen<N, T> - 1) / simd::VecLen<N, T>) == 1)
struct MATH_GSL_OWNER StaticArray<T, 1, N, false>
  : ArrayOps<T, StaticDims<T, 1, N, false>, StaticArray<T, 1, N, false>>,
    Expr<T, StaticArray<T, 1, N, false>> {
  // struct StaticArray<T, M, N, alignof(simd::Vec<simd::VecLen<N, T>, T>)>
  //   : ArrayOps<T, StaticDims<T, M, N, alignof(simd::Vec<simd::VecLen<N, T>,
  //   T>)>,
  //              StaticArray<T, M, N, alignof(simd::Vec<simd::VecLen<N, T>,
  //              T>)>> {

  using value_type = T;
  using reference = T &;
  using const_reference = const T &;
  using size_type = ptrdiff_t;
  using difference_type = ptrdiff_t;
  using iterator = T *;
  using const_iterator = const T *;
  using pointer = T *;
  using const_pointer = const T *;
  using concrete = std::true_type;

  using compressed_type = StaticArray<T, 1, N, true>;
  static constexpr ptrdiff_t L = 1;

  constexpr void compress(compressed_type *p) const { *p << *this; }
  static constexpr auto decompress(const compressed_type *p) -> StaticArray {
    return StaticArray{*p};
  }
  static constexpr ptrdiff_t W = simd::VecLen<N, T>;
  static constexpr ptrdiff_t Align = alignof(simd::Vec<W, T>);
  using S = Length<N>;
  [[nodiscard]] static constexpr auto dim() noexcept -> S { return S{}; }

  [[nodiscard, gnu::always_inline]] constexpr auto view() const -> StaticArray {
    return *this;
  }
  // Maybe this should return `StaticArray&`?
  [[nodiscard, gnu::always_inline]] constexpr auto
  mview() noexcept -> MutArray<T, S, false> {
    return {data(), S{}};
  }

  using V = simd::Vec<W, T>;
  // simd::Vec<W, T> data[M][L];
  V data_{};
  // std::array<std::array<simd::Vec<W, T>, L>, M> data;
  // constexpr operator compressed_type() { return compressed_type{*this}; }
  constexpr StaticArray() = default;
  constexpr StaticArray(V v) : data_(v) {};
  constexpr StaticArray(StaticArray const &) = default;
  constexpr StaticArray(StaticArray &&) noexcept = default;
  constexpr explicit StaticArray(const std::initializer_list<T> &list) {
    if (list.size() == 1) {
      data_ += *list.begin();
      return;
    }
    invariant(list.size() <= W);
    for (size_t i = 0; i < list.size(); ++i) data_[i] = *(list.begin() + i);
  }
  constexpr auto data() -> T * { return reinterpret_cast<T *>(&data_); }
  [[nodiscard]] constexpr auto data() const -> const T * {
    return reinterpret_cast<const T *>(&data_);
  }
  constexpr auto begin() -> T * { return data(); }
  constexpr auto end() -> T * { return data() + N; }
  [[nodiscard]] constexpr auto begin() const -> const T * { return data(); }
  [[nodiscard]] constexpr auto end() const -> const T * { return data() + N; }
  constexpr auto operator[](Range<ptrdiff_t, ptrdiff_t> r) -> MutPtrVector<T> {
    return {data() + r.b, length(r.size())};
  }
  constexpr auto
  operator[](Range<ptrdiff_t, ptrdiff_t> r) const -> PtrVector<T> {
    return {data() + r.b, length(r.size())};
  }
  template <AbstractSimilar<S> V> constexpr StaticArray(const V &b) noexcept {
    (*this) << b;
  }
  constexpr explicit StaticArray(T x) : data_{simd::vbroadcast<W, T>(x)} {}
  constexpr auto operator=(StaticArray const &) -> StaticArray & = default;
  constexpr auto operator=(StaticArray &&) noexcept -> StaticArray & = default;
  static constexpr auto numRow() -> Row<1> { return {}; }
  static constexpr auto numCol() -> Col<N> { return {}; }
  static constexpr auto safeRow() -> Row<1> { return {}; }
  static constexpr auto safeCol() -> Col<W> { return {}; }
  [[nodiscard]] constexpr auto t() const -> Transpose<T, Array<T, S, false>> {
    return {*this};
  }
  [[nodiscard]] static constexpr auto rowStride() noexcept -> RowStride<W> {
    return {};
  }
  [[nodiscard]] static constexpr auto
  size() noexcept -> std::integral_constant<ptrdiff_t, N> {
    return {};
  }
  auto operator[](ptrdiff_t, ptrdiff_t j) -> T & { return data()[j]; }
  auto operator[](ptrdiff_t, ptrdiff_t j) const -> T { return data_[j]; }
  auto operator[](ptrdiff_t j) -> T & { return data()[j]; }
  auto operator[](ptrdiff_t j) const -> T { return data_[j]; }
  template <typename Mask>
  [[gnu::always_inline]] auto operator[](ptrdiff_t,
                                         simd::index::Unroll<1, W, Mask>) const
    -> simd::Unroll<1, 1, W, T> {
    return {data_};
  }
  template <ptrdiff_t R = 1>
  [[gnu::always_inline]] static constexpr void checkinds(ptrdiff_t j) {
    invariant(j >= 0);
    invariant(j < N);
    invariant((j % W) == 0);
  }
  template <ptrdiff_t R, typename Mask>
  [[gnu::always_inline]] auto
  operator[](simd::index::Unroll<R>, simd::index::Unroll<1, W, Mask> j) const
    -> simd::Unroll<R, 1, W, T> {
    checkinds<R>(j.index_);
    simd::Unroll<R, 1, W, T> ret;
    POLYMATHFULLUNROLL
    for (ptrdiff_t r = 0; r < R; ++r) ret[r, 0] = data_;
    return ret;
  }
  struct Ref {
    StaticArray *parent_;
    constexpr auto operator=(simd::Unroll<1, 1, W, T> x) -> Ref & {
      parent_->data_ = x.vec_;
      return *this;
    }
    constexpr auto operator=(simd::Vec<W, T> x) -> Ref & {
      parent_->data_ = x;
      return *this;
    }
    constexpr auto operator=(std::convertible_to<T> auto x) -> Ref & {
      *this = simd::vbroadcast<W, T>(x);
      return *this;
    }
    constexpr operator simd::Unroll<1, 1, W, T>() { return {parent_->data_}; }
    constexpr auto operator+=(const auto &x) -> Ref & {
      return (*this) = simd::Unroll<1, 1, W, T>(*this) + x;
    }
    constexpr auto operator-=(const auto &x) -> Ref & {
      return (*this) = simd::Unroll<1, 1, W, T>(*this) - x;
    }
    constexpr auto operator*=(const auto &x) -> Ref & {
      return (*this) = simd::Unroll<1, 1, W, T>(*this) * x;
    }
    constexpr auto operator/=(const auto &x) -> Ref & {
      return (*this) = simd::Unroll<1, 1, W, T>(*this) / x;
    }
  };
  template <ptrdiff_t U, typename Mask>
  [[gnu::always_inline]] auto
  operator[](ptrdiff_t, simd::index::Unroll<1, W, Mask>) -> Ref {
    return Ref{this};
  }
  template <ptrdiff_t R, typename Mask>
  [[gnu::always_inline]] auto
  operator[](simd::index::Unroll<R>, simd::index::Unroll<1, W, Mask>) -> Ref {
    return Ref{this};
  }
  template <typename Mask>
  [[gnu::always_inline]] constexpr auto
  operator[](simd::index::Unroll<1, W, Mask>) -> decltype(auto) {
    return Ref{this};
  }
  template <typename Mask>
  [[gnu::always_inline]] constexpr auto operator[](
    simd::index::Unroll<1, W, Mask>) const -> simd::Unroll<1, 1, W, T> {
    return {data_};
  }
  constexpr auto operator==(const StaticArray &other) const -> bool {
    return bool(simd::cmp::eq<W, T>(data_, other.data_));
  }
  template <size_t I> [[nodiscard]] constexpr auto get() const -> T {
    return data_[I];
  }
  constexpr void set(T x, ptrdiff_t r, ptrdiff_t c) {
    invariant(r == 0);
    invariant(c < N);
    if constexpr (W == 1) {
      data_ = x;
    } else {
      using IT = std::conditional_t<sizeof(T) == 8, int64_t, int32_t>;
      data_ = simd::range<W, IT>() == simd::vbroadcast<W, IT>(c % W)
                ? simd::vbroadcast<W, T>(x)
                : data_;
    }
  }
  template <typename SHAPE>
  constexpr operator Array<T, SHAPE, false>() const
  requires(std::convertible_to<S, SHAPE>)
  {
    return {const_cast<T *>(data()), SHAPE(dim())};
  }
  template <typename SHAPE>
  constexpr operator MutArray<T, SHAPE, false>()
  requires(std::convertible_to<S, SHAPE>)
  {
    return {data(), SHAPE(dim())};
  }

private:
  friend auto operator<<(std::ostream &os, const StaticArray &x)
    -> std::ostream &requires(utils::Printable<T>) {
    if constexpr (MatrixDimension<S>)
      return printMatrix(os, Array<T, StridedDims<>>{x});
    else return utils::printVector(os, x.begin(), x.end());
  }
};

template <class T, ptrdiff_t N, ptrdiff_t Compress = false>
using SVector = StaticArray<T, 1z, N, Compress>;
static_assert(
  std::same_as<Row<1>, decltype(SVector<int64_t, 3, true>::numRow())>);
static_assert(std::same_as<Row<1>, decltype(SVector<int64_t, 3>::numRow())>);
static_assert(
  std::same_as<Row<1>, decltype(numRows(std::declval<SVector<int64_t, 3>>()))>);

static_assert(RowVector<SVector<int64_t, 3>>);
static_assert(!ColVector<SVector<int64_t, 3>>);
static_assert(!RowVector<Transpose<int64_t, SVector<int64_t, 3>>>);
static_assert(ColVector<Transpose<int64_t, SVector<int64_t, 3>>>);
static_assert(RowVector<StaticArray<int64_t, 1, 4, false>>);
static_assert(RowVector<StaticArray<int64_t, 1, 4, true>>);
static_assert(RowVector<StaticArray<int64_t, 1, 3, false>>);
static_assert(RowVector<StaticArray<int64_t, 1, 3, true>>);

template <class T, ptrdiff_t M, ptrdiff_t N>
constexpr auto view(const StaticArray<T, M, N> &x) {
  return x.view();
}

template <class T, class... U>
StaticArray(T, U...) -> StaticArray<T, 1, 1 + sizeof...(U)>;

static_assert(utils::Compressible<SVector<int64_t, 3>>);
static_assert(utils::Compressible<SVector<int64_t, 7>>);

} // namespace math

template <class T, ptrdiff_t N> // NOLINTNEXTLINE(cert-dcl58-cpp)
struct std::tuple_size<::math::SVector<T, N>>
  : std::integral_constant<ptrdiff_t, N> {};

template <size_t I, class T, ptrdiff_t N> // NOLINTNEXTLINE(cert-dcl58-cpp)
struct std::tuple_element<I, math::SVector<T, N>> {
  using type = T;
};
