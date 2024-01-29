#pragma once

#include "Math/Array.hpp"
#include "Math/Indexing.hpp"
#include "Utilities/Reference.hpp"
#include <type_traits>

namespace poly::math {

static_assert(
  AbstractSimilar<PtrVector<int64_t>, std::integral_constant<unsigned int, 4>>);

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

template <class T, ptrdiff_t M, ptrdiff_t N, bool Compress>
using StaticDims = std::conditional_t<
  M == 1, std::integral_constant<ptrdiff_t, N>,
  std::conditional_t<
    Compress || (((N * sizeof(T)) % simd::VecLen<N, T>) == 0), DenseDims<M, N>,
    StridedDims<M, N, calcPaddedCols<T, N, alignSIMD<T, N>()>()>>>;

template <class T, ptrdiff_t M, ptrdiff_t N, bool Compress = false>
struct StaticArray : public ArrayOps<T, StaticDims<T, M, N, Compress>,
                                     StaticArray<T, M, N, Compress>> {
  using storage_type = std::conditional_t<Compress, utils::compressed_t<T>, T>;
  static constexpr ptrdiff_t Align =
    Compress ? alignof(storage_type) : alignSIMD<T, N>();
  static constexpr ptrdiff_t PaddedCols = calcPaddedCols<T, N, Align>();
  static constexpr ptrdiff_t capacity = M * PaddedCols;

  static_assert(alignof(storage_type) <= Align);

  alignas(Align) storage_type memory_[capacity];

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
  using compressed_type = StaticArray<utils::compressed_t<T>, M, N, true>;
  using decompressed_type = StaticArray<value_type, M, N, false>;
  using S = StaticDims<T, M, N, Compress>;
  constexpr StaticArray(){}; // NOLINT(modernize-use-equals-default)
  constexpr explicit StaticArray(const T &x) noexcept { (*this) << x; }
  constexpr explicit StaticArray(
    const std::convertible_to<T> auto &x) noexcept {
    (*this) << x;
  }
  constexpr explicit StaticArray(StaticArray const &) = default;
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
    this->vcopyTo(b, utils::CopyAssign{});
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
  [[nodiscard]] constexpr auto data() const noexcept -> const T * {
    return memory_;
  }
  constexpr auto data() noexcept -> T * { return memory_; }

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
    StridedRange r{minRowCol(), unsigned(RowStride(S{})) + 1};
    auto ptr = data();
    invariant(ptr != nullptr);
    return Array<T, StridedRange>{ptr, r};
  }
  [[nodiscard]] constexpr auto antiDiag() const noexcept {
    StridedRange r{minRowCol(), unsigned(RowStride(S{})) - 1};
    auto ptr = data();
    invariant(ptr != nullptr);
    return Array<T, StridedRange>{ptr + ptrdiff_t(Col(S{})) - 1, r};
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
  [[nodiscard]] static constexpr auto rowStride() noexcept
    -> RowStride<PaddedCols> {
    return {};
  }
  [[nodiscard]] static constexpr auto empty() -> bool { return capacity == 0; }
  [[nodiscard]] static constexpr auto size() noexcept
    -> std::integral_constant<ptrdiff_t, M * N> {
    return {};
  }
  [[nodiscard]] static constexpr auto dim() noexcept -> S { return S{}; }
  [[nodiscard]] constexpr auto t() const { return Transpose{*this}; }
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
  [[nodiscard]] constexpr auto view() const noexcept -> Array<T, S, Compress> {
    const storage_type *ptr = data();
    invariant(ptr != nullptr);
    return {ptr, S{}};
  }

  [[nodiscard]] constexpr auto begin() noexcept {
    if constexpr (std::is_same_v<S, StridedRange>)
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
  [[gnu::flatten, gnu::always_inline]] constexpr auto operator[](R r,
                                                                 C c) noexcept
    -> decltype(auto) {
    return index<T>(data(), S{}, r, c);
  }
  constexpr void fill(T value) {
    std::fill_n(data(), ptrdiff_t(this->dim()), value);
  }
  [[nodiscard]] constexpr auto diag() noexcept {
    StridedRange r{unsigned(min(Row(S{}), Col(S{}))),
                   unsigned(RowStride(S{})) + 1};
    return MutArray<T, StridedRange>{data(), r};
  }
  [[nodiscard]] constexpr auto antiDiag() noexcept {
    Col c = Col(S{});
    StridedRange r{unsigned(min(Row(S{}), c)), unsigned(RowStride(S{})) - 1};
    return MutArray<T, StridedRange>{data() + ptrdiff_t(c) - 1, r};
  }
  constexpr auto operator==(const StaticArray &rhs) const noexcept -> bool {
    return std::equal(begin(), end(), rhs.begin());
  }
  template <std::size_t I> constexpr auto get() -> T & { return memory_[I]; }
  template <std::size_t I>
  [[nodiscard]] constexpr auto get() const -> const T & {
    return memory_[I];
  }
};

template <simd::SIMDSupported T, ptrdiff_t M, ptrdiff_t N>
requires((M * (N + simd::VecLen<N, T> - 1) / simd::VecLen<N, T>) > 1)
struct StaticArray<T, M, N, false>
  : ArrayOps<T, StaticDims<T, M, N, false>, StaticArray<T, M, N, false>> {
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

  [[nodiscard]] constexpr auto view() const -> StaticArray { return *this; }
  [[nodiscard]] static constexpr auto dim() noexcept -> S { return S{}; }

  static constexpr ptrdiff_t L = (N + W - 1) / W;
  static_assert(L * W == calcPaddedCols<T, N, Align>());
  using V = simd::Vec<W, T>;
  // simd::Vec<W, T> data[M][L];
  V memory_[M * L];
  // std::array<std::array<simd::Vec<W, T>, L>, M> data;
  // constexpr operator compressed_type() { return compressed_type{*this}; }
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
  [[nodiscard]] constexpr auto t() const { return Transpose{*this}; }
  template <AbstractSimilar<S> V> constexpr StaticArray(const V &b) noexcept {
    this->vcopyTo(b, utils::CopyAssign{});
  }
  constexpr explicit StaticArray(T x) {
    simd::Vec<W, T> v = simd::vbroadcast<W, T>(x);
    for (ptrdiff_t i = 0; i < M * L; ++i) memory_[i] = v;
  }
  constexpr auto operator=(StaticArray const &) -> StaticArray & = default;
  constexpr auto operator=(StaticArray &&) noexcept -> StaticArray & = default;
  constexpr StaticArray(){}; // NOLINT(modernize-use-equals-default)
  static constexpr auto numRow() -> Row<M> { return {}; }
  static constexpr auto numCol() -> Col<N> { return {}; }
  static constexpr auto safeRow() -> Row<M> { return {}; }
  static constexpr auto safeCol() -> Col<L * W> { return {}; }
  [[nodiscard]] static constexpr auto rowStride() noexcept -> RowStride<L * W> {
    return {};
  }
  [[nodiscard]] static constexpr auto size() noexcept
    -> std::integral_constant<ptrdiff_t, M * N> {
    return {};
  }
  template <ptrdiff_t U, typename Mask>
  [[gnu::always_inline]] inline auto
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
  [[gnu::flatten, gnu::always_inline]] inline auto
  operator[](simd::index::Unroll<R> i, simd::index::Unroll<C, W, Mask> j) const
    -> simd::Unroll<R, C, W, T> {
    checkinds<R>(i.index, j.index);
    simd::Unroll<R, C, W, T> ret;
    ptrdiff_t k = j.index / W;
    POLYMATHFULLUNROLL
    for (ptrdiff_t r = 0; r < R; ++r) {
      POLYMATHFULLUNROLL
      for (ptrdiff_t u = 0; u < C; ++u)
        ret[r, u] = memory_[(i.index + r) * L + k + u];
    }
    return ret;
  }
  template <class R>
  [[gnu::flatten, gnu::always_inline]] constexpr auto
  operator[](R r, ptrdiff_t c) noexcept -> decltype(auto) {
    if constexpr (std::integral<R>)
      return reinterpret_cast<T *>(memory_ + ptrdiff_t(r) * L)[c];
    else return index<T>(data(), S{}, r, c);
  }
  template <class R>
  [[gnu::flatten, gnu::always_inline]] constexpr auto
  operator[](R r, ptrdiff_t c) const noexcept -> decltype(auto) {
    if constexpr (std::integral<R>)
      return reinterpret_cast<const T *>(memory_ + ptrdiff_t(r) * L)[c];
    else return index<T>(data(), S{}, r, c);
  }
  template <ptrdiff_t R, ptrdiff_t C> struct Ref {
    StaticArray *parent;
    ptrdiff_t i, j;
    constexpr auto operator=(simd::Unroll<R, C, W, T> x) -> Ref & {
      checkinds<R>(i, j);
      ptrdiff_t k = j / W;
      POLYMATHFULLUNROLL
      for (ptrdiff_t r = 0; r < R; ++r) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t u = 0; u < C; ++u)
          parent->memory_[(i + r) * L + k + u] = x[r, u];
      }
      return *this;
    }
    constexpr auto operator=(simd::Vec<W, T> x) -> Ref & {
      checkinds<R>(i, j);
      ptrdiff_t k = j / W;
      POLYMATHFULLUNROLL
      for (ptrdiff_t r = 0; r < R; ++r) {
        POLYMATHFULLUNROLL
        for (ptrdiff_t u = 0; u < C; ++u)
          parent->memory_[(i + r) * L + k + u] = x;
      }
      return *this;
    }
    constexpr auto operator=(std::convertible_to<T> auto x) -> Ref & {
      *this = simd::Vec<W, T>{} + T(x);
      return *this;
    }
    constexpr operator simd::Unroll<R, C, W, T>() {
      return (*const_cast<const StaticArray *>(
        parent))[simd::index::Unroll<R>{i}, simd::index::Unroll<C, W>{j}];
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
  [[gnu::always_inline]] inline auto
  operator[](ptrdiff_t i, simd::index::Unroll<U, W, Mask> j) -> Ref<1, U> {
    return Ref<1, U>{this, i, j.index};
  }
  template <ptrdiff_t R, ptrdiff_t C, typename Mask>
  [[gnu::always_inline]] inline auto
  operator[](simd::index::Unroll<R> i, simd::index::Unroll<C, W, Mask> j)
    -> Ref<R, C> {
    return Ref<R, C>{this, i.index, j.index};
  }
  [[gnu::always_inline]] constexpr auto operator[](auto i) noexcept
    -> decltype(auto)
  requires((N == 1) || (M == 1))
  {
    if constexpr (M == 1) return (*this)[0, i];
    else return (*this)[i, 0];
  }
  [[gnu::always_inline]] constexpr auto operator[](auto i) const noexcept
    -> decltype(auto)
  requires((N == 1) || (M == 1))
  {
    if constexpr (M == 1) return (*this)[0, i];
    else return (*this)[i, 0];
  }
  constexpr auto operator==(const StaticArray &other) const -> bool {
    // masks return `true` if `any` are on
    for (ptrdiff_t i = 0; i < M * L; ++i)
      if (simd::cmp::ne<W, T>(memory_[i], other.memory_[i])) return false;
    return true;
  }
  template <std::size_t I> [[nodiscard]] constexpr auto get() const -> T {
    return memory_[I / W][I % W];
  }
};

template <simd::SIMDSupported T, ptrdiff_t N>
requires((N > 1) && ((N + simd::VecLen<N, T> - 1) / simd::VecLen<N, T>) == 1)
struct StaticArray<T, 1, N, false>
  : ArrayOps<T, StaticDims<T, 1, N, false>, StaticArray<T, 1, N, false>> {
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
  using S = std::integral_constant<ptrdiff_t, N>;
  [[nodiscard]] static constexpr auto dim() noexcept -> S { return S{}; }

  [[nodiscard]] constexpr auto view() const -> StaticArray { return *this; }

  using V = simd::Vec<W, T>;
  // simd::Vec<W, T> data[M][L];
  V data_;
  // std::array<std::array<simd::Vec<W, T>, L>, M> data;
  // constexpr operator compressed_type() { return compressed_type{*this}; }
  constexpr StaticArray(V v) : data_(v){};
  constexpr StaticArray(StaticArray const &) = default;
  constexpr StaticArray(StaticArray &&) noexcept = default;
  constexpr explicit StaticArray(const std::initializer_list<T> &list)
    : data_{} {
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
  template <AbstractSimilar<S> V> constexpr StaticArray(const V &b) noexcept {
    (*this) << b;
  }
  constexpr explicit StaticArray(T x) : data_{simd::vbroadcast<W, T>(x)} {}
  constexpr auto operator=(StaticArray const &) -> StaticArray & = default;
  constexpr auto operator=(StaticArray &&) noexcept -> StaticArray & = default;
  constexpr StaticArray(){}; // NOLINT(modernize-use-equals-default)
  static constexpr auto numRow() -> Row<1> { return {}; }
  static constexpr auto numCol() -> Col<N> { return {}; }
  static constexpr auto safeRow() -> Row<1> { return {}; }
  static constexpr auto safeCol() -> Col<W> { return {}; }
  [[nodiscard]] constexpr auto t() const { return Transpose{*this}; }
  [[nodiscard]] static constexpr auto rowStride() noexcept -> RowStride<W> {
    return {};
  }
  [[nodiscard]] static constexpr auto size() noexcept
    -> std::integral_constant<ptrdiff_t, N> {
    return {};
  }
  inline auto operator[](ptrdiff_t, ptrdiff_t j) -> T & { return data()[j]; }
  inline auto operator[](ptrdiff_t, ptrdiff_t j) const -> T { return data_[j]; }
  inline auto operator[](ptrdiff_t j) -> T & { return data()[j]; }
  inline auto operator[](ptrdiff_t j) const -> T { return data_[j]; }
  template <typename Mask>
  [[gnu::always_inline]] inline auto
  operator[](ptrdiff_t, simd::index::Unroll<1, W, Mask>) const
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
  [[gnu::always_inline]] inline auto
  operator[](simd::index::Unroll<R>, simd::index::Unroll<1, W, Mask> j) const
    -> simd::Unroll<R, 1, W, T> {
    checkinds<R>(j.index);
    simd::Unroll<R, 1, W, T> ret;
    POLYMATHFULLUNROLL
    for (ptrdiff_t r = 0; r < R; ++r) ret[r, 0] = data_;
    return ret;
  }
  struct Ref {
    StaticArray *parent;
    constexpr auto operator=(simd::Unroll<1, 1, W, T> x) -> Ref & {
      parent->data_ = x.vec;
      return *this;
    }
    constexpr auto operator=(simd::Vec<W, T> x) -> Ref & {
      parent->data_ = x;
      return *this;
    }
    constexpr auto operator=(std::convertible_to<T> auto x) -> Ref & {
      *this = simd::vbroadcast<W, T>(x);
      return *this;
    }
    constexpr operator simd::Unroll<1, 1, W, T>() { return {parent->data_}; }
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
  [[gnu::always_inline]] inline auto operator[](ptrdiff_t,
                                                simd::index::Unroll<1, W, Mask>)
    -> Ref {
    return Ref{this};
  }
  template <ptrdiff_t R, typename Mask>
  [[gnu::always_inline]] inline auto operator[](simd::index::Unroll<R>,
                                                simd::index::Unroll<1, W, Mask>)
    -> Ref {
    return Ref{this};
  }
  template <typename Mask>
  [[gnu::always_inline]] constexpr auto
  operator[](simd::index::Unroll<1, W, Mask>) -> decltype(auto) {
    return Ref{this};
  }
  template <typename Mask>
  [[gnu::always_inline]] constexpr auto
  operator[](simd::index::Unroll<1, W, Mask>) const
    -> simd::Unroll<1, 1, W, T> {
    simd::Unroll<1, 1, W, T> ret;
    ret.vec = data_;
    return ret;
  }
  constexpr auto operator==(const StaticArray &other) const -> bool {
    return bool(simd::cmp::eq<W, T>(data_, other.data_));
  }
  template <std::size_t I> [[nodiscard]] constexpr auto get() const -> T {
    return data_[I];
  }
};

template <class T, ptrdiff_t N, ptrdiff_t Compress = false>
using SVector = StaticArray<T, 1, N, Compress>;
static_assert(
  std::same_as<Row<1>, decltype(SVector<int64_t, 3, true>::numRow())>);
static_assert(std::same_as<Row<1>, decltype(SVector<int64_t, 3>::numRow())>);
static_assert(
  std::same_as<Row<1>, decltype(numRows(std::declval<SVector<int64_t, 3>>()))>);

static_assert(RowVector<SVector<int64_t, 3>>);
static_assert(!ColVector<SVector<int64_t, 3>>);
static_assert(!RowVector<Transpose<SVector<int64_t, 3>>>);
static_assert(ColVector<Transpose<SVector<int64_t, 3>>>);
static_assert(RowVector<StaticArray<int64_t, 1, 4, false>>);
static_assert(RowVector<StaticArray<int64_t, 1, 4, true>>);

template <class T, ptrdiff_t M, ptrdiff_t N>
inline constexpr auto view(const StaticArray<T, M, N> &x) {
  return x.view();
}

template <class T, class... U>
StaticArray(T, U...) -> StaticArray<T, 1, 1 + sizeof...(U)>;

static_assert(utils::Compressible<SVector<int64_t, 3>>);
static_assert(utils::Compressible<SVector<int64_t, 7>>);

} // namespace poly::math

template <class T, ptrdiff_t N> // NOLINTNEXTLINE(cert-dcl58-cpp)
struct std::tuple_size<::poly::math::SVector<T, N>>
  : std::integral_constant<ptrdiff_t, N> {};

template <size_t I, class T, ptrdiff_t N> // NOLINTNEXTLINE(cert-dcl58-cpp)
struct std::tuple_element<I, poly::math::SVector<T, N>> {
  using type = T;
};
