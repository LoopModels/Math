#pragma once
#include "Math/Array.hpp"
#include "Math/ArrayOps.hpp"
#include "Math/SIMD.hpp"
#include "Math/Vector.hpp"
#include <cstddef>
#include <utility>

namespace poly::math {

static_assert(
  AbstractSimilar<PtrVector<int64_t>, std::integral_constant<unsigned int, 4>>);

// TODO: add MatrixDimension support
template <typename T>
concept StaticSize = StaticInt<T>;

template <class T, StaticSize S>
class StaticArray : public ArrayOps<T, S, StaticArray<T, S>> {
  static constexpr ptrdiff_t capacity = ptrdiff_t{S{}};
  T memory[capacity]; // NOLINT(modernize-avoid-c-arrays)

public:
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
  constexpr StaticArray() = default;
  constexpr StaticArray(const T &x) noexcept {
    std::fill_n(data(), capacity, x);
  }
  constexpr StaticArray(StaticArray const &) = default;
  constexpr StaticArray(StaticArray &&) noexcept = default;
  constexpr StaticArray(const std::initializer_list<T> &list) {
    invariant(list.size(), size_t(capacity));
    std::copy(list.begin(), list.end(), data());
  }
  template <AbstractSimilar<S> V> constexpr StaticArray(const V &b) noexcept {
    (*this) << b;
  }
  [[nodiscard]] constexpr auto data() const noexcept -> const T * {
    return memory;
  }
  constexpr auto data() noexcept -> T * { return memory; }

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
  // static constexpr auto slice(NotNull<T>, Index<S> auto i){
  //   auto
  // }
  constexpr auto operator[](Index<S> auto i) const noexcept -> decltype(auto) {
    auto offset = calcOffset(S{}, i);
    auto newDim = calcNewDim(S{}, i);
    auto ptr = data();
    invariant(ptr != nullptr);
    if constexpr (std::is_same_v<decltype(newDim), Empty>)
      return ref(static_cast<const T *>(ptr), offset);
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
    StridedRange r{minRowCol(), unsigned(RowStride{S{}}) + 1};
    auto ptr = data();
    invariant(ptr != nullptr);
    return Array<T, StridedRange>{ptr, r};
  }
  [[nodiscard]] constexpr auto antiDiag() const noexcept {
    StridedRange r{minRowCol(), unsigned(RowStride{S{}}) - 1};
    auto ptr = data();
    invariant(ptr != nullptr);
    return Array<T, StridedRange>{ptr + ptrdiff_t(Col{S{}}) - 1, r};
  }
  [[nodiscard]] static constexpr auto isSquare() noexcept -> bool {
    return Row{S{}} == Col{S{}};
  }
  [[nodiscard]] constexpr auto checkSquare() const -> Optional<ptrdiff_t> {
    ptrdiff_t N = ptrdiff_t(numRow());
    if (N != ptrdiff_t(numCol())) return {};
    return N;
  }

  [[nodiscard]] constexpr auto numRow() const noexcept -> Row {
    return Row{S{}};
  }
  [[nodiscard]] constexpr auto numCol() const noexcept -> Col {
    return Col{S{}};
  }
  [[nodiscard]] constexpr auto rowStride() const noexcept -> RowStride {
    return RowStride{S{}};
  }
  [[nodiscard]] static constexpr auto empty() -> bool { return capacity == 0; }
  [[nodiscard]] static constexpr auto size() noexcept {
    if constexpr (StaticInt<S>) return S{};
    else return CartesianIndex{Row{S{}}, Col{S{}}};
  }
  [[nodiscard]] static constexpr auto dim() noexcept -> S { return S{}; }
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
  constexpr auto operator[](Index<S> auto i) noexcept -> decltype(auto) {
    auto offset = calcOffset(S{}, i);
    auto newDim = calcNewDim(S{}, i);
    if constexpr (std::is_same_v<decltype(newDim), Empty>)
      return ref(data(), offset);
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
    std::fill_n((T *)(data()), ptrdiff_t(this->dim()), value);
  }
  [[nodiscard]] constexpr auto diag() noexcept {
    StridedRange r{unsigned(min(Row{S{}}, Col{S{}})),
                   unsigned(RowStride{S{}}) + 1};
    return MutArray<T, StridedRange>{data(), r};
  }
  [[nodiscard]] constexpr auto antiDiag() noexcept {
    Col c = Col{S{}};
    StridedRange r{unsigned(min(Row{S{}}, c)), unsigned(RowStride{S{}}) - 1};
    return MutArray<T, StridedRange>{data() + ptrdiff_t(c) - 1, r};
  }
  constexpr auto operator==(const StaticArray &rhs) const noexcept -> bool {
    return std::equal(begin(), end(), rhs.begin());
  }
};

constexpr auto cld(ptrdiff_t x, ptrdiff_t y) -> ptrdiff_t {
  return (x + y - 1) / y;
}

template <PrimitiveScalar T, ptrdiff_t N>
class StaticArray<T, std::integral_constant<ptrdiff_t, N>>
  : public Unrolled<
      T, simd::vecWidth<T, std::integral_constant<ptrdiff_t, N>>(),
      cld(N, simd::vecWidth<T, std::integral_constant<ptrdiff_t, N>>())> {
  static constexpr ptrdiff_t W =
    simd::vecWidth<T, std::integral_constant<ptrdiff_t, N>>();
  static constexpr ptrdiff_t NV = cld(N, W);
  using Base = Unrolled<T, W, NV>;

public:
  constexpr StaticArray() = default;
  constexpr StaticArray(const StaticArray &) = default;
  constexpr StaticArray(StaticArray &&) noexcept = default;
  constexpr auto operator=(const StaticArray &) -> StaticArray & = default;
  constexpr auto operator=(StaticArray &&) noexcept -> StaticArray & = default;
  constexpr StaticArray(Unrolled<T, W, NV> &&rhs) noexcept : Base(rhs) {}

  struct Reference {
    StaticArray *d_;
    ptrdiff_t i_;
    constexpr operator T() const { return d_->get(i_); }
    constexpr auto operator=(T v) -> Reference & {
      d_->set(i_, v);
      return *this;
    }
  };
  static constexpr bool eager = true;
  using value_type = T;
  using reference = Reference;
  // using const_reference = const T &;
  // using size_type = unsigned;
  // using difference_type = int;
  // using iterator = T *;
  // using const_iterator = const T *;
  // using pointer = T *;
  // using const_pointer = const T *;
  //       using concrete = std::true_type;

  constexpr auto operator[](ptrdiff_t i) const -> T {
    return this->data[i / W].get(i % W);
  }
  constexpr auto operator[](ptrdiff_t i) -> Reference {
    return Reference{this, i};
  }
  constexpr auto begin() noexcept -> T * {
    void *p = this->data;
    return static_cast<T *>(p);
  }
  [[nodiscard]] constexpr auto begin() const noexcept -> const T * {
    const void *p = this->data;
    return static_cast<const T *>(p);
  }
  constexpr auto end() noexcept -> T * { return begin() + N; }
  [[nodiscard]] constexpr auto end() const noexcept -> const T * {
    return begin() + N;
  }
  [[nodiscard]] constexpr auto size() const noexcept -> ptrdiff_t { return N; }
  constexpr auto operator<<(const AbstractVector auto &rhs) -> StaticArray & {
    MutPtrVector<T> result{begin(), N};
    result << rhs;
    return *this;
  }
  constexpr auto operator<<(const Unrolled<T, W, NV> &rhs) -> StaticArray & {
    *static_cast<Unrolled<T, W, NV> *>(this) = rhs;
    return *this;
  }
};
template <class T, ptrdiff_t N>
using SVector = StaticArray<T, std::integral_constant<ptrdiff_t, N>>;

static_assert(SVector<int64_t, 3>::eager);
static_assert(EagerArray<SVector<int64_t, 3>>);
static_assert(AbstractVector<SVector<int64_t, 3>>);
static_assert(!FuseVector<SVector<int64_t, 3>>);

template <class T, ptrdiff_t N>
inline constexpr auto len(const SVector<T, N> &) noexcept {
  return std::integral_constant<ptrdiff_t, N>{};
}
static_assert(std::same_as<decltype(len(std::declval<SVector<int64_t, 3>>())),
                           std::integral_constant<ptrdiff_t, 3>>);
static_assert(StaticallySized<SVector<int64_t, 3>>);

template <typename T>
concept StaticallySizedInlineData =
  StaticallySized<T> && std::is_trivially_destructible_v<T> &&
  std::is_trivially_copyable_v<T> && requires(T t) {
    { sizeof(T) >= len(std::declval<T>()) * sizeof(utils::eltype_t<T>) };
    { t.data() } -> std::same_as<const utils::eltype_t<T> *>;
    { t.size() } -> std::same_as<ptrdiff_t>;
  };

static_assert(!DoCopy<SVector<int64_t, 4>>);

}; // namespace poly::math
