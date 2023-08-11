#pragma once
#include "Math/Array.hpp"
#include "Math/SIMDWidth.hpp"
#include "Utilities/TypeCompression.hpp"
#include <cstddef>
#include <type_traits>
#include <utility>

namespace poly::math {

static_assert(
  AbstractSimilar<PtrVector<int64_t>, std::integral_constant<unsigned int, 4>>);

// TODO: add MatrixDimension support
template <typename T>
concept StaticSize = StaticInt<T>;

template <typename S> consteval auto calcCapacity() -> ptrdiff_t {
  return ptrdiff_t{S{}};
}

template <class T, StaticSize S, ptrdiff_t Capacity = calcCapacity<S>()>
class StaticArray : public ArrayOps<T, S, StaticArray<T, S>> {
  // Capacity may be larger than `len` to allow for padding.
  static constexpr ptrdiff_t len = ptrdiff_t{S{}};
  static_assert(Capacity >= len);
  T memory[Capacity]; // NOLINT(modernize-avoid-c-arrays)

public:
  using value_type = utils::uncompressed_t<T>;
  using reference = decltype(ref((T *)nullptr, 0));
  using const_reference = decltype(ref((const T *)nullptr, 0));
  using size_type = unsigned;
  using difference_type = ptrdiff_t;
  using iterator = T *;
  using const_iterator = const T *;
  using pointer = T *;
  using const_pointer = const T *;
  using concrete = std::true_type;
  constexpr explicit StaticArray(){}; // NOLINT(modernize-use-equals-default)
  constexpr explicit StaticArray(const T &x) noexcept {
    (*this) << x;
    // std::fill_n(data(), capacity, x);
  }
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
    invariant(list.size(), size_t(len));
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
  [[nodiscard]] constexpr auto end() const noexcept { return begin() + len; }
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
  [[nodiscard]] static constexpr auto empty() -> bool { return len == 0; }
  [[nodiscard]] static constexpr auto size() noexcept {
    if constexpr (StaticInt<S>) return S{};
    else return CartesianIndex{Row{S{}}, Col{S{}}};
  }
  [[nodiscard]] static constexpr auto paddedlength() noexcept -> ptrdiff_t {
    return Capacity;
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
  [[nodiscard]] constexpr auto end() noexcept { return begin() + len; }
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
  template <std::size_t I> constexpr auto get() -> T & { return memory[I]; }
  template <std::size_t I>
  [[nodiscard]] constexpr auto get() const -> const T & {
    return memory[I];
  }
  static constexpr auto decompress(const StaticArray *p) -> decltype(auto) {
    constexpr ptrdiff_t W =
      math::simd::vecWidth<T, std::integral_constant<ptrdiff_t, len>>();
    constexpr ptrdiff_t P = ((len + W - 1) / W) * W;
    if constexpr (P == len) return *p;
    else {
      StaticArray<T, std::integral_constant<ptrdiff_t, P>> result;
      ptrdiff_t i = 0, n = W;
      const auto &A{*p};
      POLYMATHVECTORIZE
      for (; n <= len; i = n, n += W) {
        auto j = simd::unroll<W, 1>(i);
        result[j] = A[j];
      }
      result[simd::unroll<W, 1>(i)] = A[simd::unroll<W, 1>(i, len)];
      return result;
    }
  }
  template <StaticSize U, ptrdiff_t C>
  static constexpr void compress(StaticArray *p,
                                 const StaticArray<T, U, C> &rhs) {
    constexpr ptrdiff_t W =
      math::simd::vecWidth<T, std::integral_constant<ptrdiff_t, len>>();
    if constexpr (W == 1) {
      *p << rhs;
      return;
    }
    ptrdiff_t i = 0, n = W;
    auto &A{*p};
    POLYMATHVECTORIZE
    for (; n <= len; i = n, n += W) {
      auto j = simd::unroll<W, 1>(i);
      A[j] = rhs[j];
    }
    if (i < len) A[simd::unroll<W, 1>(i, len)] = rhs[simd::unroll<W, 1>(i)];
  }
};

template <class T, ptrdiff_t N,
          ptrdiff_t P = calcCapacity<std::integral_constant<ptrdiff_t, N>>()>
using SVector = StaticArray<T, std::integral_constant<ptrdiff_t, N>, P>;

static_assert(AbstractVector<SVector<int64_t, 3>>);

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
static_assert(DenseLayout<std::integral_constant<ptrdiff_t, 4>>);

}; // namespace poly::math

namespace poly::utils {

static_assert(std::same_as<uncompressed_t<math::SVector<int64_t, 15>>,
                           math::SVector<int64_t, 16>>);

}; // namespace poly::utils

template <class T, ptrdiff_t N> // NOLINTNEXTLINE(cert-dcl58-cpp)
struct std::tuple_size<::poly::math::SVector<T, N>>
  : std::integral_constant<ptrdiff_t, N> {};

template <size_t I, class T, ptrdiff_t N> // NOLINTNEXTLINE(cert-dcl58-cpp)
struct std::tuple_element<I, poly::math::SVector<T, N>> {
  using type = T;
};
