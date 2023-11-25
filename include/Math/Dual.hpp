#pragma once
#include "Math/Exp.hpp"
#include "Math/MatrixDimensions.hpp"
#include <Math/Array.hpp>
#include <Math/Constructors.hpp>
#include <Math/Math.hpp>
#include <Math/Matrix.hpp>
#include <Math/StaticArrays.hpp>
#include <Utilities/Invariant.hpp>
#include <cstddef>
#include <utility>

namespace poly::math {

template <class T, ptrdiff_t N, bool Compress = false> struct Dual {
  static_assert(Compress);
  T val{};
  SVector<T, N, true> partials{T{}};

  using decompressed_type = Dual<utils::decompressed_t<T>, N, false>;
  constexpr operator decompressed_type() const {
    return decompressed_type::decompress(this);
  }
  [[nodiscard]] constexpr auto value() -> T { return val; }
  [[nodiscard]] constexpr auto value() const -> const T & { return val; }
  [[nodiscard]] constexpr auto gradient() -> SVector<T, N, true> & {
    return partials;
  }
  [[nodiscard]] constexpr auto gradient() const -> const SVector<T, N, true> & {
    return partials;
  }

  [[gnu::always_inline]] constexpr auto operator-() const & -> Dual {
    return {-val, -partials};
  }
  [[gnu::always_inline]] constexpr auto
  operator+(const Dual &other) const & -> Dual {
    return {val + other.val, partials + other.partials};
  }
  [[gnu::always_inline]] constexpr auto operator-(const Dual &other) const
    -> Dual {
    return {val - other.val, partials - other.partials};
  }
  [[gnu::always_inline]] constexpr auto operator+=(const Dual &other)
    -> Dual & {
    val += other.val;
    partials += other.partials;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator-=(const Dual &other)
    -> Dual & {
    val -= other.val;
    partials -= other.partials;
    return *this;
  }
};

template <class T, ptrdiff_t N>
requires(std::popcount(size_t(N)) > 1)
struct Dual<T, N, true> {
  SVector<T, N + 1, true> data{T{}};

  using decompressed_type = Dual<utils::decompressed_t<T>, N, false>;
  constexpr operator decompressed_type() const {
    return decompressed_type::decompress(this);
  }
  [[nodiscard]] constexpr auto value() -> T { return data[0]; }
  [[nodiscard]] constexpr auto value() const -> const T & { return data[0]; }
  [[nodiscard]] constexpr auto gradient()
    -> MutArray<T, std::integral_constant<ptrdiff_t, N>> {
    return {data.data() + 1, {}};
  }
  [[nodiscard]] constexpr auto gradient() const
    -> Array<T, std::integral_constant<ptrdiff_t, N>> {
    return {data.data() + 1, {}};
  }

  [[gnu::always_inline]] constexpr auto operator-() const & -> Dual {
    return {-data};
  }
  [[gnu::always_inline]] constexpr auto
  operator+(const Dual &other) const & -> Dual {
    return {data + other.data};
  }
  [[gnu::always_inline]] constexpr auto operator-(const Dual &other) const
    -> Dual {
    return {data - other.data};
  }
  [[gnu::always_inline]] constexpr auto operator+=(const Dual &other)
    -> Dual & {
    data += other.data;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator-=(const Dual &other)
    -> Dual & {
    data -= other.data;
    return *this;
  }
};

template <class T, ptrdiff_t N> struct Dual<T, N, false> {
  // default decompressed separates the value and partials
  T val{};
  SVector<T, N, false> partials{T{}};

  using compressed_type = Dual<utils::compressed_t<T>, N, true>;
  using decompressed_type = Dual<utils::decompressed_t<T>, N, false>;
  static_assert(std::same_as<Dual, decompressed_type>);

  constexpr Dual() = default;
  constexpr Dual(T v) : val(v) {}
  constexpr Dual(T v, ptrdiff_t n) : val(v) { partials[n] = T{1}; }
  constexpr Dual(T v, SVector<T, N> g) : val(v) { partials << g; }
  constexpr Dual(std::integral auto v) : val(v) {}
  constexpr Dual(std::floating_point auto v) : val(v) {}
  constexpr auto value() -> T & { return val; }
  constexpr auto gradient() -> SVector<T, N> & { return partials; }
  constexpr auto gradient(ptrdiff_t i) -> T & { return partials[i]; }
  [[nodiscard]] constexpr auto value() const -> const T & { return val; }
  [[nodiscard]] constexpr auto gradient() const -> const SVector<T, N> & {
    return partials;
  }
  [[nodiscard]] constexpr auto gradient(ptrdiff_t i) const -> const T & {
    return partials[i];
  }
  [[gnu::always_inline]] constexpr auto operator-() const & -> Dual {
    return {-val, -partials};
  }
  [[gnu::always_inline]] constexpr auto
  operator+(const Dual &other) const & -> Dual {
    return {val + other.val, partials + other.partials};
  }
  [[gnu::always_inline]] constexpr auto operator-(const Dual &other) const
    -> Dual {
    return {val - other.val, partials - other.partials};
  }
  [[gnu::always_inline]] constexpr auto operator*(const Dual &other) const
    -> Dual {
    // Dual res;
    // res.val = val * other.val;
    // res.partials << val * other.partials + other.val * partials;
    // return res;
    return {val * other.val, val * other.partials + other.val * partials};
  }
  [[gnu::always_inline]] constexpr auto operator/(const Dual &other) const
    -> Dual {
    return {val / other.val, (other.val * partials - val * other.partials) /
                               (other.val * other.val)};
  }
  [[gnu::always_inline]] constexpr auto
  operator+(const T &other) const & -> Dual
  requires(!std::same_as<T, double>)
  {
    return {val + other, partials};
  }
  [[gnu::always_inline]] constexpr auto operator-(const T &other) const -> Dual
  requires(!std::same_as<T, double>)
  {
    return {val - other.val, partials - other.partials};
  }
  [[gnu::always_inline]] constexpr auto operator*(const T &other) const -> Dual
  requires(!std::same_as<T, double>)
  {
    // Dual res;
    // res.val = val * other.val;
    // res.partials << val * other.partials + other.val * partials;
    // return res;
    return {val * other, other * partials};
  }
  [[gnu::always_inline]] constexpr auto operator/(const T &other) const -> Dual
  requires(!std::same_as<T, double>)
  {
    return {val / other, partials / other};
  }
  [[gnu::always_inline]] constexpr auto operator+=(const Dual &other)
    -> Dual & {
    val += other.val;
    partials += other.partials;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator-=(const Dual &other)
    -> Dual & {
    val -= other.val;
    partials -= other.partials;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator*=(const Dual &other)
    -> Dual & {
    val *= other.val;
    partials << val * other.partials + other.val * partials;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator/=(const Dual &other)
    -> Dual & {
    val /= other.val;
    partials << (other.val * partials - val * other.partials) /
                  (other.val * other.val);
    return *this;
  }
  [[gnu::always_inline]] constexpr auto
  operator+(double other) const & -> Dual {
    return {val + other, partials};
  }
  [[gnu::always_inline]] constexpr auto operator-(double other) const -> Dual {
    return {val - other, partials};
  }
  [[gnu::always_inline]] constexpr auto operator*(double other) const -> Dual {
    return {val * other, other * partials};
  }
  [[gnu::always_inline]] constexpr auto operator/(double other) const -> Dual {
    return {val / other, partials / other};
  }
  [[gnu::always_inline]] constexpr auto operator+=(double other) -> Dual & {
    val += other;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator-=(double other) -> Dual & {
    val -= other;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator*=(double other) -> Dual & {
    val *= other;
    partials *= other;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator/=(double other) -> Dual & {
    val /= other;
    partials /= other;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator==(const Dual &other) const
    -> bool {
    return val == other.val; // && grad == other.grad;
  }
  [[gnu::always_inline]] constexpr auto operator!=(const Dual &other) const
    -> bool {
    return val != other.val; // || grad != other.grad;
  }
  constexpr auto operator==(double other) const -> bool { return val == other; }
  constexpr auto operator!=(double other) const -> bool { return val != other; }
  constexpr auto operator<(double other) const -> bool { return val < other; }
  constexpr auto operator>(double other) const -> bool { return val > other; }
  constexpr auto operator<=(double other) const -> bool { return val <= other; }
  constexpr auto operator>=(double other) const -> bool { return val >= other; }
  constexpr auto operator<(const Dual &other) const -> bool {
    return val < other.val;
  }
  constexpr auto operator>(const Dual &other) const -> bool {
    return val > other.val;
  }
  constexpr auto operator<=(const Dual &other) const -> bool {
    return val <= other.val;
  }
  constexpr auto operator>=(const Dual &other) const -> bool {
    return val >= other.val;
  }
  constexpr void compress(compressed_type *p) const {
    utils::compress(val, &(p->val));
    partials.compress(&(p->partials));
  }
  static constexpr auto decompress(const compressed_type *p) -> Dual {
    return {utils::decompress<T>(&(p->val)),
            SVector<T, N>::decompress(&(p->partials))};
  }
  // constexpr void compress(compressed_type *p) const {
  //   utils::compress(val, &(p->data[0]));
  //   p->gradient() << partials;
  // }
  // static constexpr auto decompress(const compressed_type *p) -> Dual {
  //   return {utils::decompress<T>(&(p->data[0])), SVector<T,
  //   N>(p->gradient())};
  // }
  constexpr operator compressed_type() const {
    compressed_type ret;
    compress(&ret);
    return ret;
  }
};

template <simd::SIMDSupported T, ptrdiff_t N>
requires(std::popcount(size_t(N)) > 1)
struct Dual<T, N, false> {
  static constexpr ptrdiff_t value_idx = 0; // N;
  static constexpr ptrdiff_t partial_offset = value_idx != N;
  using data_type = SVector<T, N + 1, false>;
  data_type data{T{}};
  using compressed_type = Dual<T, N, true>;
  using decompressed_type = Dual<T, N, false>;

  using V = typename data_type::V;
  static constexpr ptrdiff_t W = data_type::W;
  // constexpr Dual() = default;
  constexpr Dual() = default;
  constexpr Dual(T v) { data[value_idx] = v; }
  constexpr Dual(T v, ptrdiff_t n) {
    data[value_idx] = v;
    data[partial_offset + n] = T{1};
  }
  // constexpr Dual(T v, ptrdiff_t n, T p) {
  //   data[value_idx] = v;
  //   data[partial_offset + n] = p;
  // }
  constexpr Dual(T v, AbstractVector auto g) {
    value() = v;
    gradient() << g;
  }
  constexpr Dual(SVector<T, N + 1> d) : data{d} {}
  constexpr Dual(const AbstractVector auto &d)
  requires(std::convertible_to<utils::eltype_t<decltype(d)>, T>)
    : data{d} {}
  constexpr Dual(std::integral auto v) { value() = v; }
  constexpr Dual(std::floating_point auto v) { value() = v; }
  constexpr auto value() -> T & { return data[value_idx]; }
  constexpr auto gradient()
    -> MutArray<T, std::integral_constant<ptrdiff_t, N>> {
    return {data.data() + partial_offset, {}};
  }
  [[nodiscard]] constexpr auto value() const -> T { return data[value_idx]; }
  // zeros out partial part of the vector
  [[nodiscard]] constexpr auto vvalue() const -> V {
    // return data[value_idx];
    return (simd::range<W, int64_t>() == simd::Vec<W, int64_t>{})
             ? data.memory_[0]
             : V{};
  }
  // broadcasts value across register
  [[nodiscard]] constexpr auto vbvalue() const -> V {
    return simd::vbroadcast<W, T>(data.memory_[0]);
  }
  [[nodiscard]] constexpr auto gradient() const
    -> Array<T, std::integral_constant<ptrdiff_t, N>> {
    return {data.data() + partial_offset, {}};
  }

  constexpr auto operator-() const & -> Dual { return {-data}; }
  constexpr auto operator+(const Dual &other) const & -> Dual {
    return {data + other.data};
  }
  constexpr auto operator-(const Dual &other) const -> Dual {
    return {data - other.data};
  }
  constexpr auto operator*(const Dual &other) const -> Dual {
    // TODO: either update remaining methods to match this style,
    // or figure out how to get `conditional`'s codegen quality to match
    Dual ret;
    V vt = vbvalue(), vo = other.vbvalue(), x = vt * other.data.memory_[0];
    x = simd::firstoff<W, int64_t>() ? x + vo * data.memory_[0] : x;
    ret.data.memory_[0] = x;
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 1; i < data_type::L; ++i)
      ret.data.memory_[i] = vt * other.data.memory_[i] + vo * data.memory_[i];
    return ret;
    // return {conditional(std::plus<>{},
    //                     elementwise_not_equal(_(0, N + 1), value_idx),
    //                     value() * other.data, data * other.value())};
  }
  constexpr auto operator/(const Dual &other) const -> Dual {
    Dual ret;
    V vt = vbvalue(), vo = other.vbvalue(), vo2 = vo * vo,
      x = vo * data.memory_[0];
    ret.data.memory_[0] =
      (simd::firstoff<W, int64_t>() ? x - vt * other.data.memory_[0] : x) / vo2;
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 1; i < data_type::L; ++i)
      ret.data.memory_[i] =
        (vo * data.memory_[i] - vt * other.data.memory_[i]) / vo2;
    return ret;
    // val = value() / other.value()
    // partials = (other.value() * gradient() - value() * other.gradient()) /
    // (other.value() * other.value())
    // partials = (gradient()) / (other.value())
    //  - (value() * other.gradient()) / (other.value() * other.value())
    // T v{other.value()};
    // return {conditional(std::minus<>{},
    //                     elementwise_not_equal(_(0, N + 1), value_idx), data /
    //                     v, value() * other.data / (v * v))};
  }
  constexpr auto operator+=(const Dual &other) -> Dual & {
    data += other.data;
    return *this;
  }
  constexpr auto operator-=(const Dual &other) -> Dual & {
    data -= other.data;
    return *this;
  }
  constexpr auto operator*=(const Dual &other) -> Dual & {
    V vt = vbvalue(), vo = other.vbvalue(), x = vt * other.data.memory_[0];
    data.memory_[0] =
      simd::firstoff<W, int64_t>() ? x + vo * data.memory_[0] : x;
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 1; i < data_type::L; ++i)
      data.memory_[i] = vt * other.data.memory_[i] + vo * data.memory_[i];
    // data << conditional(std::plus<>{},
    //                     elementwise_not_equal(_(0, N + 1), value_idx),
    //                     value() * other.data, data * other.value());
    return *this;
  }
  constexpr auto operator/=(const Dual &other) -> Dual & {
    V vt = vbvalue(), vo = other.vbvalue(), vo2 = vo * vo,
      x = vo * data.memory_[0];
    data.memory_[0] =
      (simd::firstoff<W, int64_t>() ? x - vt * other.data.memory_[0] : x) / vo2;
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 1; i < data_type::L; ++i)
      data.memory_[i] =
        (vo * data.memory_[i] - vt * other.data.memory_[i]) / vo2;
    // T v{other.value()};
    // data << conditional(std::minus<>{},
    //                     elementwise_not_equal(_(0, N + 1), value_idx), data /
    //                     v, value() * other.data / (v * v));
    return *this;
  }
  constexpr auto operator+(double other) const & -> Dual {
    Dual ret = *this;
    ret.data.memory_[0] += simd::Vec<SVector<T, N + 1>::W, T>{other};
    return ret;
  }
  constexpr auto operator-(double other) const -> Dual {
    Dual ret = *this;
    ret.data.memory_[0] -= simd::Vec<SVector<T, N + 1>::W, T>{other};
    return ret;
  }
  constexpr auto operator*(double other) const -> Dual {
    return {data * other};
  }
  constexpr auto operator/(double other) const -> Dual {
    return {data / other};
  }
  constexpr auto operator+=(double other) -> Dual & {
    data.memory_[0] += simd::Vec<SVector<T, N + 1>::W, T>{other};
    return *this;
  }
  constexpr auto operator-=(double other) -> Dual & {
    data.memory_[0] -= simd::Vec<SVector<T, N + 1>::W, T>{other};
    return *this;
  }
  constexpr auto operator*=(double other) -> Dual & {
    data *= other;
    return *this;
  }
  constexpr auto operator/=(double other) -> Dual & {
    data /= other;
    return *this;
  }
  constexpr auto operator==(const Dual &other) const -> bool {
    return value() == other.value(); // && grad == other.grad;
  }
  constexpr auto operator!=(const Dual &other) const -> bool {
    return value() != other.value(); // || grad != other.grad;
  }
  constexpr auto operator==(double other) const -> bool {
    return value() == other;
  }
  constexpr auto operator!=(double other) const -> bool {
    return value() != other;
  }
  constexpr auto operator<(double other) const -> bool {
    return value() < other;
  }
  constexpr auto operator>(double other) const -> bool {
    return value() > other;
  }
  constexpr auto operator<=(double other) const -> bool {
    return value() <= other;
  }
  constexpr auto operator>=(double other) const -> bool {
    return value() >= other;
  }
  constexpr auto operator<(const Dual &other) const -> bool {
    return value() < other.value();
  }
  constexpr auto operator>(const Dual &other) const -> bool {
    return value() > other.value();
  }
  constexpr auto operator<=(const Dual &other) const -> bool {
    return value() <= other.value();
  }
  constexpr auto operator>=(const Dual &other) const -> bool {
    return value() >= other.value();
  }

  friend constexpr auto exp(Dual x) -> Dual {
    return {conditional(std::multiplies<>{},
                        elementwise_not_equal(_(0, N + 1), value_idx),
                        exp(x.value()), x.data)};
  }

  friend constexpr auto operator+(double other, Dual x) -> Dual {
    return {other + x.data};
  }
  friend constexpr auto operator-(double other, Dual x) -> Dual {
    return {other - x.data};
  }
  friend constexpr auto operator*(double other, Dual x) -> Dual {
    return {other * x.data};
  }
  friend constexpr auto operator/(double a, Dual b) -> Dual {
    Dual ret;
    V vt = simd::vbroadcast<W, double>(a), vo = b.vbvalue(), vo2 = vo * vo,
      x = vo * simd::Vec<W, double>{a};
    ret.data.memory_[0] =
      (simd::firstoff<W, int64_t>() ? x - vt * b.data.memory_[0] : x) / vo2;
    POLYMATHFULLUNROLL
    for (ptrdiff_t i = 1; i < data_type::L; ++i)
      ret.data.memory_[i] = (-vt * b.data.memory_[i]) / vo2;
    return ret;
    // T v = other / x.value();
    // return {conditional(std::multiplies<>{},
    //                     elementwise_not_equal(_(0, N + 1), value_idx), v,
    //                     -x.data / x.value())};
    // return {v, -v * x.gradient() / (x.value())};
  }
  constexpr void compress(compressed_type *p) const { p->data << data; }
  static constexpr auto decompress(const compressed_type *p) -> Dual {
    return {SVector<T, N + 1, false>{p->data}};
  }
  constexpr operator compressed_type() const {
    compressed_type ret;
    compress(&ret);
    return ret;
  }
};
static_assert(!std::convertible_to<Array<Dual<double, 7, true>,
                                         std::integral_constant<ptrdiff_t, 2>>,
                                   Dual<double, 7, false>>);
static_assert(utils::Compressible<Dual<double, 7>>);
static_assert(utils::Compressible<Dual<double, 8>>);
static_assert(
  AbstractVector<
    Conditional<poly::math::ElementwiseBinaryOp<poly::math::Range<long, long>,
                                                long, std::not_equal_to<void>>,
                poly::math::ElementwiseBinaryOp<
                  double, poly::math::StaticArray<double, 1, 8, false>,
                  std::multiplies<void>>,
                poly::math::ElementwiseBinaryOp<
                  poly::math::StaticArray<double, 1, 8, false>, double,
                  std::multiplies<void>>,
                std::plus<void>>>);

template <typename T> struct IsDualImpl : std::false_type {};
template <typename T, ptrdiff_t N, bool Compress>
struct IsDualImpl<Dual<T, N, Compress>> : std::true_type {};
template <typename T>
concept IsDual = IsDualImpl<T>::value;

// We want to support casting compressed `Dual` arrays to `double`
// when possible as a performance optimization.
// This is possible with
// 1. Dual<T,N> (+/-) Dual<T,N>
// 2. Dual<T,N> * double or double * Dual<T,N>
// 3. Dual<T,N> / double
// 4. Simple copies
template <typename T> struct ScalarizeEltViaCast {
  using type = void;
};
template <typename T>
using scalarize_elt_cast_t = typename ScalarizeEltViaCast<T>::type;
template <typename T, ptrdiff_t N>
struct ScalarizeEltViaCast<Dual<T, N, true>> {
  using type = std::conditional_t<std::same_as<T, double>, double,
                                  scalarize_elt_cast_t<T>>;
};

template <typename T> struct ScalarizeViaCast<Elementwise<std::negate<>, T>> {
  using type = scalarize_via_cast_t<T>;
};
template <typename T>
concept AdditiveOp =
  std::same_as<T, std::plus<>> || std::same_as<T, std::minus<>>;
template <typename T>
concept MultiplicativeOp =
  std::same_as<T, std::multiplies<>> || std::same_as<T, std::divides<>>;
template <typename T>
concept EltIsDual = IsDual<utils::eltype_t<T>>;

template <typename T, typename S> struct ScalarizeViaCast<Array<T, S, true>> {
  using type = scalarize_elt_cast_t<utils::compressed_t<T>>;
};
template <typename T, typename S>
struct ScalarizeViaCast<MutArray<T, S, true>> {
  using type = scalarize_elt_cast_t<utils::compressed_t<T>>;
};

template <typename T>
concept EltCastableDual =
  EltIsDual<T> && std::same_as<scalarize_via_cast_t<T>, double>;
template <AdditiveOp Op, EltCastableDual A, EltCastableDual B>
struct ScalarizeViaCast<ElementwiseBinaryOp<A, B, Op>> {
  // when we cast, we expand into rows, thus col vectors don't work
  // as they'd have to become matrices, and then number of rows
  // won't match up, unless both inputs were a ColVector
  // It is unclear if the case where both inputs are ColVectors is worth
  // the complexity, as the benefit from this optimization is being
  // able to handle things contiguously, which we in that case.
  using type = std::conditional_t<
    (ColVector<A> || ColVector<B>) ||
      (!std::same_as<utils::eltype_t<A>, utils::eltype_t<B>>),
    void, double>;
};
template <MultiplicativeOp Op, EltCastableDual A, std::convertible_to<double> T>
struct ScalarizeViaCast<ElementwiseBinaryOp<A, T, Op>> {
  using type = double;
};
template <EltCastableDual B, std::convertible_to<double> T>
struct ScalarizeViaCast<ElementwiseBinaryOp<T, B, std::multiplies<>>> {
  using type = double;
};

template <class T, ptrdiff_t N> Dual(T, SVector<T, N>) -> Dual<T, N>;

template <class T, ptrdiff_t N>
[[gnu::always_inline]] constexpr auto operator+(const T &a, const Dual<T, N> &b)
  -> Dual<T, N>
requires(!std::same_as<T, double>)
{
  return {a + b.val, b.partials};
}
template <class T, ptrdiff_t N>
[[gnu::always_inline]] constexpr auto operator-(const T &a, const Dual<T, N> &b)
  -> Dual<T, N>
requires(!std::same_as<T, double>)
{
  return {a - b.val, -b.partials};
}
template <class T, ptrdiff_t N>
[[gnu::always_inline]] constexpr auto operator*(const T &a, const Dual<T, N> &b)
  -> Dual<T, N>
requires(!std::same_as<T, double>)
{
  // Dual res;
  // res.val = val * other.val;
  // res.partials << val * other.partials + other.val * partials;
  // return res;
  return {a * b.val, a * b.partials};
}
template <class T, ptrdiff_t N>
[[gnu::always_inline]] constexpr auto operator/(const T &a, const Dual<T, N> &b)
  -> Dual<T, N>
requires(!std::same_as<T, double>)
{
  return {a / b.val, (-a * b.partials) / (b.val * b.val)};
}
template <class T, ptrdiff_t N>
[[gnu::always_inline]] constexpr auto operator+(double other,
                                                const Dual<T, N> &x)
  -> Dual<T, N> {
  return {x.value() + other, x.gradient()};
}
template <class T, ptrdiff_t N>
[[gnu::always_inline]] constexpr auto operator-(double other,
                                                const Dual<T, N> &x)
  -> Dual<T, N> {
  return {x.value() - other, -x.gradient()};
}
template <class T, ptrdiff_t N>
[[gnu::always_inline]] constexpr auto operator*(double other,
                                                const Dual<T, N> &x)
  -> Dual<T, N> {
  return {x.value() * other, other * x.gradient()};
}
template <class T, ptrdiff_t N>
[[gnu::always_inline]] constexpr auto operator/(double other,
                                                const Dual<T, N> &x)
  -> Dual<T, N> {
  return {other / x.value(), -other * x.gradient() / (x.value() * x.value())};
}
template <class T, ptrdiff_t N>
constexpr auto exp(const Dual<T, N> &x) -> Dual<T, N> {
  T expx = exp(x.value());
  return {expx, expx * x.gradient()};
}
template <class T, ptrdiff_t N>
constexpr auto sigmoid(const Dual<T, N> &x) -> Dual<T, N> {
  T s = sigmoid(x.value());
  return {s, (s - s * s) * x.gradient()};
}
template <class T, ptrdiff_t N>
constexpr auto softplus(const Dual<T, N> &x) -> Dual<T, N> {
  return {softplus(x.value()), sigmoid(x.value()) * x.gradient()};
}
template <class T, ptrdiff_t N>
constexpr auto log(const Dual<T, N> &x) -> Dual<T, N> {
  return {log2(x.value()), x.gradient() / x.value()};
}
template <class T, ptrdiff_t N>
constexpr auto log2(const Dual<T, N> &x) -> Dual<T, N> {
  constexpr double log2 = 0.6931471805599453; // log(2);
  return {log2(x.value()), x.gradient() / (log2 * x.value())};
}

constexpr auto dval(double &x) -> double & { return x; }
template <typename T, ptrdiff_t N>
constexpr auto dval(Dual<T, N> &x) -> double & {
  return dval(x.value());
}

class GradientResult {
  double x;
  MutPtrVector<double> grad;

public:
  [[nodiscard]] constexpr auto value() const -> double { return x; }
  [[nodiscard]] constexpr auto gradient() const -> MutPtrVector<double> {
    return grad;
  }
};
class HessianResultCore {
  double *ptr;
  ptrdiff_t dim;

public:
  [[nodiscard]] constexpr auto gradient() const -> MutPtrVector<double> {
    return {ptr, dim};
  }
  [[nodiscard]] constexpr auto hessian() const -> MutSquarePtrMatrix<double> {
    return {ptr + dim, SquareDims<>{{dim}}};
  }
  constexpr HessianResultCore(alloc::Arena<> *alloc, ptrdiff_t d)
    : ptr{alloc->allocate<double>(size_t(d) * (d + 1))}, dim{d} {}
};
class HessianResult : public HessianResultCore {
  double x{};

public:
  [[nodiscard]] constexpr auto value() -> double & { return x; }
  [[nodiscard]] constexpr auto value() const -> double { return x; }

  constexpr HessianResult(alloc::Arena<> *alloc, unsigned d)
    : HessianResultCore{alloc, d} {}

  template <size_t I> constexpr auto get() const {
    if constexpr (I == 0) return x;
    else if constexpr (I == 1) return gradient();
    else return hessian();
  }
};

template <ptrdiff_t N, AbstractVector T> struct DualVector {
  using value_type = Dual<utils::eltype_t<T>, N>;
  static_assert(Trivial<T>);
  T x;
  ptrdiff_t offset;
  [[nodiscard]] constexpr auto operator[](ptrdiff_t i) const -> value_type {
    value_type v{x[i]};
    if ((i >= offset) && (i < offset + N)) dval(v.gradient()[i - offset]) = 1.0;
    return v;
  }
  [[nodiscard]] constexpr auto size() const -> ptrdiff_t { return x.size(); }
  [[nodiscard]] constexpr auto numRow() const -> Row<1> { return {}; }
  [[nodiscard]] constexpr auto numCol() const -> Col<> { return {x.size()}; }
  [[nodiscard]] constexpr auto view() const -> DualVector { return *this; }
};
static_assert(AbstractVector<DualVector<8, PtrVector<double>>>);
static_assert(AbstractVector<DualVector<2, DualVector<8, PtrVector<double>>>>);

template <ptrdiff_t N>
constexpr auto dual(const AbstractVector auto &x, ptrdiff_t offset) {
  return DualVector<N, decltype(x.view())>{x.view(), offset};
}

struct Assign {
  constexpr void operator()(double &x, double y) const { x = y; }
};
struct Increment {
  constexpr void operator()(double &x, double y) const { x += y; }
};
struct ScaledIncrement {
  double scale;
  constexpr void operator()(double &x, double y) const { x += scale * y; }
};

constexpr auto gradient(alloc::Arena<> *arena, PtrVector<double> x,
                        const auto &f) {
  constexpr ptrdiff_t U = 8;
  using D = Dual<double, U>;
  ptrdiff_t N = x.size();
  MutPtrVector<double> grad = vector<double>(arena, N);
  auto p = arena->scope();
  for (ptrdiff_t i = 0;; i += U) {
    D fx = alloc::call(*arena, f, dual<U>(x, i));
    for (ptrdiff_t j = 0; ((j < U) && (i + j < N)); ++j)
      grad[i + j] = fx.gradient()[j];
    if (i + U >= N) return std::make_pair(fx.value(), grad);
  }
}
// only computes the upper triangle blocks
template <class T, ptrdiff_t N, bool Compress>
constexpr auto value(const Dual<T, N, Compress> &x) {
  return value(x.value());
}
// this can call `value` on a compressed `Dual`, whichh is why we need `value`
// to be defined on it.`
template <class T, ptrdiff_t N>
constexpr auto value(utils::Reference<Dual<T, N>> x) {
  return value(x.c->value());
}

/// fills the lower triangle of the hessian
constexpr auto hessian(HessianResultCore hr, PtrVector<double> x, const auto &f,
                       auto update) -> double {
  constexpr ptrdiff_t Ui = 8;
  constexpr ptrdiff_t Uj = 2;
  using D = Dual<double, Ui>;
  using DD = Dual<D, Uj>;
  ptrdiff_t N = x.size();
  MutPtrVector<double> grad = hr.gradient();
  MutSquarePtrMatrix<double> hess = hr.hessian();
  invariant(N == grad.size());
  invariant(N == hess.numCol());
  for (ptrdiff_t j = 0;; j += Uj) {
    bool jbr = j + Uj >= N;
    for (ptrdiff_t i = 0;; i += Ui) {
      // df^2/dx_i dx_j
      bool ibr = i + Ui - Uj >= j;
      // we want to copy into both regions _(j, j+Uj) and _(i, i+Ui)
      // these regions overlap for the last `i` iteration only
      DD fx = f(dual<Uj>(dual<Ui>(x, i), j));
      // DD fx = alloc::call(arena, f, x);
      for (ptrdiff_t k = 0; ((k < Uj) && (j + k < N)); ++k)
        for (ptrdiff_t l = 0; ((l < Ui) && (i + l < N)); ++l)
          update(hess[j + k, i + l], fx.gradient()[k].gradient()[l]);
      if (jbr)
        for (ptrdiff_t k = 0; ((k < Ui) && (i + k < N)); ++k)
          grad[i + k] = fx.value().gradient()[k];
      if (!ibr) continue;
      if (jbr) return fx.value().value();
      break;
    }
  }
}
constexpr auto hessian(HessianResultCore hr, PtrVector<double> x, const auto &f)
  -> double {
  Assign assign{};
  return hessian(hr, x, f, assign);
}

constexpr auto hessian(alloc::Arena<> *arena, PtrVector<double> x,
                       const auto &f) -> HessianResult {
  unsigned N = x.size();
  HessianResult hr{arena, N};
  hr.value() = hessian(hr, x, f);
  return hr;
}
static_assert(MatrixDimension<SquareDims<>>);

} // namespace poly::math
namespace std {
template <> struct tuple_size<poly::math::HessianResult> {
  static constexpr size_t value = 3;
};
template <> struct tuple_element<size_t(0), poly::math::HessianResult> {
  using type = double;
};
template <> struct tuple_element<size_t(1), poly::math::HessianResult> {
  using type = poly::math::MutPtrVector<double>;
};
template <> struct tuple_element<size_t(2), poly::math::HessianResult> {
  using type = poly::math::MutSquarePtrMatrix<double>;
};

} // namespace std
