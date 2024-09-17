#ifdef USE_MODULE
module;
#else
#pragma once
#endif
#include "LoopMacros.hxx"
#ifndef USE_MODULE
#include "Math/Indexing.cxx"
#include "SIMD/Intrin.cxx"
#include "SIMD/Vec.cxx"
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <ostream>
#include <type_traits>
#include <utility>

#include "Alloc/Arena.cxx"
#include "Containers/Pair.cxx"
#include "Math/Array.cxx"
#include "Math/ArrayConcepts.cxx"
#include "Math/AxisTypes.cxx"
#include "Math/Constructors.cxx"
#include "Math/ElementarySIMD.cxx"
#include "Math/ExpressionTemplates.cxx"
#include "Math/MatrixDimensions.cxx"
#include "Math/ScalarizeViaCastArrayOps.cxx"
#include "Math/StaticArrays.cxx"
#include "Utilities/Parameters.cxx"
#include "Utilities/Reference.cxx"
#include "Utilities/TypeCompression.cxx"
#else
export module Dual;

export import Elementary;
import Arena;
import Array;
import ArrayConcepts;
import ArrayConstructors;
import AxisTypes;
import CompressReference;
import ExprTemplates;
import Invariant;
import MatDim;
import Pair;
import Param;
import ScalarizeViaCast;
import SIMD;
import StaticArray;
import STL;
import TypeCompression;
#endif

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif

template <class T, ptrdiff_t N, bool Compress = false> struct Dual {
  static_assert(Compress);
  using CT = utils::compressed_t<T>;
  CT val{};
  SVector<T, N, true> partials{CT{}};

  using decompressed_type = Dual<utils::decompressed_t<T>, N, false>;
  [[gnu::always_inline]] constexpr operator decompressed_type() const {
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

  // [[gnu::always_inline]] constexpr auto operator-() const & -> Dual {
  //   return {-val, -partials};
  // }
  // [[gnu::always_inline]] constexpr auto
  // operator+(const Dual &other) const & -> Dual {
  //   return {val + other.val, partials + other.partials};
  // }
  // [[gnu::always_inline]] constexpr auto operator-(const Dual &other) const
  //   -> Dual {
  //   return {val - other.val, partials - other.partials};
  // }
  // [[gnu::always_inline]] constexpr auto operator+=(const Dual &other)
  //   -> Dual & {
  //   val += other.val;
  //   partials += other.partials;
  //   return *this;
  // }
  // [[gnu::always_inline]] constexpr auto operator-=(const Dual &other)
  //   -> Dual & {
  //   val -= other.val;
  //   partials -= other.partials;
  //   return *this;
  // }
};

template <simd::SIMDSupported T, ptrdiff_t N>
requires(std::popcount(size_t(N)) > 1)
struct Dual<T, N, true> {
  SVector<T, N + 1, true> data{T{}};

  using decompressed_type = Dual<utils::decompressed_t<T>, N, false>;
  [[gnu::always_inline]] constexpr operator decompressed_type() const {
    return decompressed_type::decompress(this);
  }
  [[nodiscard]] constexpr auto value() -> T { return data[0]; }
  [[nodiscard]] constexpr auto value() const -> const T & { return data[0]; }
  [[nodiscard]] constexpr auto gradient() -> MutArray<T, Length<N>> {
    return {data.data() + 1, {}};
  }
  [[nodiscard]] constexpr auto gradient() const -> Array<T, Length<N>> {
    return {data.data() + 1, {}};
  }

  // [[gnu::always_inline]] constexpr auto operator-() const & -> Dual {
  //   return {-data};
  // }
  // [[gnu::always_inline]] constexpr auto
  // operator+(const Dual &other) const & -> Dual {
  //   return {data + other.data};
  // }
  // [[gnu::always_inline]] constexpr auto operator-(const Dual &other) const
  //   -> Dual {
  //   return {data - other.data};
  // }
  // [[gnu::always_inline]] constexpr auto operator+=(const Dual &other)
  //   -> Dual & {
  //   data += other.data;
  //   return *this;
  // }
  // [[gnu::always_inline]] constexpr auto operator-=(const Dual &other)
  //   -> Dual & {
  //   data -= other.data;
  //   return *this;
  // }
};

template <class T, ptrdiff_t N> struct Dual<T, N, false> {
  // default decompressed separates the value and partials
  using data_type = SVector<T, N, false>;
  T val{};
  data_type partials{T{}};

  // using compressed_type = Dual<utils::compressed_t<T>, N, true>;
  using compressed_type = Dual<T, N, true>;
  using decompressed_type = Dual<utils::decompressed_t<T>, N, false>;
  static_assert(std::same_as<Dual, decompressed_type>);

  constexpr Dual() = default;
  constexpr Dual(T v) : val(v) {}
  constexpr Dual(T v, ptrdiff_t n) : val(v) { partials[n] = T{1}; }
  constexpr Dual(T v, data_type g) : val(v), partials(g) {}
  constexpr Dual(T v, AbstractVector auto g) {
    value() = v;
    gradient() << g;
  }
  constexpr Dual(std::integral auto v) : val(v) {}
  constexpr Dual(std::floating_point auto v) : val(v) {}
  // constexpr Dual(const Dual &) = default;
  // constexpr auto operator=(const Dual &) -> Dual & = default;
  constexpr auto value() -> T & { return val; }
  constexpr auto gradient() -> data_type & { return partials; }
  constexpr auto gradient(ptrdiff_t i) -> T & { return partials[i]; }
  [[nodiscard]] constexpr auto value() const -> const T & { return val; }
  [[nodiscard]] constexpr auto gradient() const -> const data_type & {
    return partials;
  }
  [[nodiscard]] constexpr auto gradient(ptrdiff_t i) const -> const T & {
    return partials[i];
  }
  [[gnu::always_inline]] constexpr auto operator-() const -> Dual {
    return {-val, -partials};
  }
  [[gnu::always_inline]] constexpr auto
  operator+(const Dual &other) const -> Dual {
    return {val + other.val, partials + other.partials};
  }
  [[gnu::always_inline]] constexpr auto
  operator-(const Dual &other) const -> Dual {
    return {val - other.val, partials - other.partials};
  }
  [[gnu::always_inline]] constexpr auto
  operator*(const Dual &other) const -> Dual {
#ifndef POLYMATHNOEXPLICITSIMDARRAY
    if constexpr (std::same_as<T, double> && (N > 1)) {
      Dual ret(val * other.val);
      using V = typename data_type::V;
      constexpr ptrdiff_t W = data_type::W;
      V va = simd::vbroadcast<W, double>(val),
        vb = simd::vbroadcast<W, double>(other.val);
      if constexpr (data_type::L == 1) {
        ret.partials.data_ = va * other.partials.data_ + vb * partials.data_;
      } else {
        POLYMATHFULLUNROLL
        for (ptrdiff_t i = 0; i < data_type::L; ++i)
          ret.partials.memory_[i] =
            va * other.partials.memory_[i] + vb * partials.memory_[i];
      }
      return ret;
    } else
#endif
      return {val * other.val, (val * other.partials) + (other.val * partials)};
  }
  [[gnu::always_inline]] constexpr auto
  operator/(const Dual &other) const -> Dual {
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
    return {val - other, partials};
  }
  [[gnu::always_inline]] constexpr auto operator*(const T &other) const -> Dual
  requires(!std::same_as<T, double>)
  {
    return {val * other, partials * other};
  }
  [[gnu::always_inline]] constexpr auto operator/(const T &other) const -> Dual
  requires(!std::same_as<T, double>)
  {
    return {val / other, partials / other};
  }
  [[gnu::always_inline]] constexpr auto
  operator+=(const Dual &other) -> Dual & {
    val += other.val;
    partials += other.partials;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto
  operator-=(const Dual &other) -> Dual & {
    val -= other.val;
    partials -= other.partials;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto
  operator*=(const Dual &other) -> Dual & {
    partials << (val * other.partials) + (other.val * partials);
    val *= other.val;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto
  operator/=(const Dual &other) -> Dual & {
    partials << (other.val * partials - val * other.partials) /
                  (other.val * other.val);
    val /= other.val;
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
    return {val * other, partials * other};
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
  [[gnu::always_inline]] constexpr auto
  operator==(const Dual &other) const -> bool {
    return val == other.val; // && grad == other.grad;
  }
  [[gnu::always_inline]] constexpr auto
  operator!=(const Dual &other) const -> bool {
    return val != other.val; // || grad != other.grad;
  }
  [[gnu::always_inline]] constexpr auto
  operator<(const Dual &other) const -> bool {
    return val < other.val;
  }
  [[gnu::always_inline]] constexpr auto
  operator>(const Dual &other) const -> bool {
    return val > other.val;
  }
  [[gnu::always_inline]] constexpr auto
  operator<=(const Dual &other) const -> bool {
    return val <= other.val;
  }
  [[gnu::always_inline]] constexpr auto
  operator>=(const Dual &other) const -> bool {
    return val >= other.val;
  }
  [[gnu::always_inline]] constexpr auto operator==(double other) const -> bool {
    return val == other;
  }
  [[gnu::always_inline]] constexpr auto operator!=(double other) const -> bool {
    return val != other;
  }
  [[gnu::always_inline]] constexpr auto operator<(double other) const -> bool {
    return val < other;
  }
  [[gnu::always_inline]] constexpr auto operator>(double other) const -> bool {
    return val > other;
  }
  [[gnu::always_inline]] constexpr auto operator<=(double other) const -> bool {
    return val <= other;
  }
  [[gnu::always_inline]] constexpr auto operator>=(double other) const -> bool {
    return val >= other;
  }
  [[gnu::always_inline]] constexpr auto operator==(T other) const -> bool
  requires(!std::same_as<T, double>)
  {
    return val == other;
  }
  [[gnu::always_inline]] constexpr auto operator!=(T other) const -> bool
  requires(!std::same_as<T, double>)
  {
    return val != other;
  }
  [[gnu::always_inline]] constexpr auto operator<(T other) const -> bool
  requires(!std::same_as<T, double>)
  {
    return val < other;
  }
  [[gnu::always_inline]] constexpr auto operator>(T other) const -> bool
  requires(!std::same_as<T, double>)
  {
    return val > other;
  }
  [[gnu::always_inline]] constexpr auto operator<=(T other) const -> bool
  requires(!std::same_as<T, double>)
  {
    return val <= other;
  }
  [[gnu::always_inline]] constexpr auto operator>=(T other) const -> bool
  requires(!std::same_as<T, double>)
  {
    return val >= other;
  }
  [[gnu::always_inline]] constexpr void compress(compressed_type *p) const {
    utils::compress(val, &(p->val));
    partials.compress(&(p->partials));
  }
  [[gnu::always_inline]] static constexpr auto
  decompress(const compressed_type *p) -> Dual {
    return {utils::decompress<T>(&(p->val)),
            SVector<T, N>::decompress(&(p->partials))};
  }
  [[gnu::always_inline]] constexpr operator compressed_type() const {
    compressed_type ret;
    compress(&ret);
    return ret;
  }

private:
  friend constexpr auto value(const Dual &x) -> T { return x.value(); }
  friend constexpr auto extractvalue(const Dual &x) {
    return extractvalue(x.value());
  }
  [[gnu::always_inline]] friend constexpr auto operator>(double other,
                                                         Dual x) -> bool {
    return other > x.value();
  }
  [[gnu::always_inline]] friend constexpr auto operator>=(double other,
                                                          Dual x) -> bool {
    return other >= x.value();
  }
  [[gnu::always_inline]] friend constexpr auto operator<(double other,
                                                         Dual x) -> bool {
    return other < x.value();
  }
  [[gnu::always_inline]] friend constexpr auto operator<=(double other,
                                                          Dual x) -> bool {
    return other <= x.value();
  }
  [[gnu::always_inline]] friend constexpr auto operator==(double other,
                                                          Dual x) -> bool {
    return other == x.value();
  }
  [[gnu::always_inline]] friend constexpr auto operator!=(double other,
                                                          Dual x) -> bool {
    return other != x.value();
  }
  friend auto operator<<(std::ostream &os, const Dual &x) -> std::ostream & {
    os << "Dual<" << N << ">{" << x.value();
    for (ptrdiff_t n = 0; n < N; ++n) os << ", " << x.gradient()[n];
    os << "}";
    return os;
  };
  [[gnu::always_inline]] friend constexpr auto operator+(T a, Dual b) -> Dual
  requires(!std::same_as<T, double>)
  {
    return {a + b.val, b.partials};
  }
  [[gnu::always_inline]] friend constexpr auto operator-(T a, Dual b) -> Dual
  requires(!std::same_as<T, double>)
  {
    return {a - b.val, -b.partials};
  }
  [[gnu::always_inline]] friend constexpr auto operator*(T a, Dual b) -> Dual
  requires(!std::same_as<T, double>)
  {
    // Dual res;
    // res.val = val * other.val;
    // res.partials << val * other.partials + other.val * partials;
    // return res;
    return {a * b.val, a * b.partials};
  }
  [[gnu::always_inline]] friend constexpr auto operator/(T a, Dual b) -> Dual
  requires(!std::same_as<T, double>)
  {
    return {a / b.val, (-a * b.partials) / (b.val * b.val)};
  }
  [[gnu::always_inline]] friend constexpr auto operator+(double other,
                                                         Dual x) -> Dual {
    return {x.value() + other, x.gradient()};
  }
  [[gnu::always_inline]] friend constexpr auto operator-(double other,
                                                         Dual x) -> Dual {
    return {other - x.value(), -x.gradient()};
  }
  [[gnu::always_inline]] friend constexpr auto operator*(double other,
                                                         Dual x) -> Dual {
    return {x.value() * other, other * x.gradient()};
  }
  [[gnu::always_inline]] friend constexpr auto operator/(double other,
                                                         Dual x) -> Dual {
    return {other / x.value(), -other * x.gradient() / (x.value() * x.value())};
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
  constexpr Dual(data_type d) : data{d} {}
  constexpr Dual(const AbstractVector auto &d)
  requires(std::convertible_to<utils::eltype_t<decltype(d)>, T>)
    : data{d} {}
  constexpr Dual(std::integral auto v) { value() = v; }
  constexpr Dual(std::floating_point auto v) { value() = v; }
  constexpr auto value() -> T & { return data[value_idx]; }
  constexpr auto gradient() -> MutArray<T, Length<N>> {
    return {data.data() + partial_offset, {}};
  }
  [[nodiscard]] constexpr auto value() const -> T { return data[value_idx]; }
  // zeros out partial part of the vector
  [[nodiscard]] constexpr auto vvalue() const -> V {
    // return data[value_idx];
    if constexpr (data_type::L == 1)
      return (simd::range<W, int64_t>() == simd::Vec<W, int64_t>{}) ? data.data_
                                                                    : V{};
    else
      return (simd::range<W, int64_t>() == simd::Vec<W, int64_t>{})
               ? data.memory_[0]
               : V{};
  }
  // broadcasts value across register
  [[nodiscard]] constexpr auto vbvalue() const -> V {
    if constexpr (data_type::L == 1) return simd::vbroadcast<W, T>(data.data_);
    else return simd::vbroadcast<W, T>(data.memory_[0]);
  }
  [[nodiscard]] constexpr auto gradient() const -> Array<T, Length<N>> {
    return {data.data() + partial_offset, {}};
  }

  [[gnu::always_inline]] constexpr auto operator-() const -> Dual {
    return {-data};
  }
  // constexpr auto operator+(const Dual &other) const -> Dual {
  //   return {data + other.data};
  // }
  // constexpr auto operator-(const Dual &other) const -> Dual {
  //   return {data - other.data};
  // }
  // constexpr auto operator*(const Dual &other) const -> Dual {
  //   // TODO: either update remaining methods to match this style,
  //   // or figure out how to get `conditional`'s codegen quality to match
  //   if constexpr (data_type::L == 1) {
  //     V vt = vbvalue(), vo = other.vbvalue(), x = vt * other.data.data_;
  //     return {
  //       {simd::fmadd<T>(vo, data.data_, x, simd::firstoff<W, int64_t>())}};
  //   } else {
  //     Dual ret;
  //     V vt = vbvalue(), vo = other.vbvalue(), x = vt * other.data.memory_[0];
  //     ret.data.memory_[0] =
  //       simd::fmadd<T>(vo, data.memory_[0], x, simd::firstoff<W, int64_t>());
  //     POLYMATHFULLUNROLL
  //     for (ptrdiff_t i = 1; i < data_type::L; ++i)
  //       ret.data.memory_[i] = vt * other.data.memory_[i] + vo *
  //       data.memory_[i];
  //     return ret;
  //   }
  //   // return {conditional(std::plus<>{},
  //   //                     elementwise_not_equal(_(0, N + 1), value_idx),
  //   //                     value() * other.data, data * other.value())};
  // }
  // constexpr auto operator/(const Dual &other) const -> Dual {
  //   Dual ret;
  //   if constexpr (data_type::L == 1) {
  //     V vt = vbvalue(), vo = other.vbvalue(), vo2 = vo * vo,
  //       x = vo * data.data_;
  //     ret.data.data_ =
  //       simd::fnmadd<T>(vt, other.data.data_, x, simd::firstoff<W,
  //       int64_t>()) / vo2;
  //   } else {
  //     V vt = vbvalue(), vo = other.vbvalue(), vo2 = vo * vo,
  //       x = vo * data.memory_[0];
  //     ret.data.memory_[0] = simd::fnmadd<T>(vt, other.data.memory_[0], x,
  //                                           simd::firstoff<W, int64_t>()) /
  //                           vo2;
  //     POLYMATHFULLUNROLL
  //     for (ptrdiff_t i = 1; i < data_type::L; ++i)
  //       ret.data.memory_[i] =
  //         (vo * data.memory_[i] - vt * other.data.memory_[i]) / vo2;
  //   }
  //   return ret;
  //   // val = value() / other.value()
  //   // partials = (other.value() * gradient() - value() * other.gradient()) /
  //   // (other.value() * other.value())
  //   // partials = (gradient()) / (other.value())
  //   //  - (value() * other.gradient()) / (other.value() * other.value())
  //   // T v{other.value()};
  //   // return {conditional(std::minus<>{},
  //   //                     elementwise_not_equal(_(0, N + 1), value_idx),
  //   data /
  //   //                     v, value() * other.data / (v * v))};
  // }
  [[gnu::always_inline]] constexpr auto operator+=(Dual other) -> Dual & {
    data += other.data;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator-=(Dual other) -> Dual & {
    data -= other.data;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator*=(Dual other) -> Dual & {
    if constexpr (data_type::L == 1) {
      V vt = vbvalue(), vo = other.vbvalue(), x = vt * other.data.data_;
      data.data_ =
        simd::fmadd<T>(vo, data.data_, x, simd::firstoff<W, int64_t>());
    } else {
      V vt = vbvalue(), vo = other.vbvalue(), x = vt * other.data.memory_[0];
      data.memory_[0] =
        simd::fmadd<T>(vo, data.memory_[0], x, simd::firstoff<W, int64_t>());
      POLYMATHFULLUNROLL
      for (ptrdiff_t i = 1; i < data_type::L; ++i)
        data.memory_[i] = vt * other.data.memory_[i] + vo * data.memory_[i];
    }
    // data << conditional(std::plus<>{},
    //                     elementwise_not_equal(_(0, N + 1), value_idx),
    //                     value() * other.data, data * other.value());
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator/=(Dual other) -> Dual & {
    if constexpr (data_type::L == 1) {
      V vt = vbvalue(), vo = other.vbvalue(), vo2 = vo * vo,
        x = vo * data.data_;
      data.data_ =
        simd::fnmadd<T>(vt, other.data.data_, x, simd::firstoff<W, int64_t>()) /
        vo2;
    } else {
      V vt = vbvalue(), vo = other.vbvalue(), vo2 = vo * vo,
        x = vo * data.memory_[0];
      data.memory_[0] = simd::fnmadd<T>(vt, other.data.memory_[0], x,
                                        simd::firstoff<W, int64_t>()) /
                        vo2;
      POLYMATHFULLUNROLL
      for (ptrdiff_t i = 1; i < data_type::L; ++i)
        data.memory_[i] =
          (vo * data.memory_[i] - vt * other.data.memory_[i]) / vo2;
    }
    // T v{other.value()};
    // data << conditional(std::minus<>{},
    //                     elementwise_not_equal(_(0, N + 1), value_idx), data /
    //                     v, value() * other.data / (v * v));
    return *this;
  }
  // constexpr auto operator+(double other) const -> Dual {
  //   Dual ret = *this;
  //   if constexpr (data_type::L == 1)
  //     ret.data.data_ += simd::Vec<SVector<T, N + 1>::W, T>{other};
  //   else ret.data.memory_[0] += simd::Vec<SVector<T, N + 1>::W, T>{other};
  //   return ret;
  // }
  // constexpr auto operator-(double other) const -> Dual {
  //   Dual ret = *this;
  //   if constexpr (data_type::L == 1)
  //     ret.data.data_ -= simd::Vec<SVector<T, N + 1>::W, T>{other};
  //   else ret.data.memory_[0] -= simd::Vec<SVector<T, N + 1>::W, T>{other};
  //   return ret;
  // }
  // constexpr auto operator*(double other) const -> Dual {
  //   return {data * other};
  // }
  // constexpr auto operator/(double other) const -> Dual {
  //   return {data / other};
  // }
  [[gnu::always_inline]] constexpr auto operator+=(double other) -> Dual & {
    if constexpr (data_type::L == 1)
      data.data_ += simd::Vec<SVector<T, N + 1>::W, T>{other};
    else data.memory_[0] += simd::Vec<SVector<T, N + 1>::W, T>{other};
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator-=(double other) -> Dual & {
    if constexpr (data_type::L == 1)
      data.data_ -= simd::Vec<SVector<T, N + 1>::W, T>{other};
    else data.memory_[0] -= simd::Vec<SVector<T, N + 1>::W, T>{other};
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator*=(double other) -> Dual & {
    data *= other;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto operator/=(double other) -> Dual & {
    data /= other;
    return *this;
  }
  [[gnu::always_inline]] constexpr auto
  operator==(const Dual &other) const -> bool {
    return value() == other.value(); // && grad == other.grad;
  }
  [[gnu::always_inline]] constexpr auto
  operator!=(const Dual &other) const -> bool {
    return value() != other.value(); // || grad != other.grad;
  }
  [[gnu::always_inline]] constexpr auto operator==(double other) const -> bool {
    return value() == other;
  }
  [[gnu::always_inline]] constexpr auto operator!=(double other) const -> bool {
    return value() != other;
  }
  [[gnu::always_inline]] constexpr auto operator<(double other) const -> bool {
    return value() < other;
  }
  [[gnu::always_inline]] constexpr auto operator>(double other) const -> bool {
    return value() > other;
  }
  [[gnu::always_inline]] constexpr auto operator<=(double other) const -> bool {
    return value() <= other;
  }
  [[gnu::always_inline]] constexpr auto operator>=(double other) const -> bool {
    return value() >= other;
  }
  [[gnu::always_inline]] constexpr auto
  operator<(const Dual &other) const -> bool {
    return value() < other.value();
  }
  [[gnu::always_inline]] constexpr auto
  operator>(const Dual &other) const -> bool {
    return value() > other.value();
  }
  [[gnu::always_inline]] constexpr auto
  operator<=(const Dual &other) const -> bool {
    return value() <= other.value();
  }
  [[gnu::always_inline]] constexpr auto
  operator>=(const Dual &other) const -> bool {
    return value() >= other.value();
  }
  [[gnu::always_inline]] static constexpr auto
  decompress(const compressed_type *p) -> Dual {
    return {SVector<T, N + 1, false>{p->data}};
  }
  [[gnu::always_inline]] constexpr operator compressed_type() const {
    compressed_type ret;
    compress(&ret);
    return ret;
  }
  [[gnu::always_inline]] constexpr void compress(compressed_type *p) const {
    p->data << data;
  }

private:
  friend constexpr auto value(const Dual &x) -> T { return x.value(); }
  friend constexpr auto extractvalue(const Dual &x) -> T { return x.value(); }
  friend constexpr auto exp(Dual x) -> Dual {
    return {conditional(std::multiplies<>{},
                        elementwise_not_equal(_(0, N + 1), value_idx),
                        exp(x.value()), x.data)};
  }
  [[gnu::always_inline]] friend constexpr auto operator/(double a,
                                                         Dual b) -> Dual {
    Dual ret;
    if constexpr (data_type::L == 1) {
      V vt = simd::vbroadcast<W, double>(a), vo = b.vbvalue(), vo2 = vo * vo,
        x = vo * simd::Vec<W, double>{a};
      ret.data.data_ =
        simd::fnmadd<T>(vt, b.data.data_, x, simd::firstoff<W, int64_t>()) /
        vo2;
    } else {
      V vt = simd::vbroadcast<W, double>(a), vo = b.vbvalue(), vo2 = vo * vo,
        x = vo * simd::Vec<W, double>{a};
      ret.data.memory_[0] = simd::fnmadd<T>(vt, b.data.memory_[0], x,
                                            simd::firstoff<W, int64_t>()) /
                            vo2;
      POLYMATHFULLUNROLL
      for (ptrdiff_t i = 1; i < data_type::L; ++i)
        ret.data.memory_[i] = (-vt * b.data.memory_[i]) / vo2;
    }
    return ret;
    // T v = other / x.value();
    // return {conditional(std::multiplies<>{},
    //                     elementwise_not_equal(_(0, N + 1), value_idx), v,
    //                     -x.data / x.value())};
    // return {v, -v * x.gradient() / (x.value())};
  }
  [[gnu::always_inline]] friend constexpr auto operator>(double other,
                                                         Dual x) -> bool {
    return other > x.value();
  }
  [[gnu::always_inline]] friend constexpr auto operator>=(double other,
                                                          Dual x) -> bool {
    return other >= x.value();
  }
  [[gnu::always_inline]] friend constexpr auto operator<(double other,
                                                         Dual x) -> bool {
    return other < x.value();
  }
  [[gnu::always_inline]] friend constexpr auto operator<=(double other,
                                                          Dual x) -> bool {
    return other <= x.value();
  }
  [[gnu::always_inline]] friend constexpr auto operator==(double other,
                                                          Dual x) -> bool {
    return other == x.value();
  }
  [[gnu::always_inline]] friend constexpr auto operator!=(double other,
                                                          Dual x) -> bool {
    return other != x.value();
  }
  friend auto operator<<(std::ostream &os, const Dual &x) -> std::ostream & {
    os << "Dual<" << N << ">{" << x.value();
    for (ptrdiff_t n = 0; n < N; ++n) os << ", " << x.gradient()[n];
    os << "}";
    return os;
  };

  [[gnu::always_inline]] friend constexpr auto operator+(Dual x,
                                                         Dual y) -> Dual {
    return {x.data + y.data};
  }

  [[gnu::always_inline]] friend constexpr auto operator-(Dual x,
                                                         Dual y) -> Dual {
    return {x.data - y.data};
  }

  [[gnu::always_inline]] friend constexpr auto operator*(Dual a,
                                                         Dual b) -> Dual {
    using D = Dual<T, N, false>;
    if constexpr (data_type::L == 1) {
      V vt = a.vbvalue(), vo = b.vbvalue(), x = vt * b.data.data_;
      return {
        {simd::fmadd<T>(vo, a.data.data_, x, simd::firstoff<D::W, int64_t>())}};
    } else {
      Dual<T, N, false> ret;
      V vt = a.vbvalue(), vo = b.vbvalue(), x = vt * b.data.memory_[0];
      ret.data.memory_[0] = simd::fmadd<T>(vo, a.data.memory_[0], x,
                                           simd::firstoff<D::W, int64_t>());
      POLYMATHFULLUNROLL
      for (ptrdiff_t i = 1; i < data_type::L; ++i)
        ret.data.memory_[i] = vt * b.data.memory_[i] + vo * a.data.memory_[i];
      return ret;
    }
  }
  [[gnu::always_inline]] friend constexpr auto operator/(Dual a,
                                                         Dual b) -> Dual {
    Dual ret;
    if constexpr (data_type::L == 1) {
      V vt = a.vbvalue(), vo = b.vbvalue(), vo2 = vo * vo,
        x = vo * a.data.data_;
      ret.data.data_ =
        simd::fnmadd<T>(vt, b.data.data_, x, simd::firstoff<W, int64_t>()) /
        vo2;
    } else {
      V vt = a.vbvalue(), vo = b.vbvalue(), vo2 = vo * vo,
        x = vo * a.data.memory_[0];
      ret.data.memory_[0] = simd::fnmadd<T>(vt, b.data.memory_[0], x,
                                            simd::firstoff<W, int64_t>()) /
                            vo2;
      POLYMATHFULLUNROLL
      for (ptrdiff_t i = 1; i < data_type::L; ++i)
        ret.data.memory_[i] =
          (vo * a.data.memory_[i] - vt * b.data.memory_[i]) / vo2;
    }
    return ret;
  }
  [[gnu::always_inline]] friend constexpr auto operator+(Dual a,
                                                         double b) -> Dual {
    if constexpr (data_type::L == 1)
      a.data.data_ += simd::Vec<SVector<T, N + 1>::W, T>{b};
    else a.data.memory_[0] += simd::Vec<SVector<T, N + 1>::W, T>{b};
    return a;
  }

  [[gnu::always_inline]] friend constexpr auto operator+(double a,
                                                         Dual b) -> Dual {
    if constexpr (data_type::L == 1)
      b.data.data_ += simd::Vec<SVector<T, N + 1>::W, T>{a};
    else b.data.memory_[0] += simd::Vec<SVector<T, N + 1>::W, T>{a};
    return b;
  }

  [[gnu::always_inline]] friend constexpr auto operator-(Dual a,
                                                         double b) -> Dual {
    if constexpr (data_type::L == 1)
      a.data.data_ -= simd::Vec<SVector<T, N + 1>::W, T>{b};
    else a.data.memory_[0] -= simd::Vec<SVector<T, N + 1>::W, T>{b};
    return a;
  }
  [[gnu::always_inline]] friend constexpr auto operator-(double a,
                                                         Dual b) -> Dual {
    if constexpr (data_type::L == 1)
      b.data.data_ = simd::Vec<SVector<T, N + 1>::W, T>{a} - b.data.data_;
    else {
      b.data.memory_[0] =
        simd::Vec<SVector<T, N + 1>::W, T>{a} - b.data.memory_[0];
      POLYMATHFULLUNROLL
      for (ptrdiff_t l = 1; l < data_type::L; ++l)
        b.data.memory_[l] = -b.data.memory_[l];
    }
    return b;
  }
  [[gnu::always_inline]] friend constexpr auto operator*(Dual a,
                                                         double b) -> Dual {
    return {a.data * b};
  }
  [[gnu::always_inline]] friend constexpr auto operator*(double a,
                                                         Dual b) -> Dual {
    return {a * b.data};
  }

  [[gnu::always_inline]] friend constexpr auto operator/(Dual a,
                                                         double b) -> Dual {
    return {a.data / b};
  }
};

static_assert(!std::convertible_to<Array<Dual<double, 7, true>, Length<2>>,
                                   Dual<double, 7, false>>);
static_assert(utils::Compressible<Dual<double, 7>>);
static_assert(utils::Compressible<Dual<double, 8>>);
static_assert(sizeof(Dual<Dual<double, 8, true>, 2, true>) ==
              sizeof(double) * 9UL * 3UL);
static_assert(sizeof(Dual<Dual<double, 8>, 2>) ==
              sizeof(double) * (8 + simd::Width<double>)*3UL);
// static_assert(
//   AbstractVector<Conditional<
//     ::math::ElementwiseBinaryOp<::math::Range<long, long>, long,
//                                 std::not_equal_to<void>>,
//     ::math::ElementwiseBinaryOp<
//       double, ::math::StaticArray<double, 1, 8, false>,
//       std::multiplies<void>>,
//     ::math::ElementwiseBinaryOp<::math::StaticArray<double, 1, 8, false>,
//                                 double, std::multiplies<void>>,
//     std::plus<void>>>);

template <class T, ptrdiff_t N> Dual(T, SVector<T, N>) -> Dual<T, N>;

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
  constexpr double logof2 = 0.6931471805599453; // log(2);
  return {log2(x.value()) * logof2, x.gradient() / x.value()};
}
template <class T, ptrdiff_t N>
constexpr auto log2(const Dual<T, N> &x) -> Dual<T, N> {
  constexpr double logof2 = 0.6931471805599453; // log(2);
  return {log2(x.value()), x.gradient() / (logof2 * x.value())};
}
template <class T, ptrdiff_t N>
constexpr auto log1p(const Dual<T, N> &x) -> Dual<T, N> {
  // d log(1+x)/dx = dx/1+x
  return {log1p(x.value()), x.gradient() / (1.0 + x.value())};
}

// Reference support...
template <class T, ptrdiff_t N>
constexpr auto exp(utils::Reference<Dual<T, N>> x) -> Dual<T, N> {
  return exp(Dual<T, N>{x});
}
template <class T, ptrdiff_t N>
constexpr auto sigmoid(utils::Reference<Dual<T, N>> x) -> Dual<T, N> {
  return sigmoid(Dual<T, N>{x});
}
template <class T, ptrdiff_t N>
constexpr auto log(utils::Reference<Dual<T, N>> x) -> Dual<T, N> {
  return log(Dual<T, N>{x});
}
template <class T, ptrdiff_t N>
constexpr auto log2(utils::Reference<Dual<T, N>> x) -> Dual<T, N> {
  return log2(Dual<T, N>{x});
}
template <class T, ptrdiff_t N>
constexpr auto log1p(utils::Reference<Dual<T, N>> x) -> Dual<T, N> {
  return log1p(Dual<T, N>{x});
}
template <int l = 8> constexpr auto smax(auto x, auto y, auto z) {
  double m =
    std::max(std::max(extractvalue(x), extractvalue(y)), extractvalue(z));
  static constexpr double f = l, i = 1 / f;
  return m + i * log(exp(f * (x - m)) + exp(f * (y - m)) + exp(f * (z - m)));
}
template <int l = 8> constexpr auto smax(auto w, auto x, auto y, auto z) {
  double m = std::max(std::max(extractvalue(w), extractvalue(y)),
                      std::max(extractvalue(x), extractvalue(z)));
  static constexpr double f = l, i = 1 / f;
  return m + i * log(exp(f * (w - m)) + exp(f * (x - m)) + exp(f * (y - m)) +
                     exp(f * (z - m)));
}
template <int l = 8, typename T, ptrdiff_t N>
constexpr auto smax(SVector<T, N> x) -> T {
  static_assert(!std::is_integral_v<T>);
  static constexpr double f = l, i = 1 / f;
  double m = -std::numeric_limits<double>::max();
  for (ptrdiff_t n = 0; n < N; ++n) m = std::max(m, extractvalue(x[n]));
  T a{};
  for (ptrdiff_t n = 0; n < N; ++n) a += exp(f * (x[n] - m));
  return m + i * log(a);
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
    return {ptr, length(dim)};
  }
  [[nodiscard]] constexpr auto hessian() const -> MutSquarePtrMatrix<double> {
    return {ptr + dim, SquareDims<>{row(dim)}};
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

template <ptrdiff_t N, AbstractVector T>
struct DualVector : Expr<Dual<utils::eltype_t<T>, N>, DualVector<N, T>> {
  using value_type = Dual<utils::eltype_t<T>, N>;
  static_assert(utils::TriviallyCopyable<T>);
  T x;
  ptrdiff_t offset;
  [[nodiscard]] constexpr auto operator[](ptrdiff_t i) const -> value_type {
    value_type v{x[i]};
    if ((i >= offset) && (i < offset + N)) dval(v.gradient()[i - offset]) = 1.0;
    return v;
  }
  [[nodiscard]] constexpr auto size() const -> ptrdiff_t { return x.size(); }
  [[nodiscard]] constexpr auto numRow() const -> Row<1> { return {}; }
  [[nodiscard]] constexpr auto numCol() const -> Col<> { return col(x.size()); }
  [[nodiscard]] constexpr auto view() const -> DualVector { return *this; }
};
static_assert(AbstractVector<DualVector<8, PtrVector<double>>>);
static_assert(AbstractVector<DualVector<2, DualVector<8, PtrVector<double>>>>);

template <ptrdiff_t N>
constexpr auto dual(const AbstractVector auto &x, ptrdiff_t offset) {
  return DualVector<N, decltype(x.view())>{.x = x.view(), .offset = offset};
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

constexpr auto
gradient(alloc::Arena<> *arena, PtrVector<double> x,
         const auto &f) -> containers::Pair<double, MutPtrVector<double>> {
  constexpr ptrdiff_t U = 8;
  using D = Dual<double, U>;
  ptrdiff_t N = x.size();
  MutPtrVector<double> grad = vector<double>(arena, N);
  auto p = arena->scope();
  for (ptrdiff_t i = 0;; i += U) {
    D fx = alloc::call(*arena, f, dual<U>(x, i));
    for (ptrdiff_t j = 0; ((j < U) && (i + j < N)); ++j)
      grad[i + j] = fx.gradient()[j];
    if (i + U >= N) return {fx.value(), grad};
  }
}
//

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
constexpr auto hessian(HessianResultCore hr, PtrVector<double> x,
                       const auto &f) -> double {
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
static_assert(std::same_as<utils::compressed_t<Dual<Dual<double, 8>, 2>>,
                           Dual<Dual<double, 8>, 2, true>>);

template <typename T, ptrdiff_t N, bool Compress>
struct IsDualImpl<::math::Dual<T, N, Compress>> : std::true_type {};

template <typename T, ptrdiff_t N>
struct ScalarizeEltViaCast<Dual<T, N, true>> {
  using type = std::conditional_t<std::same_as<T, double>, double,
                                  scalarize_elt_cast_t<utils::compressed_t<T>>>;
};

} // namespace math

#ifdef USE_MODULE
export {
#endif
  template <> struct std::tuple_size<math::HessianResult> {
    static constexpr size_t value = 3;
  };
  template <> struct std::tuple_element<size_t(0), math::HessianResult> {
    using type = double;
  };
  template <> struct std::tuple_element<size_t(1), math::HessianResult> {
    using type = math::MutPtrVector<double>;
  };
  template <> struct std::tuple_element<size_t(2), math::HessianResult> {
    using type = math::MutSquarePtrMatrix<double>;
  };

#ifdef USE_MODULE
} // namespace std
#endif
