#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#include "LoopMacros.hxx"
#ifndef USE_MODULE
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <type_traits>

#include "Math/ArrayConcepts.cxx"
#include "Math/AxisTypes.cxx"
#include "Math/ExpressionTemplates.cxx"
#include "SIMD/SIMD.cxx"
#include "Utilities/Widen.cxx"
#else
export module UniformScaling;

import ArrayConcepts;
import AxisTypes;
import ExprTemplates;
import SIMD;
import STL;
import Widen;
#endif

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif

template <class T> struct UniformScaling {
  using value_type = T;
  T value_;
  constexpr UniformScaling(T x) : value_(x) {}
  constexpr auto operator[](ptrdiff_t r, ptrdiff_t c) const -> T {
    return r == c ? value_ : T{};
  }
  template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename M>
  [[gnu::always_inline]] constexpr auto
  operator[](simd::index::Unroll<R> r,
             simd::index::Unroll<C, W, M> c) const -> simd::Unroll<R, C, W, T> {
    using I = utils::signed_integer_t<sizeof(T)>;
    using VI = simd::Vec<W, I>;
    simd::Vec<W, T> vz{}, vv = simd::vbroadcast<W, T>(value_);
    if constexpr (R * C == 1) {
      return {
        (simd::vbroadcast<W, I>(r.index_ - c.index_) == simd::range<W, I>())
          ? vv
          : vz};
    } else {
      simd::Unroll<R, C, W, T> ret;
      POLYMATHFULLUNROLL
      for (ptrdiff_t i = 0; i < R; ++i) {
        VI vr = simd::vbroadcast<W, I>(i + r.index_),
           vc = simd::range<W, I>() + c.index_;
        POLYMATHFULLUNROLL
        for (ptrdiff_t j = 0; j < C; ++j, vc += W)
          ret[i, j] = (vr == vc) ? vv : vz;
      }
      return ret;
    }
  }
  template <ptrdiff_t C, ptrdiff_t W, typename M>
  [[gnu::always_inline]] constexpr auto
  operator[](ptrdiff_t r,
             simd::index::Unroll<C, W, M> c) const -> simd::Unroll<1, C, W, T> {
    return (*this)[simd::index::Unroll<1>{r}, c];
  }
  // template <ptrdiff_t C, ptrdiff_t W, typename M>
  // [[gnu::always_inline]] constexpr auto
  // operator[](simd::index::Unroll<C, W, M> r, ptrdiff_t c) const
  //   -> simd::Unroll<1, C, W, T> {
  //   return (*this)[r, simd::index::Unroll<1>{c}];
  // }

  static constexpr auto numRow() -> Row<0> { return {}; }
  static constexpr auto numCol() -> Col<0> { return {}; }
  static constexpr auto size() -> std::integral_constant<ptrdiff_t, 0> {
    return {};
  }
  static constexpr auto shape() -> CartesianIndex<ptrdiff_t, ptrdiff_t> {
    return {0, 0};
  }
  static constexpr auto dim() -> DenseDims<0, 0> { return {{}, {}}; }
  [[nodiscard]] constexpr auto view() const -> auto { return *this; };
  template <class U> constexpr auto operator*(const U &x) const {
    if constexpr (std::is_same_v<std::remove_cvref_t<T>, std::true_type>)
      return UniformScaling<U>{x};
    else return UniformScaling<U>{value_ * x};
  }
  constexpr auto isEqual(const AbstractMatrix auto &A) const -> bool {
    auto R = ptrdiff_t(A.numRow());
    if (R != A.numCol()) return false;
    for (ptrdiff_t r = 0; r < R; ++r)
      for (ptrdiff_t c = 0; c < R; ++c)
        if (A[r, c] != ((r == c) * value_)) return false;
    return true;
  }
  constexpr auto operator==(const AbstractMatrix auto &A) const -> bool {
    return isEqual(A);
  }

  constexpr auto operator-() const -> UniformScaling { return -value_; }

  constexpr auto operator+(const auto &b) const {
    return elementwise(*this, view(b), std::plus<>{});
  }
  constexpr auto operator-(const auto &b) const {
    return elementwise(*this, view(b), std::minus<>{});
  }
  template <typename B>
  constexpr auto operator*(const B &b) const
  requires(std::common_with<std::remove_cvref_t<B>, T> || AbstractTensor<B>)
  {
    if constexpr (!std::common_with<std::remove_cvref_t<B>, T>) {
      auto BB{b.view()};
      return elementwise(value_, BB, std::multiplies<>{});
    } else
      return UniformScaling<std::common_type_t<T, std::remove_reference_t<B>>>(
        value_ * b);
  }

  template <std::common_with<T> S>
  constexpr auto operator+(UniformScaling<S> b) const {
    return UniformScaling<std::common_type_t<S, T>>(value_ + b.value_);
  }
  template <std::common_with<T> S>
  constexpr auto operator-(UniformScaling<S> b) const {
    return UniformScaling<std::common_type_t<S, T>>(value_ - b.value_);
  }
  template <std::common_with<T> S>
  constexpr auto operator*(UniformScaling<S> b) const {
    return UniformScaling<std::common_type_t<S, T>>(value_ * b.value_);
  }

private:
  friend constexpr auto operator+(std::common_with<T> auto b,
                                  UniformScaling a) {
    return elementwise(b, a, std::plus<>{});
  }
  friend constexpr auto operator-(std::common_with<T> auto b,
                                  UniformScaling a) {
    return elementwise(b, a, std::minus<>{});
  }
  friend constexpr auto operator/(std::common_with<T> auto b,
                                  UniformScaling a) {
    return elementwise(b, a.value_, std::divides<>{});
  }
  friend constexpr auto operator%(std::common_with<T> auto b,
                                  UniformScaling a) {
    return elementwise(b, a.value_, std::modulus<>{});
  }
  // friend constexpr auto operator&(std::common_with<T> auto b, UniformScaling
  // a) {
  //   return elementwise(b, a.view(), std::bit_and<>{});
  // }
  // friend constexpr auto operator|(std::common_with<T> auto b, UniformScaling
  // a) {
  //   return elementwise(b, a.vie(), std::bit_or<>{});
  // }
  // friend constexpr auto operator^(std::common_with<T> auto b, UniformScaling
  // a) {
  //   return elementwise(b, a.view(), std::bit_xor<>{});
  // }
  template <typename B>
  friend constexpr auto operator*(const B &b, UniformScaling a)
  requires(std::common_with<std::remove_cvref_t<B>, T> || AbstractTensor<B>)
  {
    if constexpr (!std::common_with<std::remove_cvref_t<B>, T>) {
      auto BB{b.view()};
      return elementwise(BB, a.value_, std::multiplies<>{});
    } else
      return UniformScaling<std::common_type_t<T, std::remove_reference_t<B>>>(
        b * a.value_);
  }

  friend inline auto operator<<(std::ostream &os,
                                UniformScaling S) -> std::ostream & {
    return os << "UniformScaling(" << S.value_ << ")";
  }
  [[nodiscard]] constexpr auto t() const -> UniformScaling { return *this; }
  friend constexpr auto operator==(const AbstractMatrix auto &A,
                                   const UniformScaling &B) -> bool {
    return B.isEqual(A);
  }
  template <class U>
  friend constexpr auto operator*(const U &x, UniformScaling d) {
    if constexpr (std::is_same_v<std::remove_cvref_t<T>, std::true_type>)
      return UniformScaling<U>{x};
    else return UniformScaling<U>{d.value_ * x};
  }
};

[[maybe_unused]] constexpr inline UniformScaling<std::true_type> I{
  std::true_type{}}; // identity

template <class T> UniformScaling(T) -> UniformScaling<T>;
static_assert(AbstractMatrix<UniformScaling<int64_t>>);

} // namespace math
