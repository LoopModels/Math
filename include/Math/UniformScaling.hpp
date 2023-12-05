#pragma once
#include "Math/Matrix.hpp"
#include "SIMD/Indexing.hpp"
#include "SIMD/Masks.hpp"
#include "SIMD/Unroll.hpp"
namespace poly::math {

template <class T> struct UniformScaling {
  using value_type = T;
  T value;
  constexpr UniformScaling(T x) : value(x) {}
  constexpr auto operator[](ptrdiff_t r, ptrdiff_t c) const -> T {
    return r == c ? value : T{};
  }
  template <ptrdiff_t R, ptrdiff_t C, ptrdiff_t W, typename M>
  [[gnu::always_inline]] constexpr auto
  operator[](simd::index::Unroll<R> r, simd::index::Unroll<C, W, M> c) const
    -> simd::Unroll<R, C, W, T> {
    using I = simd::IntegerOfSize<T>;
    using VI = simd::Vec<W, I>;
    simd::Vec<W, T> vz{}, vv = simd::vbroadcast<W, T>(value);
    if constexpr (R * C == 1) {
      return {(simd::vbroadcast<W, I>(r.index - c.index) == simd::range<W, I>())
                ? vv
                : vz};
    } else {
      simd::Unroll<R, C, W, T> ret;
      POLYMATHFULLUNROLL
      for (ptrdiff_t i = 0; i < R; ++i) {
        VI vr = simd::vbroadcast<W, I>(i + r.index),
           vc = simd::range<W, I>() + c.index;
        POLYMATHFULLUNROLL
        for (ptrdiff_t j = 0; j < C; ++j, vc += W)
          ret[i, j] = (vr == vc) ? vv : vz;
      }
      return ret;
    }
  }
  template <ptrdiff_t C, ptrdiff_t W, typename M>
  [[gnu::always_inline]] constexpr auto
  operator[](ptrdiff_t r, simd::index::Unroll<C, W, M> c) const
    -> simd::Unroll<1, C, W, T> {
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
    else return UniformScaling<U>{value * x};
  }
  constexpr auto isEqual(const AbstractMatrix auto &A) const -> bool {
    auto R = ptrdiff_t(A.numRow());
    if (R != A.numCol()) return false;
    for (ptrdiff_t r = 0; r < R; ++r)
      for (ptrdiff_t c = 0; c < R; ++c)
        if (A[r, c] != ((r == c) * value)) return false;
    return true;
  }
  constexpr auto operator==(const AbstractMatrix auto &A) const -> bool {
    return isEqual(A);
  }
  friend inline auto operator<<(std::ostream &os, UniformScaling S)
    -> std::ostream & {
    return os << "UniformScaling(" << S.value << ")";
  }
  [[nodiscard]] constexpr auto t() const -> UniformScaling { return *this; }
};
template <class T>
constexpr auto operator==(const AbstractMatrix auto &A,
                          const UniformScaling<T> &B) -> bool {
  return B.isEqual(A);
}
template <class T, class U>
constexpr auto operator*(const U &x, UniformScaling<T> d) {
  if constexpr (std::is_same_v<std::remove_cvref_t<T>, std::true_type>)
    return UniformScaling<U>{x};
  else return UniformScaling<U>{d.value * x};
}

[[maybe_unused]] static constexpr inline UniformScaling<std::true_type> I{
  std::true_type{}}; // identity

template <class T> UniformScaling(T) -> UniformScaling<T>;
static_assert(AbstractMatrix<UniformScaling<int64_t>>);

} // namespace poly::math
