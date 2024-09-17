#ifdef USE_MODULE
module;
#else
#pragma once
#endif
// We'll follow Julia style, so anything that's not a constructor, destructor,
// nor an operator will be outside of the struct/class.

#include "LoopMacros.hxx"
#ifndef USE_MODULE
#include "Math/ArrayConcepts.cxx"
#include "Math/AxisTypes.cxx"
#include "Math/CheckSizes.cxx"
#include "Math/MatrixDimensions.cxx"
#include "Math/Ranges.cxx"
#include "Math/Reductions.cxx"
#include "Math/ScalarizeViaCastArrayOps.cxx"
#include "SIMD/Unroll.cxx"
#include "SIMD/UnrollIndex.cxx"
#include "SIMD/Vec.cxx"
#include "Utilities/Parameters.cxx"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>
#else
export module ExprTemplates;

import ArrayConcepts;
import AxisTypes;
import CheckSizes;
import Indexing;
import Param;
import Range;
import Reductions;
import ScalarizeViaCast;
import SIMD;
import STL;
#endif

using utils::TriviallyCopyable;

template <typename T, typename I> consteval auto getWidth() -> ptrdiff_t {
  if constexpr (std::same_as<I, ptrdiff_t>) return simd::Width<T>;
  else return simd::VecLen<ptrdiff_t(I{}), T>;
}

using namespace math;

template <class T, class C>
concept Compatible =
  (AbstractTensor<C> && std::convertible_to<T, utils::eltype_t<C>>) ||
  (AbstractTensor<T> && std::convertible_to<C, utils::eltype_t<T>>) ||
  (AbstractVector<C> && AbstractVector<T>) ||
  (AbstractMatrix<C> && AbstractMatrix<T>);
template <typename T, typename C>
concept TrivialCompatible = utils::TriviallyCopyable<T> && Compatible<T, C>;
template <typename T>
concept TrivialTensor = utils::TriviallyCopyable<T> && AbstractTensor<T>;

// returns the element type of `A` when used in a binary op with `B`
template <class A, class B>
using indextype_t =
  std::conditional_t<AbstractTensor<B> &&
                       std::convertible_to<A, utils::eltype_t<B>>,
                     A, utils::eltype_t<A>>;

template <typename Op, typename A>
concept FuncOfElt = std::is_invocable_v<Op, utils::eltype_t<A>>;
template <typename Op, typename A, typename B>
concept BinaryFuncOfElts =
  std::is_invocable_v<Op, indextype_t<A, B>, indextype_t<B, A>>;

template <utils::TriviallyCopyable A, FuncOfElt<A> Op> struct Elementwise;

constexpr auto size(const std::integral auto) -> ptrdiff_t { return 1; }
constexpr auto size(const std::floating_point auto) -> ptrdiff_t { return 1; }
constexpr auto size(const AbstractVector auto &x) -> ptrdiff_t {
  return x.size();
}

template <typename T>
concept HasConcreteSize = requires(T) {
  std::is_same_v<typename std::remove_reference_t<T>::concrete, std::true_type>;
};

static_assert(!HasConcreteSize<int64_t>);
template <typename T, typename U>
using is_concrete_t =
  std::conditional_t<HasConcreteSize<T> || HasConcreteSize<U>, std::true_type,
                     std::false_type>;

template <utils::TriviallyCopyable A, TrivialCompatible<A> B,
          BinaryFuncOfElts<A, B> Op>
struct ElementwiseBinaryOp;

template <AbstractTensor A, AbstractTensor B> struct MatMatMul;

template <typename A, typename B>
using argtyp_t =
  std::conditional_t<AbstractTensor<B> &&
                       std::convertible_to<A, utils::eltype_t<B>> &&
                       (std::integral<utils::eltype_t<B>> ||
                        std::floating_point<utils::eltype_t<B>>),
                     utils::eltype_t<B>, A>;

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif
template <utils::TriviallyCopyable A, FuncOfElt<A> Op>
constexpr auto elementwise(A, Op) -> Elementwise<A, Op>;

template <utils::TriviallyCopyable A, utils::TriviallyCopyable B,
          BinaryFuncOfElts<A, B> Op>
constexpr auto elementwise(A, B, Op)
  -> ElementwiseBinaryOp<argtyp_t<A, B>, argtyp_t<B, A>, Op>;
}; // namespace math

template <TrivialTensor C, utils::TriviallyCopyable A,
          utils::TriviallyCopyable B>
struct AbstractSelect {
  using value_type = std::common_type_t<utils::eltype_t<A>, utils::eltype_t<B>>;
  static constexpr bool isvector = AbstractVector<C>;
  [[no_unique_address]] C c;
  [[no_unique_address]] A a;
  [[no_unique_address]] B b;

  [[nodiscard]] constexpr auto numRow() const
  requires(!isvector)
  {
    auto n = unwrapRow(c.numRow());
    if constexpr (AbstractMatrix<A>) {
      auto nn = check_sizes(n, unwrapRow(a.numRow()));
      if constexpr (AbstractMatrix<B>)
        return row(check_sizes(nn, unwrapRow(b.numRow())));
      else return row(nn);
    } else if (AbstractMatrix<B>)
      return row(check_sizes(n, unwrapRow(b.numRow())));
    else return row(n);
  }
  [[nodiscard]] constexpr auto numCol() const
  requires(!isvector)
  {
    auto n = unwrapCol(c.numCol());
    if constexpr (AbstractMatrix<A>) {
      auto nn = check_sizes(n, unwrapCol(a.numCol()));
      if constexpr (AbstractMatrix<B>)
        return col(check_sizes(nn, unwrapCol(b.numCol())));
      else return col(nn);
    } else if (AbstractMatrix<B>)
      return col(check_sizes(n, unwrapCol(b.numCol())));
    else return col(n);
  }
  [[nodiscard]] constexpr auto size() const {
    if constexpr (!isvector) {
      return unwrapRow(numRow()) * unwrapCol(numCol());
    } else {
      auto N = c.size();
      if constexpr (AbstractVector<A>) {
        auto M = check_sizes(N, a.size());
        if constexpr (AbstractVector<B>) return check_sizes(M, b.size());
        else return M;
      } else if constexpr (AbstractVector<B>) {
        return check_sizes(N, b.size());
      } else {
        return N;
      }
    }
  }
};

// constexpr auto bin2(std::integral auto x) { return (x * (x - 1)) >> 1; }

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif

template <typename T, typename A> class Expr {
  constexpr auto v() const { return static_cast<const A *>(this)->view(); }
  static constexpr bool primitive_elt =
    std::integral<T> || std::floating_point<T>;

  friend constexpr auto operator+(std::convertible_to<T> auto b, const A &a) {
    return elementwise(b, a.view(), std::plus<>{});
  }
  friend constexpr auto operator-(std::convertible_to<T> auto b, const A &a) {
    return elementwise(b, a.view(), std::minus<>{});
  }
  friend constexpr auto operator/(std::convertible_to<T> auto b, const A &a) {
    return elementwise(b, a.view(), std::divides<>{});
  }
  friend constexpr auto operator%(std::convertible_to<T> auto b, const A &a) {
    return elementwise(b, a.view(), std::modulus<>{});
  }
  friend constexpr auto operator&(std::convertible_to<T> auto b, const A &a) {
    return elementwise(b, a.view(), std::bit_and<>{});
  }
  friend constexpr auto operator|(std::convertible_to<T> auto b, const A &a) {
    return elementwise(b, a.view(), std::bit_or<>{});
  }
  friend constexpr auto operator^(std::convertible_to<T> auto b, const A &a) {
    return elementwise(b, a.view(), std::bit_xor<>{});
  }
  friend constexpr auto operator*(std::convertible_to<T> auto b, const A &a) {
    return elementwise(b, a.view(), std::multiplies<>{});
  }

public:
  constexpr auto operator-() const { return elementwise(v(), std::negate<>{}); }
  constexpr auto operator!() const {
    return elementwise(v(), std::logical_not<>{});
  }
  constexpr auto operator~() const {
    return elementwise(v(), std::bit_not<>{});
  }

  template <Compatible<A> B> constexpr auto operator+(const B &b) const {
    return elementwise(v(), view(b), std::plus<>{});
  }
  template <Compatible<A> B> constexpr auto operator-(const B &b) const {
    return elementwise(v(), view(b), std::minus<>{});
  }
  template <Compatible<A> B> constexpr auto operator/(const B &b) const {
    return elementwise(v(), view(b), std::divides<>{});
  }
  template <Compatible<A> B> constexpr auto operator%(const B &b) const {
    return elementwise(v(), view(b), std::modulus<>{});
  }
  template <Compatible<A> B> constexpr auto operator&(const B &b) const {
    return elementwise(v(), view(b), std::bit_and<>{});
  }
  template <Compatible<A> B> constexpr auto operator|(const B &b) const {
    return elementwise(v(), view(b), std::bit_or<>{});
  }
  template <Compatible<A> B> constexpr auto operator^(const B &b) const {
    return elementwise(v(), view(b), std::bit_xor<>{});
  }
  template <typename B>
  constexpr auto operator*(const B &b) const
  requires(std::convertible_to<std::remove_cvref_t<B>, T> || AbstractTensor<B>)
  {
    if constexpr (!std::convertible_to<std::remove_cvref_t<B>, T>) {
      auto AA{v()};
      auto BB{b.view()};
      invariant(ptrdiff_t(numCols(AA)), ptrdiff_t(numRows(BB)));
      if constexpr (RowVector<decltype(AA)> && ColVector<decltype(BB)>)
        return dot(AA, BB.t());
      else if constexpr (AbstractVector<decltype(AA)> &&
                         AbstractVector<decltype(BB)>)
        return elementwise(AA, BB, std::multiplies<>{});
      else return MatMatMul<decltype(AA), decltype(BB)>{.a_ = AA, .b_ = BB};
    } else return elementwise(v(), b, std::multiplies<>{});
  }

  [[gnu::flatten]] constexpr auto
  operator==(const AbstractTensor auto &B) const -> bool {
    auto [Ma, Na] = shape(v());
    auto [Mb, Nb] = shape(B);

    if ((Ma != Mb) || (Na != Nb)) return false;
    using CT = std::common_type_t<T, utils::eltype_t<decltype(B)>>;
    auto M = check_sizes(Ma, Mb);
    auto N = check_sizes(Na, Nb);
    auto a{v()};
    auto b{view(B)};
    if constexpr (simd::Width<CT> <= 2) {
      if constexpr (AbstractMatrix<A>) {
        for (ptrdiff_t r = 0; r < M; ++r)
          for (ptrdiff_t i = 0; i < N; ++i)
            if (a[r, i] != b[r, i]) return false;
      } else {
        ptrdiff_t L = RowVector<A> ? N : M;
        for (ptrdiff_t i = 0; i < L; ++i)
          if (a[i] != b[i]) return false;
      }
    } else if constexpr (AbstractMatrix<A>) {
      constexpr ptrdiff_t W = getWidth<CT, decltype(N)>();
      for (ptrdiff_t r = 0; r < M; ++r) {
        for (ptrdiff_t i = 0;; i += W) {
          auto u{simd::index::unrollmask<1, W>(N, i)};
          if (!u) break;
          if (simd::cmp::ne<W, CT>(a[r, u], b[r, u])) return false;
        }
      }
    } else {
      constexpr ptrdiff_t W = RowVector<A> ? getWidth<CT, decltype(N)>()
                                           : getWidth<CT, decltype(M)>();
      ptrdiff_t L = RowVector<A> ? N : M;
      for (ptrdiff_t i = 0;; i += W) {
        auto u{simd::index::unrollmask<1, W>(L, i)};
        if (!u) break;
        if (simd::cmp::ne<W, CT>(a[u], b[u])) return false;
      }
    }
    return true;
  }
};
template <typename T, typename A>
struct Transpose : public Expr<T, Transpose<T, A>> {
  static_assert(AbstractTensor<A>,
                "Argument to transpose is not a matrix or vector.");
  static_assert(std::is_trivially_copyable_v<A>,
                "Argument to transpose is not trivially copyable.");

  using value_type = T;
  static constexpr bool has_reduction_loop = HasInnerReduction<A>;
  [[no_unique_address]] A a_;
  constexpr auto operator[](auto i) const
  requires(AbstractVector<A>)
  {
    return a_[i];
  }
  constexpr auto operator[](auto i, auto j) const
  requires(AbstractMatrix<A>)
  {
    return a_[j, i];
  }
  [[nodiscard]] constexpr auto numRow() const {
    return transpose_dim(a_.numCol());
  }
  [[nodiscard]] constexpr auto numCol() const {
    return transpose_dim(a_.numRow());
  }
  [[nodiscard]] constexpr auto view() const {
    return Transpose<T, std::remove_cvref_t<decltype(a_.view())>>(a_.view());
  };
  [[nodiscard]] constexpr auto size() const { return a_.size(); }
  [[nodiscard]] constexpr auto dim() const {
    return DenseDims(numRow(), numCol());
  }
  constexpr Transpose(A b) : a_(b) {}
  constexpr auto t() const -> A { return a_; }
  constexpr auto operator<<(const auto &b) -> Transpose & {
    a_ << transpose(b);
    return *this;
  }
  constexpr auto operator+=(const auto &b) -> Transpose & {
    a_ += transpose(b);
    return *this;
  }
  constexpr auto operator-=(const auto &b) -> Transpose & {
    a_ -= transpose(b);
    return *this;
  }
  constexpr auto operator*=(const auto &b) -> Transpose & {
    a_ *= transpose(b);
    return *this;
  }
  constexpr auto operator/=(const auto &b) -> Transpose & {
    a_ /= transpose(b);
    return *this;
  }
};
template <typename A> Transpose(A) -> Transpose<utils::eltype_t<A>, A>;

} // namespace math

template <utils::TriviallyCopyable A, FuncOfElt<A> Op>
struct Elementwise : public Expr<decltype(std::declval<Op>()(
                                   std::declval<utils::eltype_t<A>>())),
                                 Elementwise<A, Op>> {
  using value_type =
    decltype(std::declval<Op>()(std::declval<utils::eltype_t<A>>()));
  static constexpr bool has_reduction_loop = HasInnerReduction<A>;
  [[no_unique_address]] A a;
  [[no_unique_address]] Op op;
  constexpr auto operator[](auto i) const
  requires(LinearlyIndexableOrConvertible<A, value_type>)
  {
    return op(a[i]);
  }
  constexpr auto operator[](auto i, auto j) const
  requires(CartesianIndexableOrConvertible<A, value_type>)
  {
    return op(a[i, j]);
  }

  [[nodiscard]] constexpr auto size() const
  requires(DefinesSize<A>)
  {
    return a.size();
  }
  [[nodiscard]] constexpr auto numRow() const
  requires(DefinesShape<A>)
  {
    return a.numRow();
  }
  [[nodiscard]] constexpr auto numCol() const
  requires(DefinesShape<A>)
  {
    return a.numCol();
  }
  [[nodiscard]] constexpr auto view() const { return *this; };
  template <typename T> constexpr auto reinterpretImpl() {
    auto ra = reinterpret<T>(a);
    return Elementwise<decltype(ra), Op>(ra, op);
  }
};
template <utils::TriviallyCopyable A, TrivialCompatible<A> B,
          BinaryFuncOfElts<A, B> Op>
struct ElementwiseBinaryOp
  : public math::Expr<decltype(std::declval<Op>()(
                        std::declval<indextype_t<A, B>>(),
                        std::declval<indextype_t<B, A>>())),
                      ElementwiseBinaryOp<A, B, Op>> {
  using elta = indextype_t<A, B>;
  using eltb = indextype_t<B, A>;

  static constexpr bool has_reduction_loop =
    HasInnerReduction<A> || HasInnerReduction<B>;
  using value_type =
    decltype(std::declval<Op>()(std::declval<elta>(), std::declval<eltb>()));
  using concrete = is_concrete_t<A, B>;
  static constexpr bool isvector = AbstractVector<A> || AbstractVector<B>;
  static constexpr bool ismatrix = AbstractMatrix<A> || AbstractMatrix<B>;
  static_assert(isvector != ismatrix);

  [[no_unique_address]] A a;
  [[no_unique_address]] B b;
  [[no_unique_address]] Op op;
  constexpr auto operator[](auto i) const
  requires LinearlyIndexableOrConvertible<A, elta> &&
           LinearlyIndexableOrConvertible<B, eltb>
  {
    if constexpr (LinearlyIndexable<A, elta>)
      if constexpr (LinearlyIndexable<B, eltb>) return op(a[i], b[i]);
      else return op(a[i], b);
    else if constexpr (LinearlyIndexable<B, eltb>) return op(a, b[i]);
    else return op(a, b);
  }
  constexpr auto operator[](auto i, auto j) const {
    if constexpr (CartesianIndexable<A, elta>)
      if constexpr (CartesianIndexable<B, eltb>) return op(a[i, j], b[i, j]);
      else if constexpr (std::convertible_to<B, eltb>) return op(a[i, j], b);
      else if constexpr (RowVector<B>) return op(a[i, j], b[j]);
      else return op(a[i, j], b[i]);
    else if constexpr (std::convertible_to<A, elta>)
      if constexpr (CartesianIndexable<B, eltb>) return op(a, b[i, j]);
      else if constexpr (std::convertible_to<B, eltb>) return op(a, b);
      else if constexpr (RowVector<B>) return op(a, b[j]);
      else return op(a, b[i]);
    else if constexpr (RowVector<A>)
      if constexpr (CartesianIndexable<B, eltb>) return op(a[j], b[i, j]);
      else if constexpr (std::convertible_to<B, eltb>) return op(a[j], b);
      else if constexpr (RowVector<B>) return op(a[j], b[j]);
      else return op(a[j], b[i]);
    else if constexpr (CartesianIndexable<B, eltb>) return op(a[i], b[i, j]);
    else if constexpr (std::convertible_to<B, eltb>) return op(a[i], b);
    else if constexpr (RowVector<B>) return op(a[i], b[j]);
    else return op(a[i], b[i]);
  }

  [[nodiscard]] constexpr auto numRow() const {
    return row(check_sizes(unwrapRow(numRows(a)), unwrapRow(numRows(b))));
  }
  [[nodiscard]] constexpr auto numCol() const {
    return col(check_sizes(unwrapCol(numCols(a)), unwrapCol(numCols(b))));
  }
  [[nodiscard]] constexpr auto size() const {
    return unwrapRow(numRow()) * unwrapCol(numCol());
  }
  [[nodiscard]] constexpr auto view() const { return *this; };
  template <typename T> constexpr auto reinterpretImpl() {
    auto ra = reinterpret<T>(a);
    auto rb = reinterpret<T>(b);
    return elementwise(ra, rb, op);
  }
};

template <utils::TriviallyCopyable A, FuncOfElt<A> Op>
Elementwise(A, Op) -> Elementwise<A, Op>;

// // promote primitive element types, e.g. so
// // operator+(int, AbstractVector<int64_t>)
// // turns into
// // operator+(int64_t, AbstractVector<int64_t>)
template <utils::TriviallyCopyable A, utils::TriviallyCopyable B,
          BinaryFuncOfElts<A, B> Op>
ElementwiseBinaryOp(A, B, Op)
  -> ElementwiseBinaryOp<argtyp_t<A, B>, argtyp_t<B, A>, Op>;

template <TrivialTensor C, utils::TriviallyCopyable A,
          utils::TriviallyCopyable B>
struct Select : public AbstractSelect<C, A, B>,
                public math::Expr<typename AbstractSelect<C, A, B>::value_type,
                                  Select<C, A, B>> {
  using value_type = AbstractSelect<C, A, B>::value_type;
  static constexpr bool has_reduction_loop =
    HasInnerReduction<A> || HasInnerReduction<B>;
  constexpr auto operator[](auto i) const
  requires LinearlyIndexable<C, bool> &&
           LinearlyIndexableOrConvertible<A, value_type> &&
           LinearlyIndexableOrConvertible<B, value_type>
  {
    if constexpr (LinearlyIndexable<A, value_type>)
      if constexpr (LinearlyIndexable<B, value_type>)
        return this->c[i] ? this->a[i] : this->b[i];
      else return this->c[i] ? this->a[i] : this->b;
    else if constexpr (LinearlyIndexable<B, value_type>)
      return this->c[i] ? this->a : this->b[i];
    else return this->c[i] ? this->a : this->b;
  }
  constexpr auto operator[](auto i, auto j) const
  requires CartesianIndexableOrConvertible<C, bool> &&
           CartesianIndexableOrConvertible<A, value_type> &&
           CartesianIndexableOrConvertible<B, value_type>
  {
    if constexpr (CartesianIndexable<A, value_type>)
      if constexpr (CartesianIndexable<B, value_type>)
        return this->c[i, j] ? this->a[i, j] : this->b[i, j];
      else return this->c[i, j] ? this->a[i, j] : this->b;
    else if constexpr (CartesianIndexable<B, value_type>)
      return this->c[i, j] ? this->a : this->b[i, j];
    else return this->c[i, j] ? this->a : this->b;
  }
  [[nodiscard]] constexpr auto view() const -> Select { return *this; };
};
template <TrivialTensor C, utils::TriviallyCopyable A,
          utils::TriviallyCopyable B>
Select(C c, A a, B b) -> Select<C, A, B>;

template <TrivialTensor C, utils::TriviallyCopyable A,
          utils::TriviallyCopyable B, BinaryFuncOfElts<A, B> Op>
struct Conditional
  : public AbstractSelect<C, A, B>,
    public math::Expr<typename AbstractSelect<C, A, B>::value_type,
                      Conditional<C, A, B, Op>> {
  using value_type = AbstractSelect<C, A, B>::value_type;
  static constexpr bool has_reduction_loop =
    HasInnerReduction<A> || HasInnerReduction<B>;
  [[no_unique_address]] Op op;

  constexpr auto operator[](ptrdiff_t i) const -> value_type
  requires LinearlyIndexableOrConvertible<C, bool> &&
           LinearlyIndexableOrConvertible<A, value_type> &&
           LinearlyIndexableOrConvertible<B, value_type>
  {
    if constexpr (LinearlyIndexable<A, value_type>)
      if constexpr (LinearlyIndexable<B, value_type>)
        return this->c[i] ? op(this->a[i], this->b[i]) : this->a[i];
      else return this->c[i] ? op(this->a[i], this->b) : this->a[i];
    else if constexpr (LinearlyIndexable<B, value_type>)
      return this->c[i] ? op(this->a, this->b[i]) : this->a;
    else return this->c[i] ? op(this->a, this->b) : this->a;
  }
  template <ptrdiff_t U, ptrdiff_t W, typename M>
  constexpr auto operator[](simd::index::Unroll<U, W, M> i) const
  requires LinearlyIndexableOrConvertible<C, bool> &&
           LinearlyIndexableOrConvertible<A, value_type> &&
           LinearlyIndexableOrConvertible<B, value_type>
  {
    if constexpr (W == 1) {
      auto c = this->c[i];
      simd::Unroll<U, 1, 1, value_type> x = get<value_type>(this->a, i),
                                        y = op(x, get<value_type>(this->b, i));
      POLYMATHFULLUNROLL
      for (ptrdiff_t u = 0; u < U; ++u)
        x.data[u] = c.data[u] ? y.data[u] : x.data[u];
      return x;
    } else if constexpr (LinearlyIndexable<A, value_type>) {
      auto c = get<bool>(this->c, i);
      simd::Unroll<1, U, W, value_type> x = get<value_type>(this->a, i),
                                        y = op(x, get<value_type>(this->b, i));
      POLYMATHFULLUNROLL
      for (ptrdiff_t u = 0; u < U; ++u)
        x.data[u] = c.data[u] ? y.data[u] : x.data[u];
      return x;
    } else if constexpr (LinearlyIndexable<B, value_type>) {
      auto c = this->c[i];
      using V = simd::Vec<W, value_type>;
      value_type x_ = get<value_type>(this->a, i);
      V x = simd::vbroadcast<W, value_type>(x_);
      simd::Unroll<1, U, W, value_type> y = op(x, get<value_type>(this->b, i));
      POLYMATHFULLUNROLL
      for (ptrdiff_t u = 0; u < U; ++u)
        if constexpr (LinearlyIndexable<B, value_type>)
          y.data[u] = c.data[u] ? y.data[u] : x;
      return y;
    } else {
      auto c = this->c[i];
      using V = simd::Vec<W, value_type>;
      value_type x_ = get<value_type>(this->a, i);
      value_type y_ = op(x_, get<value_type>(this->b, i));
      V x = simd::vbroadcast<W, value_type>(x_),
        y = simd::vbroadcast<W, value_type>(y_);
      simd::Unroll<1, U, W, value_type> z;
      POLYMATHFULLUNROLL
      for (ptrdiff_t u = 0; u < U; ++u) z.data[u] = c.data[u] ? y : x;
      return z;
    }
    //   auto c = get<bool>(this->c, i);
    //   auto x = get<value_type>(this->a, i);
    //   auto y = op(x, get<value_type>(this->b, i));
    //   simd::Unroll<1, U, W, value_type> z;
    //   POLYMATHFULLUNROLL
    //   for (ptrdiff_t u = 0; u < U; ++u)
    //     if constexpr (LinearlyIndexable<A, value_type>)
    //       z.data[u] = !c.data[u] ? x.data[u] : y.data[u];
    //     else if constexpr (LinearlyIndexable<B, value_type>)
    //       z.data[u] = !c.data[u] ? x : y.data[u];
    //     else z.data[u] = c.data[u] ? y : x;
    //   return z;
    // }
  }
  constexpr auto operator[](auto i, auto j) const
  requires CartesianIndexableOrConvertible<C, bool> &&
           CartesianIndexableOrConvertible<A, value_type> &&
           CartesianIndexableOrConvertible<B, value_type>
  {
    if constexpr (LinearlyIndexable<A, value_type>)
      if constexpr (LinearlyIndexable<B, value_type>)
        return this->c[i, j] ? op(this->a[i, j], this->b[i, j]) : this->a[i, j];
      else return this->c[i, j] ? op(this->a[i, j], this->b) : this->a[i, j];
    else if constexpr (LinearlyIndexable<B, value_type>)
      return this->c[i, j] ? op(this->a, this->b[i, j]) : this->a;
    else return this->c[i, j] ? op(this->a, this->b) : this->a;
  }
  [[nodiscard]] constexpr auto view() const -> Conditional { return *this; };
};
template <AbstractTensor A, AbstractTensor B>
struct MatMatMul : public math::Expr<
                     std::common_type_t<utils::eltype_t<A>, utils::eltype_t<B>>,
                     MatMatMul<A, B>> {
  using value_type = std::common_type_t<utils::eltype_t<A>, utils::eltype_t<B>>;
  using concrete = is_concrete_t<A, B>;
  static constexpr bool has_reduction_loop = true;
  static constexpr bool ismata = AbstractMatrix<A>;
  static constexpr bool ismatb = AbstractMatrix<B>;
  static_assert((ismata && (ismatb || ColVector<B>)) ||
                (RowVector<A> && ismatb) || (ColVector<A> && RowVector<B>));
  static constexpr bool ismatrix = ismata && ismatb;
  // we could have
  // ColVector * RowVector
  // RowVector * Matrix
  // Matrix * ColVector
  // Matrix * Matrix
  [[no_unique_address]] A a_;
  [[no_unique_address]] B b_;
  [[gnu::always_inline]] constexpr auto operator[](auto i, auto j) const
  requires(ismatrix)
  {
    static_assert(AbstractMatrix<B>, "B should be an AbstractMatrix");
    invariant(ptrdiff_t(a_.numCol()) > 0);
    decltype(a_[i, 0] * b_[0, j] + a_[i, 1] * b_[1, j]) s{};
    POLYMATHNOVECTORIZE
    for (ptrdiff_t k = 0; k < ptrdiff_t(a_.numCol()); ++k)
      s += a_[i, k] * b_[k, j];
    return s;
  }
  // If `T isa Dual<Dual<double,7>,2>`, we would not want to construct
  // intermediates that require masked loads/stores, as compilers have
  // trouble optimizing these away.
  // We can imagine two strategies for avoiding this:
  // 1. Do not write/construct any intermediates, but write into the result
  // directly.
  // 2. Decompress/compress on load/store, so temporaries do not need masks.
  // `1.` seems ideal, but harder to implement.
  // For example, here, `s` would have to alias the result. Then we also have
  // `s +=`, which would have to invoke somehow call
  // `Dual<Dual<double,7>,2>::operator+=` without storing.
  // Or, we'd need a method that can see through it, so we operate on the
  // teminal values, but still in one go to only have one instance of the
  // reduction loop. Thus, `1.` seems conceptually simple but without a clear
  // implementation strategy.
  //
  //
  [[gnu::always_inline]] constexpr auto operator[](auto i) const
  requires(!ismatrix)
  {
    if constexpr (RowVector<A>) {
      invariant(a_.size() == b_.numRow());
      invariant(a_.size() > 0);
      decltype(a_[0] * b_[0, i] + a_[1] * b_[1, i]) s{};
      POLYMATHNOVECTORIZE
      for (ptrdiff_t k = 0; k < a_.numCol(); ++k) {
        POLYMATHFAST
        s += a_[k] * b_[k, i];
      }
      return s;
    } else { // ColVector<B>
      invariant(a_.numCol() == b_.size());
      invariant(b_.size() > 0);
      decltype(a_[i, 0] * b_[0] + a_[i, 1] * b_[1]) s{};
      for (ptrdiff_t k = 0; k < a_.numCol(); ++k) {
        POLYMATHFAST
        s += a_[i, k] * b_[k];
      }
      return s;
    }
  }
  [[nodiscard]] constexpr auto numRow() const {
    if constexpr (AbstractMatrix<A>) return a_.numRow();
    else return Row<1>{};
  }
  [[nodiscard]] constexpr auto numCol() const {
    if constexpr (AbstractMatrix<B>) return b_.numCol();
    else return Col<1>{};
  }
  [[nodiscard]] constexpr auto size() const {
    if constexpr (ismata)
      if constexpr (ismatb)
        return unwrapRow(a_.numRow()) * unwrapCol(b_.numCol());
      else return unwrapRow(a_.numRow());
    else if constexpr (RowVector<A>) return unwrapCol(b_.numCol());
    else a_.size() * b_.size();
  }
  [[nodiscard]] constexpr auto view() const { return *this; };
  [[nodiscard]] constexpr auto t() const -> Transpose<value_type, MatMatMul> {
    return {*this};
  };
};

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif

template <utils::TriviallyCopyable A, FuncOfElt<A> Op>
constexpr auto elementwise(A a, Op op) -> Elementwise<A, Op> {
  return {.a = a, .op = op};
}

template <utils::TriviallyCopyable A, utils::TriviallyCopyable B,
          BinaryFuncOfElts<A, B> Op>
constexpr auto elementwise(A a, B b, Op op)
  -> ElementwiseBinaryOp<argtyp_t<A, B>, argtyp_t<B, A>, Op> {
  return {.a = argtyp_t<A, B>(a), .b = argtyp_t<B, A>(b), .op = op};
}

constexpr auto select(const AbstractTensor auto &c, const auto &a,
                      const auto &b) {
  return Select(view(c), view(a), view(b));
}
constexpr auto conditional(auto op, const AbstractTensor auto &c, const auto &a,
                           const auto &b) {
  auto vc = view(c);
  auto va = view(a);
  auto vb = view(b);
  return Conditional<decltype(vc), decltype(va), decltype(vb), decltype(op)>{
    {vc, va, vb}, op};
}
constexpr auto elementwise_equal(const auto &a, const auto &b) {
  return elementwise(view(a), view(b), std::equal_to<>{});
}
constexpr auto elementwise_not_equal(const auto &a, const auto &b) {
  return elementwise(view(a), view(b), std::not_equal_to<>{});
}
constexpr auto elementwise_greater(const auto &a, const auto &b) {
  return elementwise(view(a), view(b), std::greater<>{});
}
constexpr auto elementwise_less(const auto &a, const auto &b) {
  return elementwise(view(a), view(b), std::less<>{});
}
constexpr auto elementwise_greater_equal(const auto &a, const auto &b) {
  return elementwise(view(a), view(b), std::greater_equal<>{});
}
constexpr auto elementwise_less_equal(const auto &a, const auto &b) {
  return elementwise(view(a), view(b), std::less_equal<>{});
}

template <typename T> constexpr auto transpose(const T &a) {
  if constexpr (requires(T t) {
                  { t.t() } -> AbstractTensor;
                })
    return a.t();
  else return Transpose{view(a)};
}

template <typename T> struct ScalarizeViaCast<Elementwise<std::negate<>, T>> {
  using type = scalarize_via_cast_t<T>;
};
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

} // namespace math
