#pragma once
// We'll follow Julia style, so anything that's not a constructor, destructor,
// nor an operator will be outside of the struct/class.

#include "Math/Array.hpp"
#include "Math/AxisTypes.hpp"
#include "Math/Indexing.hpp"
#include "Math/Matrix.hpp"
#include "Math/MatrixDimensions.hpp"
#include "Utilities/LoopMacros.hpp"
#include "Utilities/Parameters.hpp"
#include "Utilities/TypePromotion.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>
#include <type_traits>
#include <utility>

namespace poly::math {
/// Extract the value of a `Dual` number
constexpr auto value(std::floating_point auto x) { return x; }
struct Rational;

template <class T, class C>
concept RangeOffsetPair =
  (std::convertible_to<T, Range<ptrdiff_t, ptrdiff_t>> &&
   std::convertible_to<C, ptrdiff_t>) ||
  (std::convertible_to<C, Range<ptrdiff_t, ptrdiff_t>> &&
   std::convertible_to<T, ptrdiff_t>);

template <class T, class C>
concept Compatible =
  (AbstractTensor<C> && std::convertible_to<T, utils::eltype_t<C>>) ||
  (AbstractTensor<T> && std::convertible_to<C, utils::eltype_t<T>>) ||
  (AbstractVector<C> && AbstractVector<T>) ||
  (AbstractMatrix<C> && AbstractMatrix<T>);

template <class A, class B> struct ElTypes {
  using eltype =
    std::conditional_t<AbstractTensor<B> &
                         std::convertible_to<A, utils::eltype_t<B>>,
                       A, utils::eltype_t<A>>;
};
// returns the element type of `A` when used in a binary op with `B`
template <class A, class B> using indextype_t = typename ElTypes<A, B>::eltype;

template <typename T>
concept Trivial =
  std::is_trivially_destructible_v<T> && std::is_trivially_copyable_v<T>;
template <typename T, typename C>
concept TrivialCompatible = Trivial<T> && Compatible<T, C>;
template <typename T>
concept TrivialTensor = Trivial<T> && AbstractTensor<T>;
template <typename T>
concept TrivialVec = Trivial<T> && AbstractVector<T>;
template <typename T>
concept TrivialMat = Trivial<T> && AbstractMatrix<T>;
//   // TODO: binary func invocable trait?
// template <typename Op, typename T, typename S>
// concept BinaryFunction = std::is_invocable_v<Op, T, S>;

template <typename Op, typename A, typename B>
concept BinaryFuncOfElts =
  std::is_invocable_v<Op, indextype_t<A, B>, indextype_t<B, A>>;

template <typename T, typename I> consteval auto getWidth() -> ptrdiff_t {
  if constexpr (std::same_as<I, ptrdiff_t>) return simd::Width<T>;
  else return simd::VecLen<ptrdiff_t(I{}), T>;
}

// TODO: make this part of ArrayOps!!!
[[gnu::flatten]] constexpr auto operator==(const AbstractTensor auto &A,
                                           const AbstractTensor auto &B)
  -> bool {
  auto [Ma, Na] = shape(A);
  auto [Mb, Nb] = shape(B);

  if ((Ma != Mb) || (Na != Nb)) return false;
  using T = std::common_type_t<utils::eltype_t<decltype(A)>,
                               utils::eltype_t<decltype(B)>>;
  auto M = check_sizes(Ma, Mb);
  auto N = check_sizes(Na, Nb);
  if constexpr (simd::Width<T> <= 2) {
    if constexpr (AbstractMatrix<decltype(A)>) {
      for (ptrdiff_t r = 0; r < M; ++r)
        for (ptrdiff_t i = 0; i < N; ++i)
          if (A[r, i] != B[r, i]) return false;
    } else {
      ptrdiff_t L = RowVector<decltype(A)> ? N : M;
      for (ptrdiff_t i = 0; i < L; ++i)
        if (A[i] != B[i]) return false;
    }
  } else if constexpr (AbstractMatrix<decltype(A)>) {
    constexpr ptrdiff_t W = getWidth<T, decltype(N)>();
    for (ptrdiff_t r = 0; r < M; ++r) {
      for (ptrdiff_t i = 0;; i += W) {
        auto u{simd::index::unrollmask<1, W>(N, i)};
        if (!u) break;
        if (simd::cmp::ne<W, T>(A[r, u], B[r, u])) return false;
      }
    }
  } else {
    constexpr ptrdiff_t W = RowVector<decltype(A)> ? getWidth<T, decltype(N)>()
                                                   : getWidth<T, decltype(M)>();
    ptrdiff_t L = RowVector<decltype(A)> ? N : M;
    for (ptrdiff_t i = 0;; i += W) {
      auto u{simd::index::unrollmask<1, W>(L, i)};
      if (!u) break;
      if (simd::cmp::ne<W, T>(A[u], B[u])) return false;
    }
  }
  return true;
}

template <typename Op, typename A> struct Elementwise {
  using value_type =
    decltype(std::declval<Op>()(std::declval<utils::eltype_t<A>>()));
  static constexpr bool has_reduction_loop = HasInnerReduction<A>;
  [[no_unique_address]] Op op;
  [[no_unique_address]] A a;
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
  template <typename T> constexpr auto reinterpret() {
    auto ra = math::reinterpret<T>(a);
    return Elementwise<Op, decltype(ra)>(op, ra);
  }
};
template <typename Op, typename A> Elementwise(Op, A) -> Elementwise<Op, A>;

static_assert(
  ColVector<Elementwise<std::negate<>, Array<int64_t, StridedRange>>>);

constexpr auto size(const std::integral auto) -> ptrdiff_t { return 1; }
constexpr auto size(const std::floating_point auto) -> ptrdiff_t { return 1; }
constexpr auto size(const AbstractVector auto &x) -> ptrdiff_t {
  return x.size();
}

static_assert(utils::ElementOf<int, DenseMatrix<int64_t>>);
static_assert(utils::ElementOf<int64_t, DenseMatrix<int64_t>>);
static_assert(utils::ElementOf<int64_t, DenseMatrix<double>>);
static_assert(!utils::ElementOf<DenseMatrix<double>, DenseMatrix<double>>);

template <typename T>
concept HasConcreteSize = requires(T) {
  std::is_same_v<typename std::remove_reference_t<T>::concrete, std::true_type>;
};

static_assert(HasConcreteSize<DenseMatrix<int64_t>>);
static_assert(!HasConcreteSize<int64_t>);
static_assert(!HasConcreteSize<UniformScaling<std::true_type>>);
// template <typename T>
// using is_concrete_t =
//   std::conditional_t<HasConcreteSize<T>, std::true_type, std::false_type>;
template <typename T, typename U>
using is_concrete_t =
  std::conditional_t<HasConcreteSize<T> || HasConcreteSize<U>, std::true_type,
                     std::false_type>;

template <Trivial A, TrivialCompatible<A> B, BinaryFuncOfElts<A, B> Op>
struct ElementwiseBinaryOp {
  using elta = indextype_t<A, B>;
  using eltb = indextype_t<B, A>;

  static constexpr bool has_reduction_loop =
    HasInnerReduction<A> || HasInnerReduction<B>;
  using common = std::common_type_t<elta, eltb>;
  using value_type = decltype(std::declval<Op>()(std::declval<common>(),
                                                 std::declval<common>()));
  using concrete = is_concrete_t<A, B>;
  static constexpr bool isvector = AbstractVector<A> || AbstractVector<B>;
  static constexpr bool ismatrix = AbstractMatrix<A> || AbstractMatrix<B>;
  static_assert(isvector != ismatrix);

  [[no_unique_address]] Op op;
  [[no_unique_address]] A a;
  [[no_unique_address]] B b;
  constexpr auto operator[](auto i) const
  requires LinearlyIndexableOrConvertible<A, common> &&
           LinearlyIndexableOrConvertible<B, common>
  {
    if constexpr (LinearlyIndexable<A, common>)
      if constexpr (LinearlyIndexable<B, common>) return op(a[i], b[i]);
      else return op(a[i], b);
    else if constexpr (LinearlyIndexable<B, common>) return op(a, b[i]);
    else return op(a, b);
  }
  constexpr auto operator[](auto i, auto j) const {
    if constexpr (CartesianIndexable<A, common>)
      if constexpr (CartesianIndexable<B, common>) return op(a[i, j], b[i, j]);
      else if constexpr (std::convertible_to<B, common>) return op(a[i, j], b);
      else if constexpr (RowVector<B>) return op(a[i, j], b[j]);
      else return op(a[i, j], b[i]);
    else if constexpr (std::convertible_to<A, common>)
      if constexpr (CartesianIndexable<B, common>) return op(a, b[i, j]);
      else if constexpr (std::convertible_to<B, common>) return op(a, b);
      else if constexpr (RowVector<B>) return op(a, b[j]);
      else return op(a, b[i]);
    else if constexpr (RowVector<A>)
      if constexpr (CartesianIndexable<B, common>) return op(a[j], b[i, j]);
      else if constexpr (std::convertible_to<B, common>) return op(a[j], b);
      else if constexpr (RowVector<B>) return op(a[j], b[j]);
      else return op(a[j], b[i]);
    else if constexpr (CartesianIndexable<B, common>) return op(a[i], b[i, j]);
    else if constexpr (std::convertible_to<B, common>) return op(a[i], b);
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
  [[nodiscard]] constexpr auto view() const -> auto & { return *this; };
  template <typename T> constexpr auto reinterpret() {
    auto ra = math::reinterpret<T>(a);
    auto rb = math::reinterpret<T>(b);
    return ElementwiseBinaryOp<decltype(ra), decltype(rb), Op>(op, ra, rb);
  }
};
static_assert(AbstractMatrix<ElementwiseBinaryOp<Array<int64_t, DenseDims<>>,
                                                 int64_t, std::multiplies<>>>);
static_assert(ColVector<ElementwiseBinaryOp<Array<int64_t, StridedRange>,
                                            int64_t, std::multiplies<>>>);
template <TrivialTensor C, Trivial A, Trivial B> struct AbstractSelect {
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

inline constexpr auto view(const Trivial auto &x) { return x; }
inline constexpr auto view(const auto &x) { return x.view(); }
template <class T, class S> constexpr auto view(const Array<T, S> &x) {
  return x;
}

template <TrivialTensor C, Trivial A, Trivial B>
struct Select : public AbstractSelect<C, A, B> {
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
template <TrivialTensor C, Trivial A, Trivial B>
Select(C c, A a, B b) -> Select<C, A, B>;
constexpr auto select(const AbstractTensor auto &c, const auto &a,
                      const auto &b) {
  return Select(view(c), view(a), view(b));
}

template <TrivialTensor C, Trivial A, Trivial B, BinaryFuncOfElts<A, B> Op>
struct Conditional : public AbstractSelect<C, A, B> {
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
constexpr auto conditional(auto op, const AbstractTensor auto &c, const auto &a,
                           const auto &b) {
  auto vc = view(c);
  auto va = view(a);
  auto vb = view(b);
  return Conditional<decltype(vc), decltype(va), decltype(vb), decltype(op)>{
    {vc, va, vb}, op};
}

template <AbstractTensor A, AbstractTensor B> struct MatMatMul {
  using value_type = utils::promote_eltype_t<A, B>;
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
  [[no_unique_address]] A a;
  [[no_unique_address]] B b;
  [[gnu::always_inline]] constexpr auto operator[](auto i, auto j) const
  requires(ismatrix)
  {
    static_assert(AbstractMatrix<B>, "B should be an AbstractMatrix");
    invariant(ptrdiff_t(a.numCol()) > 0);
    decltype(a[i, 0] * b[0, j] + a[i, 1] * b[1, j]) s{};
    POLYMATHNOVECTORIZE
    for (ptrdiff_t k = 0; k < ptrdiff_t(a.numCol()); ++k)
      s += a[i, k] * b[k, j];
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
      invariant(a.size() == b.numRow());
      invariant(a.size() > 0);
      decltype(a[0] * b[0, i] + a[1] * b[1, i]) s{};
      POLYMATHNOVECTORIZE
      for (ptrdiff_t k = 0; k < a.numCol(); ++k) {
        POLYMATHFAST
        s += a[k] * b[k, i];
      }
      return s;
    } else { // ColVector<B>
      invariant(a.numCol() == b.size());
      invariant(b.size() > 0);
      decltype(a[i, 0] * b[0] + a[i, 1] * b[1]) s{};
      for (ptrdiff_t k = 0; k < a.numCol(); ++k) {
        POLYMATHFAST
        s += a[i, k] * b[k];
      }
      return s;
    }
  }
  [[nodiscard]] constexpr auto numRow() const {
    if constexpr (AbstractMatrix<A>) return a.numRow();
    else return Row<1>{};
  }
  [[nodiscard]] constexpr auto numCol() const {
    if constexpr (AbstractMatrix<B>) return b.numCol();
    else return Col<1>{};
  }
  [[nodiscard]] constexpr auto size() const {
    if constexpr (ismata)
      if constexpr (ismatb)
        return unwrapRow(a.numRow()) * unwrapCol(b.numCol());
      else return unwrapRow(a.numRow());
    else if constexpr (RowVector<A>) return unwrapCol(b.numCol());
    else a.size() * b.size();
  }
  [[nodiscard]] constexpr auto view() const { return *this; };
  [[nodiscard]] constexpr auto t() const { return Transpose{*this}; };
};

//
// Vectors
//

static_assert(!AbstractMatrix<StridedVector<int64_t>>);

// static_assert(std::is_trivially_copyable_v<MutStridedVector<int64_t>>);
static_assert(std::is_trivially_copyable_v<
              Elementwise<std::negate<>, StridedVector<int64_t>>>);
static_assert(Trivial<Elementwise<std::negate<>, StridedVector<int64_t>>>);

constexpr void swap(MutPtrMatrix<int64_t> A, Row<> i, Row<> j) {
  if (i == j) return;
  Col N = A.numCol();
  invariant((i < A.numRow()) && (j < A.numRow()));
  for (ptrdiff_t n = 0; n < N; ++n)
    std::swap(A[ptrdiff_t(i), n], A[ptrdiff_t(j), n]);
}
constexpr void swap(MutPtrMatrix<int64_t> A, Col<> i, Col<> j) {
  if (i == j) return;
  Row M = A.numRow();
  invariant((i < A.numCol()) && (j < A.numCol()));
  for (ptrdiff_t m = 0; m < M; ++m)
    std::swap(A[m, ptrdiff_t(i)], A[m, ptrdiff_t(j)]);
}

template <int Bits, class T>
constexpr bool is_uint_v =
  sizeof(T) == (Bits / 8) && std::is_integral_v<T> && !std::is_signed_v<T>;

template <class T>
constexpr auto zeroUpper(T x) -> T
requires is_uint_v<16, T>
{
  return x & 0x00ff;
}
template <class T>
constexpr auto zeroLower(T x) -> T
requires is_uint_v<16, T>
{
  return x & 0xff00;
}
template <class T>
constexpr auto upperHalf(T x) -> T
requires is_uint_v<16, T>
{
  return x >> 8;
}

template <class T>
constexpr auto zeroUpper(T x) -> T
requires is_uint_v<32, T>
{
  return x & 0x0000ffff;
}
template <class T>
constexpr auto zeroLower(T x) -> T
requires is_uint_v<32, T>
{
  return x & 0xffff0000;
}
template <class T>
constexpr auto upperHalf(T x) -> T
requires is_uint_v<32, T>
{
  return x >> 16;
}
template <class T>
constexpr auto zeroUpper(T x) -> T
requires is_uint_v<64, T>
{
  return x & 0x00000000ffffffff;
}
template <class T>
constexpr auto zeroLower(T x) -> T
requires is_uint_v<64, T>
{
  return x & 0xffffffff00000000;
}
template <class T>
constexpr auto upperHalf(T x) -> T
requires is_uint_v<64, T>
{
  return x >> 32;
}

static_assert(
  AbstractMatrix<MatMatMul<PtrMatrix<int64_t>, PtrMatrix<int64_t>>>);

static_assert(std::copy_constructible<PtrMatrix<int64_t>>);
// static_assert(std::is_trivially_copyable_v<MutPtrMatrix<int64_t>>);
static_assert(std::is_trivially_copyable_v<PtrMatrix<int64_t>>);
static_assert(Trivial<PtrMatrix<int64_t>>);
static_assert(Trivial<int>);
static_assert(TriviallyCopyable<std::multiplies<>>);
static_assert(
  Trivial<ElementwiseBinaryOp<PtrMatrix<int64_t>, int, std::multiplies<>>>);
static_assert(Trivial<MatMatMul<PtrMatrix<int64_t>, PtrMatrix<int64_t>>>);

template <TriviallyCopyable OP, Trivial A, Trivial B>
ElementwiseBinaryOp(OP, A, B) -> ElementwiseBinaryOp<A, B, OP>;

constexpr auto bin2(std::integral auto x) { return (x * (x - 1)) >> 1; }

template <typename T>
inline auto operator<<(std::ostream &os, SmallSparseMatrix<T> const &A)
  -> std::ostream & {
  ptrdiff_t k = 0;
  os << "[ ";
  for (ptrdiff_t i = 0; i < A.numRow(); ++i) {
    if (i) os << "  ";
    uint32_t m = A.rows[i] & 0x00ffffff;
    ptrdiff_t j = 0;
    while (m) {
      if (j) os << " ";
      uint32_t tz = std::countr_zero(m);
      m >>= (tz + 1);
      j += (tz + 1);
      while (tz--) os << " 0 ";
      const T &x = A.nonZeros[k++];
      if (x >= 0) os << " ";
      os << x;
    }
    for (; j < A.numCol(); ++j) os << "  0";
    os << "\n";
  }
  os << " ]";
  invariant(k == A.nonZeros.size());
  return os;
}
template <AbstractMatrix T>
inline auto operator<<(std::ostream &os, const T &A) -> std::ostream & {
  Matrix<std::remove_const_t<typename T::value_type>> B{A};
  return printMatrix(os, PtrMatrix<typename T::value_type>(B));
}

constexpr auto operator-(const AbstractVector auto &a) {
  auto AA{a.view()};
  return Elementwise<std::negate<>, decltype(AA)>{.op = std::negate<>{},
                                                  .a = AA};
}
constexpr auto operator-(const AbstractMatrix auto &a) {
  auto AA{a.view()};
  return Elementwise<std::negate<>, decltype(AA)>{.op = std::negate<>{},
                                                  .a = AA};
}

constexpr auto operator!(const AbstractVector auto &a) {
  auto AA{a.view()};
  return Elementwise<std::logical_not<>, decltype(AA)>{.op = std::negate<>{},
                                                       .a = AA};
}
constexpr auto operator!(const AbstractMatrix auto &a) {
  auto AA{a.view()};
  return Elementwise<std::logical_not<>, decltype(AA)>{.op = std::negate<>{},
                                                       .a = AA};
}

constexpr auto operator~(const AbstractVector auto &a) {
  auto AA{a.view()};
  return Elementwise<std::bit_not<>, decltype(AA)>{.op = std::negate<>{},
                                                   .a = AA};
}
constexpr auto operator~(const AbstractMatrix auto &a) {
  auto AA{a.view()};
  return Elementwise<std::bit_not<>, decltype(AA)>{.op = std::negate<>{},
                                                   .a = AA};
}
static_assert(AbstractMatrix<Elementwise<std::negate<>, PtrMatrix<int64_t>>>);
static_assert(AbstractMatrix<Array<int64_t, SquareDims<>>>);
static_assert(AbstractMatrix<ManagedArray<int64_t, SquareDims<>>>);

constexpr auto abs2(auto x) { return x * x; }
template <AbstractTensor B> constexpr auto norm2(const B &A) {
  utils::eltype_t<B> s = 0;
  if constexpr (!LinearlyIndexable<B, utils::eltype_t<B>>) {
    for (ptrdiff_t i = 0; i < A.numRow(); ++i) {
      for (ptrdiff_t j = 0; j < A.numCol(); ++j) {
        POLYMATHFAST
        s += abs2(A[i, j]);
      }
    }
  } else
    for (ptrdiff_t j = 0, L = ptrdiff_t(A.size()); j < L; ++j) {
      POLYMATHFAST
      s += abs2(A[j]);
    }
  return s;
}

constexpr auto norm2(const auto &a) {
  decltype(a[0] * a[0] + a[1] * a[1]) s{};
  for (auto x : a) {
    POLYMATHFAST
    s += abs2(x, x);
  }
  return s;
}
constexpr auto dot(const auto &a, const auto &b) {
  ptrdiff_t L = a.size();
  invariant(L, b.size());
  decltype(a[0] * b[0] + a[1] * b[1]) s{};
  for (ptrdiff_t i = 0; i < L; ++i) {
    POLYMATHFAST
    s += a[i] * b[i];
  }
  return s;
}

// we can have RowVector*RowVector or ColVector*ColVector(elementwise)
// RowVector*ColVector (dot product)
// Matrix*ColVector
// RowVector*Matrix
// Matrix*Matrix
// We can't have vector*matrix
constexpr auto operator*(const AbstractTensor auto &a,
                         const AbstractTensor auto &b) {
  auto AA{a.view()};
  auto BB{b.view()};
  invariant(ptrdiff_t(numCols(AA)), ptrdiff_t(numRows(BB)));
  if constexpr (RowVector<decltype(AA)> && ColVector<decltype(BB)>)
    return dot(AA, BB.t());
  else if constexpr (AbstractVector<decltype(AA)> &&
                     AbstractVector<decltype(BB)>)
    return ElementwiseBinaryOp(std::multiplies<>{}, AA, BB);
  else return MatMatMul<decltype(AA), decltype(BB)>{.a = AA, .b = BB};
}
template <AbstractVector M, utils::ElementOf<M> S>
constexpr auto operator*(const M &b, S a) {
  using T = utils::eltype_t<M>;
  if constexpr (sizeof(T) <= 8)
    return ElementwiseBinaryOp(std::multiplies<>{}, view(b),
                               utils::eltype_t<M>(a));
  else return ElementwiseBinaryOp(std::multiplies<>{}, view(b), a);
}
template <AbstractVector M, utils::ElementOf<M> S>
constexpr auto operator*(S a, const M &b) {
  using T = utils::eltype_t<M>;
  if constexpr (sizeof(T) <= 8)
    return ElementwiseBinaryOp(std::multiplies<>{}, utils::eltype_t<M>(a),
                               view(b));
  else return ElementwiseBinaryOp(std::multiplies<>{}, a, view(b));
}
template <AbstractMatrix M, utils::ElementOf<M> S>
constexpr auto operator*(S a, const M &b) {
  using T = utils::eltype_t<M>;
  if constexpr (sizeof(T) <= 8)
    return ElementwiseBinaryOp(std::multiplies<>{}, utils::eltype_t<M>(a),
                               view(b));
  else return ElementwiseBinaryOp(std::multiplies<>{}, a, view(b));
}
template <AbstractMatrix M, utils::ElementOf<M> S>
constexpr auto operator*(const M &b, S a) {
  using T = utils::eltype_t<M>;
  if constexpr (sizeof(T) <= 8)
    return ElementwiseBinaryOp(std::multiplies<>{}, view(b),
                               utils::eltype_t<M>(a));
  else return ElementwiseBinaryOp(std::multiplies<>{}, view(b), a);
}

template <class A, Compatible<A> B>
constexpr auto operator+(const A &a, const B &b)
requires(!RangeOffsetPair<A, B>)
{
  return ElementwiseBinaryOp(std::plus<>{}, view(a), view(b));
}
template <class A, Compatible<A> B>
constexpr auto operator-(const A &a, const B &b)
requires(!RangeOffsetPair<A, B>)
{
  return ElementwiseBinaryOp(std::minus<>{}, view(a), view(b));
}
template <class A, Compatible<A> B>
constexpr auto operator/(const A &a, const B &b) {
  return ElementwiseBinaryOp(std::divides<>{}, view(a), view(b));
}
template <class A, Compatible<A> B>
constexpr auto operator%(const A &a, const B &b) {
  return ElementwiseBinaryOp(std::modulus<>{}, view(a), view(b));
}
template <class A, Compatible<A> B>
constexpr auto operator&(const A &a, const B &b) {
  return ElementwiseBinaryOp(std::bit_and<>{}, view(a), view(b));
}
template <class A, Compatible<A> B>
constexpr auto operator|(const A &a, const B &b) {
  return ElementwiseBinaryOp(std::bit_or<>{}, view(a), view(b));
}
template <class A, Compatible<A> B>
constexpr auto operator^(const A &a, const B &b) {
  return ElementwiseBinaryOp(std::bit_xor<>{}, view(a), view(b));
}

template <class A, Compatible<A> B>
constexpr auto elementwise_equal(const A &a, const B &b) {
  return ElementwiseBinaryOp(std::equal_to<>{}, view(a), view(b));
}
template <class A, Compatible<A> B>
constexpr auto elementwise_not_equal(const A &a, const B &b) {
  return ElementwiseBinaryOp(std::not_equal_to<>{}, view(a), view(b));
}
template <class A, Compatible<A> B>
constexpr auto elementwise_greater(const A &a, const B &b) {
  return ElementwiseBinaryOp(std::greater<>{}, view(a), view(b));
}
template <class A, Compatible<A> B>
constexpr auto elementwise_less(const A &a, const B &b) {
  return ElementwiseBinaryOp(std::less<>{}, view(a), view(b));
}
template <class A, Compatible<A> B>
constexpr auto elementwise_greater_equal(const A &a, const B &b) {
  return ElementwiseBinaryOp(std::greater_equal<>{}, view(a), view(b));
}
template <class A, Compatible<A> B>
constexpr auto elementwise_less_equal(const A &a, const B &b) {
  return ElementwiseBinaryOp(std::less_equal<>{}, view(a), view(b));
}

static_assert(AbstractMatrix<ElementwiseBinaryOp<PtrMatrix<int64_t>, int,
                                                 std::multiplies<>>>,
              "ElementwiseBinaryOp isa AbstractMatrix failed");

static_assert(
  !AbstractVector<MatMatMul<PtrMatrix<int64_t>, PtrMatrix<int64_t>>>,
  "MatMul should not be an AbstractVector!");
static_assert(AbstractMatrix<MatMatMul<PtrMatrix<int64_t>, PtrMatrix<int64_t>>>,
              "MatMul is not an AbstractMatrix!");
static_assert(AbstractMatrix<Transpose<PtrMatrix<int64_t>>>);

static_assert(
  AbstractVector<decltype(-std::declval<StridedVector<int64_t>>())>);
static_assert(
  AbstractVector<decltype(-std::declval<StridedVector<int64_t>>() * 0)>);
// static_assert(std::ranges::range<StridedVector<int64_t>>);

static_assert(AbstractVector<Vector<int64_t>>);
static_assert(AbstractVector<const Vector<int64_t>>);
static_assert(AbstractVector<Vector<int64_t> &>);
static_assert(AbstractMatrix<IntMatrix<>>);
static_assert(AbstractMatrix<IntMatrix<> &>);
static_assert(!AbstractMatrix<Array<int64_t, StridedRange>>);
static_assert(!AbstractMatrix<
              Elementwise<std::negate<void>, Array<int64_t, StridedRange>>>);

static_assert(std::copyable<ManagedArray<int64_t, StridedDims<>>>);
static_assert(std::copyable<ManagedArray<int64_t, DenseDims<>>>);
static_assert(std::copyable<ManagedArray<int64_t, SquareDims<>>>);

template <typename T, typename I> struct SliceView {
  using value_type = T;
  [[no_unique_address]] MutPtrVector<T> a;
  [[no_unique_address]] PtrVector<I> i;
  struct Iterator {
    [[no_unique_address]] MutPtrVector<T> a;
    [[no_unique_address]] PtrVector<I> i;
    [[no_unique_address]] ptrdiff_t j;
    auto operator==(const Iterator &k) const -> bool { return j == k.j; }
    auto operator++() -> Iterator & {
      ++j;
      return *this;
    }
    auto operator*() -> T & { return a[i[j]]; }
    auto operator*() const -> const T & { return a[i[j]]; }
    auto operator->() -> T * { return &a[i[j]]; }
    auto operator->() const -> const T * { return &a[i[j]]; }
  };
  constexpr auto begin() -> Iterator { return Iterator{a, i, 0}; }
  constexpr auto end() -> Iterator { return Iterator{a, i, i.size()}; }
  auto operator[](ptrdiff_t j) -> T & { return a[i[j]]; }
  auto operator[](ptrdiff_t j) const -> const T & { return a[i[j]]; }
  [[nodiscard]] constexpr auto size() const -> ptrdiff_t { return i.size(); }
  constexpr auto view() -> SliceView<T, I> { return *this; }
  [[nodiscard]] constexpr auto numRow() const -> Row<1> { return {}; }
  [[nodiscard]] constexpr auto numCol() const { return col(size()); }
};

static_assert(AbstractVector<SliceView<int64_t, unsigned>>);

template <typename T> auto countNonZero(PtrMatrix<T> x) -> ptrdiff_t {
  ptrdiff_t count = 0;
  for (ptrdiff_t r = 0; r < x.numRow(); ++r) count += countNonZero(x(r, _));
  return count;
}
static_assert(std::same_as<decltype(_(0, 4) + 8), Range<ptrdiff_t, ptrdiff_t>>);
[[gnu::always_inline, gnu::flatten]] constexpr auto
any(const AbstractTensor auto &A, const auto &f) -> bool {
  auto [M, N] = shape(A);
  using T = utils::eltype_t<decltype(A)>;
  constexpr ptrdiff_t W = simd::Width<T>;
  if constexpr (W > 1) {
    if constexpr (AbstractMatrix<decltype(A)>) {
      for (ptrdiff_t r = 0; r < M; ++r) {
        for (ptrdiff_t i = 0;; i += W) {
          auto u{simd::index::unrollmask<1, W>(N, i)};
          if (!u) break;
          if (f(A[r, u])) return true;
        }
      }
    } else {
      ptrdiff_t L = RowVector<decltype(A)> ? N : M;
      for (ptrdiff_t i = 0;; i += W) {
        auto u{simd::index::unrollmask<1, W>(L, i)};
        if (!u) break;
        if (f(A[u])) return true;
      }
    }
  } else if constexpr (AbstractMatrix<decltype(A)>) {
    for (ptrdiff_t r = 0; r < M; ++r)
      for (ptrdiff_t i = 0; i < N; ++i)
        if (f(A[r, i])) return true;
  } else
    for (ptrdiff_t i = 0; i < N; ++i)
      if (f(A[i])) return true;
  return false;
}
constexpr auto anyNEZero(const AbstractTensor auto &A) -> bool {
  using T = utils::eltype_t<decltype(A)>;
  constexpr ptrdiff_t W = simd::Width<T>;
  if constexpr (W > 1)
    return any(A, [](simd::Vec<W, T> v) -> bool {
      return bool(simd::cmp::ne<W, T>(v, simd::Vec<W, T>{}));
    });
  else return any(A, [](T x) -> bool { return x != T{}; });
}
constexpr auto anyLTZero(const AbstractTensor auto &A) -> bool {
  using T = utils::eltype_t<decltype(A)>;
  constexpr ptrdiff_t W = simd::Width<T>;
  if constexpr (simd::SIMDSupported<T>)
    return any(A, [](simd::Vec<W, T> v) -> bool {
      return bool(simd::cmp::lt<W, T>(v, simd::Vec<W, T>{}));
    });
  else return any(A, [](T x) -> bool { return x < T{}; });
}
constexpr auto anyGTZero(const AbstractTensor auto &A) -> bool {
  using T = utils::eltype_t<decltype(A)>;
  constexpr ptrdiff_t W = simd::Width<T>;
  if constexpr (simd::SIMDSupported<T>)
    return any(A, [](simd::Vec<W, T> v) -> bool {
      return bool(simd::cmp::gt<W, T>(v, simd::Vec<W, T>{}));
    });
  else return any(A, [](T x) -> bool { return x > T{}; });
}
constexpr auto countNonZero(const auto &x) -> ptrdiff_t {
  return std::count_if(x.begin(), x.end(), [](auto a) { return a != 0; });
  // return std::ranges::count_if(x, [](auto x) { return x != 0; });
}

constexpr auto allZero(const AbstractTensor auto &A) -> bool {
  return !anyNEZero(A);
}
constexpr auto allLEZero(const AbstractTensor auto &A) -> bool {
  return !anyGTZero(A);
}
constexpr auto allGEZero(const AbstractTensor auto &A) -> bool {
  return !anyLTZero(A);
}
} // namespace poly::math
