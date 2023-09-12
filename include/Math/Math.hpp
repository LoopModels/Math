#pragma once
// We'll follow Julia style, so anything that's not a constructor, destructor,
// nor an operator will be outside of the struct/class.

#include "Math/Array.hpp"
#include "Math/AxisTypes.hpp"
#include "Math/Indexing.hpp"
#include "Math/Matrix.hpp"
#include "Math/MatrixDimensions.hpp"
#include "Math/Vector.hpp"
#include "Utilities/TypePromotion.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
// #ifndef NDEBUG
// #include <memory>
// #include <stacktrace>
// using stacktrace =
//     std::basic_stacktrace<std::allocator<std::stacktrace_entry>>;
// #endif

namespace poly::math {
struct Rational;

template <class A>
concept VecOrMat = AbstractVector<A> || AbstractMatrix<A>;

template <class T, class C>
concept Compatible =
  (VecOrMat<C> && std::convertible_to<T, utils::eltype_t<C>>) ||
  (VecOrMat<T> && std::convertible_to<C, utils::eltype_t<T>>) ||
  (AbstractVector<C> && AbstractVector<T>) ||
  (AbstractMatrix<C> && AbstractMatrix<T>);

template <class A, class B> struct ElTypes {
  using eltype =
    std::conditional_t<VecOrMat<B> & std::convertible_to<A, utils::eltype_t<B>>,
                       A, utils::eltype_t<A>>;
};
// returns the element type of `A` when used in a binary op with `B`
template <class A, class B> using indextype_t = typename ElTypes<A, B>::eltype;

template <typename T>
concept Trivial =
  std::is_trivially_destructible_v<T> && std::is_trivially_copyable_v<T>;
template <typename T, typename C>
concept TrivialCompatibile = Trivial<T> && Compatible<T, C>;
template <typename T>
concept TrivialVecOrMat = Trivial<T> && VecOrMat<T>;
template <typename T>
concept TrivialVec = Trivial<T> && AbstractVector<T>;
template <typename T>
concept TrivialMat = Trivial<T> && AbstractMatrix<T>;
template <typename T>
concept TrivialDataMatrix = Trivial<T> && DataMatrix<T>;

//   // TODO: binary func invocable trait?
// template <typename Op, typename T, typename S>
// concept BinaryFunction = std::is_invocable_v<Op, T, S>;

template <typename Op, typename A, typename B>
concept BinaryFuncOfElts =
  std::is_invocable_v<Op, indextype_t<A, B>, indextype_t<B, A>>;

[[gnu::flatten]] constexpr auto operator==(const AbstractMatrix auto &A,
                                           const AbstractMatrix auto &B)
  -> bool {
  const Row M = B.numRow();
  const Col N = B.numCol();
  if ((M != A.numRow()) || (N != A.numCol())) return false;
  for (ptrdiff_t r = 0; r < M; ++r)
    for (ptrdiff_t c = 0; c < N; ++c)
      if (A(r, c) != B(r, c)) return false;
  return true;
}

template <typename Op, typename A> struct Elementwise {
  using value_type =
    decltype(std::declval<Op>()(std::declval<utils::eltype_t<A>>()));
  [[no_unique_address]] Op op;
  [[no_unique_address]] A a;
  constexpr auto operator[](ptrdiff_t i) const { return op(a[i]); }
  constexpr auto operator()(ptrdiff_t i, ptrdiff_t j) const {
    return op(a(i, j));
  }

  [[nodiscard]] constexpr auto size() const { return a.size(); }
  [[nodiscard]] constexpr auto dim() const { return a.dim(); }
  [[nodiscard]] constexpr auto numRow() const -> Row { return a.numRow(); }
  [[nodiscard]] constexpr auto numCol() const -> Col { return a.numCol(); }
  [[nodiscard]] constexpr auto view() const { return *this; };
};
// scalars broadcast
template <typename S> constexpr auto get(const S &s, ptrdiff_t) -> S {
  return s;
}
template <typename S>
constexpr auto get(const S &s, ptrdiff_t, ptrdiff_t) -> S {
  return s;
}
template <typename S, LinearlyIndexable<S> V>
constexpr auto get(const V &v, ptrdiff_t i) -> S {
  return v[i];
}
template <typename S, CartesianIndexable<S> V>
constexpr auto get(const V &v, ptrdiff_t i, ptrdiff_t j) -> S {
  return v(i, j);
}

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

template <Trivial A, TrivialCompatibile<A> B, BinaryFuncOfElts<A, B> Op>
struct ElementwiseBinaryOp {
  using elta = indextype_t<A, B>;
  using eltb = indextype_t<B, A>;

  using value_type =
    decltype(std::declval<Op>()(std::declval<elta>(), std::declval<eltb>()));
  // using value_type = utils::promote_eltype_t<A, B>;
  using concrete = is_concrete_t<A, B>;
  static constexpr bool isvector = AbstractVector<A> || AbstractVector<B>;
  static constexpr bool ismatrix = AbstractMatrix<A> || AbstractMatrix<B>;

  [[no_unique_address]] Op op;
  [[no_unique_address]] A a;
  [[no_unique_address]] B b;
  constexpr auto operator[](ptrdiff_t i) const -> value_type
  requires LinearlyIndexableOrConvertible<A, elta> &&
           LinearlyIndexableOrConvertible<B, eltb>
  {
    return op(get<elta>(a, i), get<eltb>(b, i));
  }
  constexpr auto operator()(ptrdiff_t i, ptrdiff_t j) const -> value_type
  requires CartesianIndexableOrConvertible<A, elta> &&
           CartesianIndexableOrConvertible<B, eltb>
  {
    return op(get<elta>(a, i, j), get<eltb>(b, i, j));
  }

  [[nodiscard]] constexpr auto numRow() const -> Row
  requires(ismatrix)
  {
    if constexpr (AbstractMatrix<A> && AbstractMatrix<B>) {
      if constexpr (HasConcreteSize<A>)
        if constexpr (HasConcreteSize<B>) {
          const Row N = a.numRow();
          invariant(N, b.numRow());
          return N;
        } else return a.numRow();
      else if constexpr (HasConcreteSize<B>) return b.numRow();
      else return 0;
    } else if constexpr (AbstractMatrix<A>) return a.numRow();
    else if constexpr (AbstractMatrix<B>) return b.numRow();
  }
  [[nodiscard]] constexpr auto numCol() const -> Col
  requires(ismatrix)
  {
    if constexpr (AbstractMatrix<A> && AbstractMatrix<B>) {
      if constexpr (HasConcreteSize<A>)
        if constexpr (HasConcreteSize<B>) {
          const Col N = a.numCol();
          invariant(N, b.numCol());
          return N;
        } else return a.numCol();
      else if constexpr (HasConcreteSize<B>) return b.numCol();
      else return 0;
    } else if constexpr (AbstractMatrix<A>) return a.numCol();
    else if constexpr (AbstractMatrix<B>) return b.numCol();
  }
  [[nodiscard]] constexpr auto dim() const -> DenseDims
  requires(ismatrix)
  {
    return {numRow(), numCol()};
  }

  [[nodiscard]] constexpr auto size() const {
    if constexpr (AbstractVector<A> && AbstractVector<B>) {
      const ptrdiff_t N = a.size();
      invariant(N == b.size());
      return N;
    } else if constexpr (AbstractVector<A>) return ptrdiff_t(a.size());
    else if constexpr (AbstractVector<B>) return ptrdiff_t(b.size());
    else return CartesianIndex<Row, Col>{numRow(), numCol()};
  }
  [[nodiscard]] constexpr auto view() const -> auto & { return *this; };
};

template <TrivialVecOrMat C, Trivial A, Trivial B> struct AbstractSelect {
  using value_type = std::common_type_t<utils::eltype_t<A>, utils::eltype_t<B>>;
  static constexpr bool isvector = AbstractVector<C>;
  static constexpr bool ismatrix = AbstractMatrix<C>;
  [[no_unique_address]] C c;
  [[no_unique_address]] A a;
  [[no_unique_address]] B b;

  [[nodiscard]] constexpr auto numRow() const -> Row
  requires(ismatrix)
  {
    Row m = c.numRow();
    if constexpr (AbstractMatrix<A>) invariant(m, a.numRow());
    if constexpr (AbstractMatrix<B>) invariant(m, b.numRow());
    return m;
  }
  [[nodiscard]] constexpr auto numCol() const -> Col
  requires(ismatrix)
  {
    Col n = c.numCol();
    if constexpr (AbstractMatrix<A>) invariant(n, a.numCol());
    if constexpr (AbstractMatrix<B>) invariant(n, b.numCol());
    return n;
  }
  [[nodiscard]] constexpr auto dim() const -> DenseDims
  requires(ismatrix)
  {
    return {numRow(), numCol()};
  }
  [[nodiscard]] constexpr auto size() const {
    if constexpr (ismatrix) return CartesianIndex<Row, Col>{numRow(), numCol()};
    else {
      ptrdiff_t N = c.size();
      if constexpr (AbstractVector<A>) invariant(ptrdiff_t(a.size()), N);
      if constexpr (AbstractVector<B>) invariant(ptrdiff_t(b.size()), N);
      return N;
    }
  }
  [[nodiscard]] constexpr auto view() const -> auto & { return *this; };
};

inline constexpr auto view(const Trivial auto &x) { return x; }
inline constexpr auto view(const auto &x) { return x.view(); }
template <class T, class S> constexpr auto view(const Array<T, S> &x) {
  return x;
}
constexpr auto transpose(const auto &a) { return Transpose{view(a)}; }
template <typename T> constexpr auto transpose(const Transpose<T> &a) -> T {
  return a.transpose();
}

template <TrivialVecOrMat C, Trivial A, Trivial B>
struct Select : public AbstractSelect<C, A, B> {
  using value_type = AbstractSelect<C, A, B>::value_type;
  constexpr auto operator[](ptrdiff_t i) const -> value_type
  requires LinearlyIndexableOrConvertible<C, bool> &&
           LinearlyIndexableOrConvertible<A, value_type> &&
           LinearlyIndexableOrConvertible<B, value_type>
  {
    return get<bool>(this->c, i) ? get<value_type>(this->a, i)
                                 : get<value_type>(this->b, i);
  }
  constexpr auto operator()(ptrdiff_t i, ptrdiff_t j) const -> value_type
  requires CartesianIndexableOrConvertible<C, bool> &&
           CartesianIndexableOrConvertible<A, value_type> &&
           CartesianIndexableOrConvertible<B, value_type>
  {
    return get<bool>(this->c, i, j) ? get<value_type>(this->a, i, j)
                                    : get<value_type>(this->b, i, j);
  }
};
template <TrivialVecOrMat C, Trivial A, Trivial B>
Select(C c, A a, B b) -> Select<C, A, B>;
constexpr auto select(const VecOrMat auto &c, const auto &a, const auto &b) {
  return Select(view(c), view(a), view(b));
}

template <TrivialVecOrMat C, Trivial A, Trivial B, BinaryFuncOfElts<A, B> Op>
struct Conditional : public AbstractSelect<C, A, B> {
  using value_type = AbstractSelect<C, A, B>::value_type;
  [[no_unique_address]] Op op;

  constexpr auto operator[](ptrdiff_t i) const -> value_type
  requires LinearlyIndexableOrConvertible<C, bool> &&
           LinearlyIndexableOrConvertible<A, value_type> &&
           LinearlyIndexableOrConvertible<B, value_type>
  {
    auto x = get<value_type>(this->a, i);
    return get<bool>(this->c, i) ? op(x, get<value_type>(this->b, i)) : x;
  }
  constexpr auto operator()(ptrdiff_t i, ptrdiff_t j) const -> value_type
  requires CartesianIndexableOrConvertible<C, bool> &&
           CartesianIndexableOrConvertible<A, value_type> &&
           CartesianIndexableOrConvertible<B, value_type>
  {
    auto x = get<value_type>(this->a, i, j);
    return get<bool>(this->c, i, j) ? op(x, get<value_type>(this->b, i, j)) : x;
  }
};
constexpr auto conditional(auto op, const VecOrMat auto &c, const auto &a,
                           const auto &b) {
  auto vc = view(c);
  auto va = view(a);
  auto vb = view(b);
  return Conditional<decltype(vc), decltype(va), decltype(vb), decltype(op)>{
    {vc, va, vb}, op};
}

template <AbstractMatrix A, AbstractMatrix B> struct MatMatMul {
  using value_type = utils::promote_eltype_t<A, B>;
  using concrete = is_concrete_t<A, B>;
  [[no_unique_address]] A a;
  [[no_unique_address]] B b;
  constexpr auto operator()(ptrdiff_t i, ptrdiff_t j) const -> value_type {
    static_assert(AbstractMatrix<B>, "B should be an AbstractMatrix");
    value_type s{};
    for (ptrdiff_t k = 0; k < ptrdiff_t(a.numCol()); ++k)
      s += a(i, k) * b(k, j);
    return s;
  }
  [[nodiscard]] constexpr auto numRow() const -> Row { return a.numRow(); }
  [[nodiscard]] constexpr auto numCol() const -> Col { return b.numCol(); }
  [[nodiscard]] constexpr auto size() const -> CartesianIndex<Row, Col> {
    invariant(ptrdiff_t(a.numCol()) == ptrdiff_t(b.numRow()));
    return {numRow(), numCol()};
  }
  [[nodiscard]] constexpr auto dim() const -> DenseDims {
    invariant(ptrdiff_t(a.numCol()) == ptrdiff_t(b.numRow()));
    return {numRow(), numCol()};
  }
  [[nodiscard]] constexpr auto view() const { return *this; };
  [[nodiscard]] constexpr auto transpose() const { return Transpose{*this}; };
};
template <AbstractMatrix A, AbstractVector B> struct MatVecMul {
  using value_type = utils::promote_eltype_t<A, B>;
  using concrete = is_concrete_t<A, B>;
  [[no_unique_address]] A a;
  [[no_unique_address]] B b;
  constexpr auto operator[](ptrdiff_t i) const -> value_type {
    invariant(a.numCol() == b.size());
    value_type s = 0;
    for (ptrdiff_t k = 0; k < a.numCol(); ++k) s += a(i, k) * b[k];
    return s;
  }
  [[nodiscard]] constexpr auto size() const -> ptrdiff_t {
    return ptrdiff_t(a.numRow());
  }
  constexpr auto view() const { return *this; };
};

//
// Vectors
//

static_assert(!AbstractMatrix<StridedVector<int64_t>>);

// static_assert(std::is_trivially_copyable_v<MutStridedVector<int64_t>>);
static_assert(std::is_trivially_copyable_v<
              Elementwise<std::negate<>, StridedVector<int64_t>>>);
static_assert(Trivial<Elementwise<std::negate<>, StridedVector<int64_t>>>);

constexpr void swap(MutPtrMatrix<int64_t> A, Row i, Row j) {
  if (i == j) return;
  Col N = A.numCol();
  invariant((i < A.numRow()) && (j < A.numRow()));
  for (Col n = 0; n < N; ++n) std::swap(A(i, n), A(j, n));
}
constexpr void swap(MutPtrMatrix<int64_t> A, Col i, Col j) {
  if (i == j) return;
  Row M = A.numRow();
  invariant((i < A.numCol()) && (j < A.numCol()));
  for (Row m = 0; m < M; ++m) std::swap(A(m, i), A(m, j));
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

template <typename T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>;
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
static_assert(AbstractMatrix<Array<int64_t, SquareDims>>);
static_assert(AbstractMatrix<ManagedArray<int64_t, SquareDims>>);

constexpr auto operator*(const AbstractMatrix auto &a,
                         const AbstractMatrix auto &b) {
  auto AA{a.view()};
  auto BB{b.view()};
  invariant(ptrdiff_t(AA.numCol()) == ptrdiff_t(BB.numRow()));
  return MatMatMul<decltype(AA), decltype(BB)>{.a = AA, .b = BB};
}
constexpr auto operator*(const AbstractMatrix auto &a,
                         const AbstractVector auto &b) {
  auto AA{a.view()};
  auto BB{b.view()};
  invariant(ptrdiff_t(AA.numCol()) == BB.size());
  return MatVecMul<decltype(AA), decltype(BB)>{.a = AA, .b = BB};
}
constexpr auto operator*(const AbstractVector auto &a,
                         const AbstractVector auto &b) {
  return ElementwiseBinaryOp(std::multiplies<>{}, view(a), view(b));
}
template <AbstractVector M, utils::ElementOf<M> S>
constexpr auto operator*(const M &b, S a) {
  return ElementwiseBinaryOp(std::multiplies<>{}, view(b), view(a));
}
template <AbstractVector M, utils::ElementOf<M> S>
constexpr auto operator*(S a, const M &b) {
  return ElementwiseBinaryOp(std::multiplies<>{}, view(a), view(b));
}
template <AbstractMatrix M, utils::ElementOf<M> S>
constexpr auto operator*(S a, const M &b) {
  return ElementwiseBinaryOp(std::multiplies<>{}, view(a), view(b));
}
template <AbstractMatrix M, utils::ElementOf<M> S>
constexpr auto operator*(const M &b, S a) {
  return ElementwiseBinaryOp(std::multiplies<>{}, view(b), view(a));
}

template <class A, Compatible<A> B>
constexpr auto operator+(const A &a, const B &b) {
  return ElementwiseBinaryOp(std::plus<>{}, view(a), view(b));
}
template <class A, Compatible<A> B>
constexpr auto operator-(const A &a, const B &b) {
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

template <AbstractVector V>
constexpr auto operator*(const Transpose<V> &at, const AbstractVector auto &b) {
  utils::promote_eltype_t<V, decltype(b)> s{};
  auto a = at.transpose();
  ptrdiff_t l = a.size();
  invariant(l == b.size());
  for (ptrdiff_t i = 0; i < l; ++i) s += a[i] * b[i];
  return s;
}

static_assert(
  AbstractVector<decltype(-std::declval<StridedVector<int64_t>>())>);
static_assert(
  AbstractVector<decltype(-std::declval<StridedVector<int64_t>>() * 0)>);
// static_assert(std::ranges::range<StridedVector<int64_t>>);

static_assert(AbstractVector<Vector<int64_t>>);
static_assert(AbstractVector<const Vector<int64_t>>);
static_assert(AbstractVector<Vector<int64_t> &>);
static_assert(AbstractMatrix<IntMatrix>);
static_assert(AbstractMatrix<IntMatrix &>);

static_assert(std::copyable<ManagedArray<int64_t, StridedDims>>);
static_assert(std::copyable<ManagedArray<int64_t, DenseDims>>);
static_assert(std::copyable<ManagedArray<int64_t, SquareDims>>);

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
};

static_assert(AbstractVector<SliceView<int64_t, unsigned>>);
constexpr auto abs2(auto x) { return x * x; }

template <AbstractVector B> constexpr auto norm2(const B &A) {
  utils::eltype_t<B> s = 0;
  for (ptrdiff_t j = 0; j < A.size(); ++j) s += abs2(A[j]);
  return s;
}
template <AbstractMatrix B> constexpr auto norm2(const B &A) {
  utils::eltype_t<B> s = 0;
  for (ptrdiff_t i = 0; i < A.numRow(); ++i)
    for (ptrdiff_t j = 0; j < A.numCol(); ++j) s += abs2(A(i, j));
  return s;
}

// template <class T, size_t N> struct Zip {
//   std::array<T, N> a;
//   constexpr Zip(std::initializer_list<T> b) : a(b) {}
//   constexpr auto operator[](ptrdiff_t i) -> T & { return a[i]; }
//   constexpr auto operator[](ptrdiff_t i) const -> const T & { return a[i]; }
//   constexpr auto size() const -> ptrdiff_t { return N; }
//   constexpr auto begin() -> typename std::array<T, N>::iterator {
//     return a.begin();
//   }
//   constexpr auto end() -> typename std::array<T, N>::iterator {
//     return a.end();
//   }
//   constexpr auto begin() const -> typename std::array<T, N>::const_iterator {
//     return a.begin();
//   }
//   constexpr auto end() const -> typename std::array<T, N>::const_iterator {
//     return a.end();
//   }

// };

// exports:
// NOLINTNEXTLINE(bugprone-reserved-identifier)
template <typename T> auto anyNEZero(PtrMatrix<T> x) -> bool {
  for (ptrdiff_t r = 0; r < x.numRow(); ++r)
    if (anyNEZero(x(r, _))) return true;
  return false;
}
template <typename T> auto allZero(PtrMatrix<T> x) -> bool {
  return !anyNEZero(x);
}
template <typename T> auto allLEZero(PtrMatrix<T> x) -> bool {
  for (ptrdiff_t r = 0; r < x.numRow(); ++r)
    if (!allLEZero(x(r, _))) return false;
  return true;
}
template <typename T> auto allGEZero(PtrMatrix<T> x) -> bool {
  for (ptrdiff_t r = 0; r < x.numRow(); ++r)
    if (!allGEZero(x(r, _))) return false;
  return true;
}
template <typename T> auto countNonZero(PtrMatrix<T> x) -> ptrdiff_t {
  ptrdiff_t count = 0;
  for (ptrdiff_t r = 0; r < x.numRow(); ++r) count += countNonZero(x(r, _));
  return count;
}
} // namespace poly::math
