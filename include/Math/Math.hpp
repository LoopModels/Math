#pragma once
// We'll follow Julia style, so anything that's not a constructor, destructor,
// nor an operator will be outside of the struct/class.

#include "Math/Array.hpp"
#include "Math/AxisTypes.hpp"
#include "Math/Indexing.hpp"
#include "Math/Matrix.hpp"
#include "Math/MatrixDimensions.hpp"
#include "Math/SIMD.hpp"
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

namespace poly::math {
struct Rational;

static_assert(DoCopy<int64_t>);
static_assert(DoCopy<PtrVector<int64_t>>);
static_assert(DoCopy<DensePtrMatrix<int64_t>>);
static_assert(!DoCopy<DenseMatrix<int64_t>>);

struct Negate {
  constexpr auto operator()(const auto &x) const -> decltype(auto) {
    return -x;
  }
};
struct Plus {
  constexpr auto operator()(const auto &x, const auto &y) const
    -> decltype(auto) {
    return x + y;
  }
};
struct Minus {
  constexpr auto operator()(const auto &x, const auto &y) const
    -> decltype(auto) {
    return x - y;
  }
};
struct Mul {
  constexpr auto operator()(const auto &x, const auto &y) const
    -> decltype(auto) {
    return x * y;
  }
};
struct Div {
  constexpr auto operator()(const auto &x, const auto &y) const
    -> decltype(auto) {
    return x / y;
  }
};

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

template <typename Op, typename A> struct ElementwiseUnaryOp {
  using value_type = typename A::value_type;
  [[no_unique_address]] Op op;
  [[no_unique_address]] A a;
  constexpr auto operator[](auto i) const -> decltype(auto) { return op(a[i]); }
  constexpr auto operator()(auto i, auto j) const -> decltype(auto) {
    return op(a(i, j));
  }

  [[nodiscard]] constexpr auto size() const { return a.size(); }
  [[nodiscard]] constexpr auto paddedlength() const -> ptrdiff_t {
    return paddedlen(a);
  }
  [[nodiscard]] constexpr auto dim() const { return a.dim(); }
  [[nodiscard]] constexpr auto numRow() const -> Row { return a.numRow(); }
  [[nodiscard]] constexpr auto numCol() const -> Col { return a.numCol(); }
};
// scalars broadcast
constexpr auto get(const Scalar auto &A, ptrdiff_t) -> decltype(auto) {
  return A;
}
constexpr auto get(const Scalar auto &A, ptrdiff_t, ptrdiff_t)
  -> decltype(auto) {
  return A;
}
constexpr auto get(const AbstractVector auto &A, ptrdiff_t i)
  -> decltype(auto) {
  return A[i];
}
constexpr auto get(const AbstractMatrix auto &A, ptrdiff_t i, ptrdiff_t j)
  -> decltype(auto) {
  return A(i, j);
}

template <typename Op, typename A>
ElementwiseUnaryOp(Op, A) -> ElementwiseUnaryOp<Op, const A &>;
template <typename Op, DoCopy A>
ElementwiseUnaryOp(Op, A) -> ElementwiseUnaryOp<Op, std::remove_cvref_t<A>>;

// unroll index
template <ptrdiff_t W, ptrdiff_t N, typename P>
constexpr auto get(const PrimitiveScalar auto &A, simd::Unroll<W, N, P>)
  -> decltype(auto) {
  return A;
}
template <ptrdiff_t W, ptrdiff_t N, typename P>
constexpr auto get(const AbstractVector auto &A, simd::Unroll<W, N, P> i)
  -> decltype(auto) {
  return A[i];
}
// tile index
template <ptrdiff_t W, ptrdiff_t M, ptrdiff_t N, typename P>
constexpr auto get(const PrimitiveScalar auto &A, simd::Tile<W, M, N, P>)
  -> decltype(auto) {
  return A;
}
template <ptrdiff_t W, ptrdiff_t M, ptrdiff_t N, typename P>
constexpr auto get(const AbstractMatrix auto &A, simd::Tile<W, M, N, P> i)
  -> decltype(auto) {
  return A[i];
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
template <class Op, class A, class B> struct ElementwiseVectorBinaryOp {
  using value_type = utils::promote_eltype_t<A, B>;
  using concrete = is_concrete_t<A, B>;

  [[no_unique_address]] Op op;
  [[no_unique_address]] A a;
  [[no_unique_address]] B b;
  constexpr ElementwiseVectorBinaryOp(Op _op, A _a, B _b)
    : op(_op), a(_a), b(_b) {}
  constexpr auto operator[](auto i) const -> decltype(auto) {
    return op(get(a, i), get(b, i));
  }
  [[nodiscard]] constexpr auto paddedlength() const -> ptrdiff_t {
    if constexpr (AbstractVector<A> && AbstractVector<B>)
      return std::min<ptrdiff_t>(paddedlen(a), paddedlen(b));
    else if constexpr (AbstractVector<A>) return paddedlen(a);
    else return paddedlen(b);
  }
  [[nodiscard]] constexpr auto size() const -> ptrdiff_t {
    if constexpr (AbstractVector<A> && AbstractVector<B>) {
      const ptrdiff_t N = a.size();
      invariant(N == b.size());
      return N;
    } else if constexpr (AbstractVector<A>) {
      return a.size();
    } else { // if constexpr (AbstractVector<B>) {
      return b.size();
    }
  }
  [[nodiscard]] constexpr auto dim() const -> ptrdiff_t { return size(); }
}; // namespace poly::math
template <class Op, class A, class B> struct ElementwiseMatrixBinaryOp {
  using value_type = utils::promote_eltype_t<A, B>;
  using concrete = is_concrete_t<A, B>;
  [[no_unique_address]] Op op;
  [[no_unique_address]] A a;
  [[no_unique_address]] B b;
  constexpr ElementwiseMatrixBinaryOp(Op _op, A _a, B _b)
    : op(_op), a(_a), b(_b) {}
  constexpr auto operator()(auto i, auto j) const -> decltype(auto) {
    return op(get(a, i, j), get(b, i, j));
  }
  [[nodiscard]] constexpr auto numRow() const -> Row {
    // static_assert(AbstractMatrix<A> || std::integral<A> ||
    //                 std::floating_point<A>,
    //               "Argument A to elementwise binary op is not a matrix.");
    // static_assert(AbstractMatrix<B> || std::integral<B> ||
    //                 std::floating_point<B>,
    //               "Argument B to elementwise binary op is not a matrix.");
    if constexpr (AbstractMatrix<A> && AbstractMatrix<B>) {
      if constexpr (HasConcreteSize<A>) {
        if constexpr (HasConcreteSize<B>) {
          const Row N = a.numRow();
          invariant(N, b.numRow());
          return N;
        } else {
          return a.numRow();
        }
      } else if constexpr (HasConcreteSize<B>) {
        return b.numRow();
      } else {
        return 0;
      }
    } else if constexpr (AbstractMatrix<A>) {
      return a.numRow();
    } else if constexpr (AbstractMatrix<B>) {
      return b.numRow();
    }
  }
  [[nodiscard]] constexpr auto numCol() const -> Col {
    // static_assert(AbstractMatrix<A> || std::integral<A> ||
    //                 std::floating_point<A>,
    //               "Argument A to elementwise binary op is not a matrix.");
    // static_assert(AbstractMatrix<B> || std::integral<B> ||
    //                 std::floating_point<B>,
    //               "Argument B to elementwise binary op is not a matrix.");
    if constexpr (AbstractMatrix<A> && AbstractMatrix<B>) {
      if constexpr (HasConcreteSize<A>) {
        if constexpr (HasConcreteSize<B>) {
          const Col N = a.numCol();
          invariant(N, b.numCol());
          return N;
        } else {
          return a.numCol();
        }
      } else if constexpr (HasConcreteSize<B>) {
        return b.numCol();
      } else {
        return 0;
      }
    } else if constexpr (AbstractMatrix<A>) {
      return a.numCol();
    } else if constexpr (AbstractMatrix<B>) {
      return b.numCol();
    }
  }
  [[nodiscard]] constexpr auto size() const -> CartesianIndex<Row, Col> {
    return {numRow(), numCol()};
  }
  [[nodiscard]] constexpr auto dim() const -> DenseDims {
    return {numRow(), numCol()};
  }
};

template <AbstractMatrix A, AbstractMatrix B> struct MatMatMul {
  using value_type = utils::promote_eltype_t<A, B>;
  using concrete = is_concrete_t<A, B>;
  [[no_unique_address]] A a;
  [[no_unique_address]] B b;
  constexpr auto operator()(ptrdiff_t i, ptrdiff_t j) const -> decltype(auto) {
    static_assert(AbstractMatrix<B>, "B should be an AbstractMatrix");
    value_type s{}; // hopefully NRVO
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
  [[nodiscard]] constexpr auto transpose() const { return Transpose{*this}; };
};

template <AbstractMatrix A, AbstractVector B> struct MatVecMul {
  using value_type = utils::promote_eltype_t<A, B>;
  using concrete = is_concrete_t<A, B>;
  [[no_unique_address]] const A &a;
  [[no_unique_address]] const B &b;
  constexpr auto operator[](ptrdiff_t i) const -> decltype(auto) {
    static_assert(AbstractVector<B>, "B should be an AbstractVector");
    value_type s = 0;
    for (ptrdiff_t k = 0; k < a.numCol(); ++k) s += a(i, k) * b[k];
    return s;
  }
  template <ptrdiff_t W, ptrdiff_t N, typename P>
  constexpr auto operator[](simd::Unroll<W, N, P> i) const
    -> Unrolled<value_type, W, N> {
    // this is really quite inefficient compared to how
    // vector.transpose() * matrix
    // would be, but v'*m isn't supported yet.
    static_assert(AbstractVector<B>, "B should be an AbstractVector");
    Unrolled<value_type, W, N> s{};
    for (ptrdiff_t k = 0; k < a.numCol(); ++k) s += a(_, k)[i] * b[k];
    return s;
  }
  [[nodiscard]] constexpr auto size() const -> ptrdiff_t {
    return ptrdiff_t(a.numRow());
  }
  [[nodiscard]] constexpr auto dim() const -> ptrdiff_t {
    return ptrdiff_t(a.numRow());
  }
};
template <class Op, class A, class B>
ElementwiseVectorBinaryOp(Op, const A &, const B &)
  -> ElementwiseVectorBinaryOp<
    Op, std::conditional_t<DoCopy<A>, std::remove_cvref_t<A>, const A &>,
    std::conditional_t<DoCopy<B>, std::remove_cvref_t<B>, const B &>>;
template <class Op, class A, class B>
ElementwiseMatrixBinaryOp(Op, const A &, const B &)
  -> ElementwiseMatrixBinaryOp<
    Op, std::conditional_t<DoCopy<A>, std::remove_cvref_t<A>, const A &>,
    std::conditional_t<DoCopy<B>, std::remove_cvref_t<B>, const B &>>;
template <AbstractMatrix A, AbstractMatrix B>
MatMatMul(const A &a, const B &b) -> MatMatMul<
  std::conditional_t<DoCopy<A>, std::remove_cvref_t<A>, const A &>,
  std::conditional_t<DoCopy<B>, std::remove_cvref_t<B>, const B &>>;
template <AbstractMatrix A, AbstractVector B>
MatVecMul(const A &a, const B &b) -> MatVecMul<
  std::conditional_t<DoCopy<A>, std::remove_cvref_t<A>, const A &>,
  std::conditional_t<DoCopy<B>, std::remove_cvref_t<B>, const B &>>;

//
// Vectors
//

static_assert(!AbstractMatrix<StridedVector<int64_t>>);

// static_assert(std::is_trivially_copyable_v<MutStridedVector<int64_t>>);
static_assert(std::is_trivially_copyable_v<
              ElementwiseUnaryOp<Negate, StridedVector<int64_t>>>);
static_assert(Trivial<ElementwiseUnaryOp<Negate, StridedVector<int64_t>>>);

constexpr auto allMatch(const AbstractVector auto &x0,
                        const AbstractVector auto &x1) -> bool {
  ptrdiff_t N = x0.size();
  if (N != x1.size()) return false;
  for (ptrdiff_t n = 0; n < N; ++n)
    if (x0(n) != x1(n)) return false;
  return true;
}

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
static_assert(TriviallyCopyable<Mul>);
static_assert(Trivial<ElementwiseMatrixBinaryOp<Mul, PtrMatrix<int64_t>, int>>);
static_assert(Trivial<MatMatMul<PtrMatrix<int64_t>, PtrMatrix<int64_t>>>);

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
  return ElementwiseUnaryOp{Negate{}, a};
}
constexpr auto operator-(const AbstractMatrix auto &a) {
  return ElementwiseUnaryOp{Negate{}, a};
}
static_assert(AbstractMatrix<ElementwiseUnaryOp<Negate, PtrMatrix<int64_t>>>);
static_assert(AbstractMatrix<Array<int64_t, SquareDims>>);
static_assert(AbstractMatrix<ManagedArray<int64_t, SquareDims>>);

constexpr auto operator*(const AbstractMatrix auto &a,
                         const AbstractMatrix auto &b) {
  invariant(ptrdiff_t(a.numCol()) == ptrdiff_t(b.numRow()));
  return MatMatMul(a, b);
  // return MatMatMul<decltype(a), decltype(b)>{.a = a, .b = b};
}
constexpr auto operator*(const AbstractMatrix auto &a,
                         const AbstractVector auto &b) {
  invariant(ptrdiff_t(a.numCol()) == b.size());
  return MatVecMul{a, b};
}
constexpr auto operator*(const AbstractVector auto &a,
                         const AbstractVector auto &b) {
  return ElementwiseVectorBinaryOp(Mul{}, a, b);
}

template <AbstractVector M, utils::ElementOf<M> S>
constexpr auto operator+(S a, const M &b) {
  return ElementwiseVectorBinaryOp(Plus{}, a, b);
}
template <AbstractVector M, utils::ElementOf<M> S>
constexpr auto operator+(const M &b, S a) {
  return ElementwiseVectorBinaryOp(Plus{}, b, a);
}
template <AbstractMatrix M, utils::ElementOf<M> S>
constexpr auto operator+(S a, const M &b) {
  return ElementwiseMatrixBinaryOp(Plus{}, a, b);
}
template <AbstractMatrix M, utils::ElementOf<M> S>
constexpr auto operator+(const M &b, S a) {
  return ElementwiseMatrixBinaryOp(Plus{}, b, a);
}

template <AbstractVector M, utils::ElementOf<M> S>
constexpr auto operator-(S a, const M &b) {
  return ElementwiseVectorBinaryOp(Minus{}, a, b);
}
template <AbstractVector M, utils::ElementOf<M> S>
constexpr auto operator-(const M &b, S a) {
  return ElementwiseVectorBinaryOp(Minus{}, b, a);
}
template <AbstractMatrix M, utils::ElementOf<M> S>
constexpr auto operator-(S a, const M &b) {
  return ElementwiseMatrixBinaryOp(Minus{}, a, b);
}
template <AbstractMatrix M, utils::ElementOf<M> S>
constexpr auto operator-(const M &b, S a) {
  return ElementwiseMatrixBinaryOp(Minus{}, b, a);
}

template <AbstractVector M, utils::ElementOf<M> S>
constexpr auto operator*(S a, const M &b) {
  return ElementwiseVectorBinaryOp(Mul{}, a, b);
}
template <AbstractVector M, utils::ElementOf<M> S>
constexpr auto operator*(const M &b, S a) {
  return ElementwiseVectorBinaryOp(Mul{}, b, a);
}
template <AbstractMatrix M, utils::ElementOf<M> S>
constexpr auto operator*(S a, const M &b) {
  return ElementwiseMatrixBinaryOp(Mul{}, a, b);
}
template <AbstractMatrix M, utils::ElementOf<M> S>
constexpr auto operator*(const M &b, S a) {
  return ElementwiseMatrixBinaryOp(Mul{}, b, a);
}

template <AbstractVector M, utils::ElementOf<M> S>
constexpr auto operator/(S a, const M &b) {
  return ElementwiseVectorBinaryOp(Div{}, a, b);
}
template <AbstractVector M, utils::ElementOf<M> S>
constexpr auto operator/(const M &b, S a) {
  return ElementwiseVectorBinaryOp(Div{}, b, a);
}
template <AbstractMatrix M, utils::ElementOf<M> S>
constexpr auto operator/(S a, const M &b) {
  return ElementwiseMatrixBinaryOp(Div{}, a, b);
}
template <AbstractMatrix M, utils::ElementOf<M> S>
constexpr auto operator/(const M &b, S a) {
  return ElementwiseMatrixBinaryOp(Div{}, b, a);
}

constexpr auto operator+(const AbstractVector auto &a,
                         const AbstractVector auto &b) {
  return ElementwiseVectorBinaryOp(Plus{}, a, b);
}
constexpr auto operator+(const AbstractMatrix auto &a,
                         const AbstractMatrix auto &b) {
  return ElementwiseMatrixBinaryOp(Plus{}, a, b);
}
constexpr auto operator-(const AbstractVector auto &a,
                         const AbstractVector auto &b) {
  return ElementwiseVectorBinaryOp(Minus{}, a, b);
}
constexpr auto operator-(const AbstractMatrix auto &a,
                         const AbstractMatrix auto &b) {
  return ElementwiseMatrixBinaryOp(Minus{}, a, b);
}

constexpr auto operator/(const AbstractVector auto &a,
                         const AbstractVector auto &b) {
  return ElementwiseVectorBinaryOp(Div{}, a, b);
}

static_assert(
  AbstractMatrix<ElementwiseMatrixBinaryOp<Mul, PtrMatrix<int64_t>, int>>,
  "ElementwiseBinaryOp isa AbstractMatrix failed");

static_assert(
  !AbstractVector<MatMatMul<PtrMatrix<int64_t>, PtrMatrix<int64_t>>>,
  "MatMul should not be an AbstractVector!");
static_assert(AbstractMatrix<MatMatMul<PtrMatrix<int64_t>, PtrMatrix<int64_t>>>,
              "MatMul is not an AbstractMatrix!");
static_assert(AbstractMatrix<Transpose<PtrMatrix<int64_t>>>);

template <AbstractVector V>
constexpr auto operator*(const Transpose<V> &a, const AbstractVector auto &b) {
  typename V::value_type s = 0;
  for (ptrdiff_t i = 0; i < b.size(); ++i) s += a.a(i) * b(i);
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
  [[nodiscard]] constexpr auto dim() const -> ptrdiff_t { return i.size(); }
};

static_assert(AbstractVector<SliceView<int64_t, unsigned>>);

template <AbstractVector B> constexpr auto norm2(const B &A) {
  using T = typename B::value_type;
  T s = 0;
  for (ptrdiff_t j = 0; j < A.numCol(); ++j) s += A(j) * A(j);
  return s;
}
template <AbstractMatrix B> constexpr auto norm2(const B &A) {
  using T = typename B::value_type;
  T s = 0;
  for (ptrdiff_t i = 0; i < A.numRow(); ++i)
    for (ptrdiff_t j = 0; j < A.numCol(); ++j) s += A(i, j) * A(i, j);
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
