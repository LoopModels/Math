#pragma once

#include "Math/Array.hpp"
#include "Math/MatrixDimensions.hpp"
#include "Utilities/Allocators.hpp"

namespace poly::math {
using utils::Arena, utils::WArena, utils::OwningArena;
template <class T>
constexpr auto vector(std::allocator<T>, unsigned int M) -> Vector<T> {
  return Vector<T>(M);
}
template <class T>
constexpr auto vector(WArena<T> alloc, unsigned int M)
  -> ResizeableView<T, unsigned> {
  return {alloc.allocate(M), M, M};
}
template <class T, size_t SlabSize, bool BumpUp>
constexpr auto vector(Arena<SlabSize, BumpUp> *alloc, unsigned int M)
  -> ResizeableView<T, unsigned> {
  return {alloc->template allocate<T>(M), M, M};
}

template <class T>
constexpr auto vector(std::allocator<T>, unsigned int M, T x) -> Vector<T> {
  return {M, x};
}
template <class T>
constexpr auto vector(WArena<T> alloc, unsigned int M, T x)
  -> ResizeableView<T, unsigned> {
  ResizeableView<T, unsigned> a{alloc.allocate(M), M, M};
  a.fill(x);
  return a;
}
template <class T, size_t SlabSize, bool BumpUp>
constexpr auto vector(Arena<SlabSize, BumpUp> *alloc, unsigned int M, T x)
  -> ResizeableView<T, unsigned> {
  ResizeableView<T, unsigned> a{alloc->template allocate<T>(M), M, M};
  a.fill(x);
  return a;
}

template <class T>
constexpr auto matrix(std::allocator<T>, unsigned int M) -> SquareMatrix<T> {
  return SquareDims{M};
}
template <class T>
constexpr auto matrix(WArena<T> alloc, unsigned int M)
  -> MutSquarePtrMatrix<T> {
  return {alloc.allocate(M * M), SquareDims{M}};
}
template <class T, size_t SlabSize, bool BumpUp>
constexpr auto matrix(Arena<SlabSize, BumpUp> *alloc, unsigned int M)
  -> MutSquarePtrMatrix<T> {
  return {alloc->template allocate<T>(M * M), SquareDims{M}};
}
template <class T>
constexpr auto matrix(std::allocator<T>, unsigned int M, T x)
  -> SquareMatrix<T> {
  return {SquareDims{M}, x};
}
template <class T>
constexpr auto matrix(WArena<T> alloc, unsigned int M, T x)
  -> MutSquarePtrMatrix<T> {
  MutSquarePtrMatrix<T> A{alloc.allocate(M * M), SquareDims{M}};
  A.fill(x);
  return A;
}
template <class T, size_t SlabSize, bool BumpUp>
constexpr auto matrix(Arena<SlabSize, BumpUp> *alloc, unsigned int M, T x)
  -> MutSquarePtrMatrix<T> {
  MutSquarePtrMatrix<T> A{alloc->template allocate<T>(M * M), SquareDims{M}};
  A.fill(x);
  return A;
}

template <class T>
constexpr auto matrix(std::allocator<T>, Row M, Col N) -> DenseMatrix<T> {
  return DenseDims{M, N};
}
template <class T>
constexpr auto matrix(WArena<T> alloc, Row M, Col N) -> MutDensePtrMatrix<T> {
  return {alloc.allocate(M * N), DenseDims{M, N}};
}
template <class T, size_t SlabSize, bool BumpUp>
constexpr auto matrix(Arena<SlabSize, BumpUp> *alloc, Row M, Col N)
  -> MutDensePtrMatrix<T> {
  return {alloc->template allocate<T>(M * N), M, N};
}
template <class T>
constexpr auto matrix(std::allocator<T>, Row M, Col N, T x) -> DenseMatrix<T> {
  return {DenseDims{M, N}, x};
}
template <class T>
constexpr auto matrix(WArena<T> alloc, Row M, Col N, T x)
  -> MutDensePtrMatrix<T> {
  MutDensePtrMatrix<T> A{alloc.allocate(M * N), DenseDims{M, N}};
  A.fill(x);
  return A;
}
template <class T, size_t SlabSize, bool BumpUp>
constexpr auto matrix(Arena<SlabSize, BumpUp> *alloc, Row M, Col N, T x)
  -> MutDensePtrMatrix<T> {
  MutDensePtrMatrix<T> A{alloc->template allocate<T>(M * N), DenseDims{M, N}};
  A.fill(x);
  return A;
}

template <class T>
constexpr auto identity(std::allocator<T>, unsigned int M) -> SquareMatrix<T> {
  SquareMatrix<T> A{M, T{}};
  A.diag() << T{1};
  return A;
}
template <class T>
constexpr auto identity(WArena<T> alloc, unsigned int M)
  -> MutSquarePtrMatrix<T> {
  MutSquarePtrMatrix<T> A{matrix(alloc, M, T{})};
  A.diag() << T{1};
  return A;
}
template <class T, size_t SlabSize, bool BumpUp>
constexpr auto identity(Arena<SlabSize, BumpUp> *alloc, unsigned int M)
  -> MutSquarePtrMatrix<T> {
  MutSquarePtrMatrix<T> A{matrix(alloc, M, T{})};
  A.diag() << T{1};
  return A;
}

template <typename T, typename I>
concept Alloc = requires(T t, unsigned int M, Row r, Col c, I i) {
  { identity<I>(t, M) } -> std::convertible_to<MutSquarePtrMatrix<I>>;
  { matrix<I>(t, M) } -> std::convertible_to<MutSquarePtrMatrix<I>>;
  { matrix<I>(t, M, i) } -> std::convertible_to<MutSquarePtrMatrix<I>>;
  { matrix<I>(t, r, c) } -> std::convertible_to<MutDensePtrMatrix<I>>;
  { matrix(t, r, c, i) } -> std::convertible_to<MutDensePtrMatrix<I>>;
  { vector<I>(t, M) } -> std::convertible_to<MutPtrVector<I>>;
};
static_assert(Alloc<std::allocator<int64_t>, int64_t>);
static_assert(Alloc<Arena<> *, int64_t>);
} // namespace poly::math
