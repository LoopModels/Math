#pragma once

#include "Alloc/Arena.hpp"
#include "Math/Array.hpp"
#include "Math/MatrixDimensions.hpp"

namespace poly::math {
using alloc::Arena, alloc::WArena, alloc::OwningArena, utils::eltype_t;
template <alloc::FreeAllocator A>
constexpr auto vector(A a, unsigned int M)
  -> ManagedArray<eltype_t<A>, unsigned, PreAllocStorage<eltype_t<A>>(), A> {
  return {M, a};
}

template <alloc::Allocator A, typename T>
using rebound_alloc =
  typename std::allocator_traits<A>::template rebind_alloc<T>;

template <class T, alloc::FreeAllocator A>
constexpr auto vector(A a, unsigned int M) {
  if constexpr (std::same_as<T, eltype_t<A>>) return vector(a, M);
  else return vector(rebound_alloc<A, T>{}, M);
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

template <alloc::FreeAllocator A>
constexpr auto vector(A a, unsigned int M, eltype_t<A> x)
  -> ManagedArray<eltype_t<A>, unsigned, PreAllocStorage<eltype_t<A>>(), A> {
  return {M, x, a};
}

template <typename T, alloc::FreeAllocator A>
constexpr auto vector(A a, unsigned int M, std::type_identity_t<T> x) {
  if constexpr (std::same_as<T, eltype_t<A>>) return vector(a, M, x);
  else return vector(rebound_alloc<A, T>{}, M, x);
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

template <alloc::FreeAllocator A>
constexpr auto matrix(A a, unsigned int M)
  -> ManagedArray<eltype_t<A>, SquareDims, PreAllocSquareStorage<eltype_t<A>>(),
                  A> {
  return {SquareDims{M}, a};
}

template <class T, alloc::FreeAllocator A>
constexpr auto matrix(A a, unsigned int M) {
  if constexpr (std::same_as<T, eltype_t<A>>) return matrix(a, M);
  else return matrix(rebound_alloc<A, T>{}, M);
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
template <alloc::FreeAllocator A>
constexpr auto matrix(A a, unsigned int M, eltype_t<A> x)
  -> ManagedArray<eltype_t<A>, SquareDims, PreAllocSquareStorage<eltype_t<A>>(),
                  A> {
  return {SquareDims{M}, x, a};
}
template <class T, alloc::FreeAllocator A>
constexpr auto matrix(A a, unsigned int M, std::type_identity_t<T> x) {
  if constexpr (std::same_as<T, eltype_t<A>>) return matrix(a, M, x);
  else return matrix(rebound_alloc<A, T>{}, M, x);
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

template <alloc::FreeAllocator A>
constexpr auto matrix(A a, Row M, Col N)
  -> ManagedArray<eltype_t<A>, DenseDims, PreAllocStorage<eltype_t<A>>(), A> {
  return {DenseDims{M, N}, a};
}
template <class T, alloc::FreeAllocator A>
constexpr auto matrix(A a, Row M, Col N) {
  if constexpr (std::same_as<T, eltype_t<A>>) return matrix(a, M, N);
  else return matrix(rebound_alloc<A, T>{}, M, N);
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
template <alloc::FreeAllocator A>
constexpr auto matrix(A a, Row M, Col N, eltype_t<A> x)
  -> ManagedArray<eltype_t<A>, DenseDims, PreAllocStorage<eltype_t<A>>(), A> {
  return {DenseDims{M, N}, x, a};
}
template <class T, alloc::FreeAllocator A>
constexpr auto matrix(A a, Row M, Col N, std::type_identity_t<T> x) {
  if constexpr (std::same_as<T, eltype_t<A>>) return matrix(a, M, N, x);
  else return matrix(rebound_alloc<A, T>{}, M, N, x);
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

template <alloc::FreeAllocator A>
constexpr auto identity(A a, unsigned int M)
  -> ManagedArray<eltype_t<A>, SquareDims, PreAllocSquareStorage<eltype_t<A>>(),
                  A> {
  ManagedArray<eltype_t<A>, SquareDims, PreAllocSquareStorage<eltype_t<A>>(), A>
    B{M, eltype_t<A>{}, a};
  B.diag() << eltype_t<A>{1};
  return B;
}
template <class T, alloc::FreeAllocator A>
constexpr auto identity(A a, unsigned int M) {
  if constexpr (std::same_as<T, eltype_t<A>>) return identity(a, M);
  else return identity(rebound_alloc<A, T>{}, M);
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
static_assert(Alloc<alloc::Mallocator<int64_t>, int64_t>);
static_assert(Alloc<Arena<> *, int64_t>);
} // namespace poly::math
