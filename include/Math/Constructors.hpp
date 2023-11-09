#pragma once

#include "Alloc/Arena.hpp"
#include "Math/Array.hpp"
#include "Math/MatrixDimensions.hpp"

namespace poly::math {
using alloc::Arena, alloc::WArena, alloc::OwningArena, utils::eltype_t;
template <alloc::FreeAllocator A>
constexpr auto vector(A a, ptrdiff_t M)
  -> ManagedArray<eltype_t<A>, ptrdiff_t,
                  containers::PreAllocStorage<eltype_t<A>, ptrdiff_t>(), A> {
  return {M, a};
}

template <alloc::Allocator A, typename T>
using rebound_alloc =
  typename std::allocator_traits<A>::template rebind_alloc<T>;

template <class T, alloc::FreeAllocator A>
constexpr auto vector(A a, ptrdiff_t M) {
  if constexpr (std::same_as<T, eltype_t<A>>) return vector(a, M);
  else return vector(rebound_alloc<A, T>{}, M);
}
template <class T>
constexpr auto vector(WArena<T> alloc, ptrdiff_t M)
  -> ResizeableView<T, ptrdiff_t> {
  return {alloc.allocate(M), M, M};
}
template <class T, size_t SlabSize, bool BumpUp>
constexpr auto vector(Arena<SlabSize, BumpUp> *alloc, ptrdiff_t M)
  -> ResizeableView<T, ptrdiff_t> {
  return {alloc->template allocate<T>(M), M, M};
}

template <alloc::FreeAllocator A>
constexpr auto vector(A a, ptrdiff_t M, eltype_t<A> x)
  -> ManagedArray<eltype_t<A>, ptrdiff_t,
                  containers::PreAllocStorage<eltype_t<A>, ptrdiff_t>(), A> {
  return {M, x, a};
}

template <typename T, alloc::FreeAllocator A>
constexpr auto vector(A a, ptrdiff_t M, std::type_identity_t<T> x) {
  if constexpr (std::same_as<T, eltype_t<A>>) return vector(a, M, x);
  else return vector(rebound_alloc<A, T>{}, M, x);
}

template <class T>
constexpr auto vector(WArena<T> alloc, ptrdiff_t M, T x)
  -> ResizeableView<T, ptrdiff_t> {
  ResizeableView<T, ptrdiff_t> a{alloc.allocate(M), M, M};
  a.fill(x);
  return a;
}
template <class T, size_t SlabSize, bool BumpUp>
constexpr auto vector(Arena<SlabSize, BumpUp> *alloc, ptrdiff_t M, T x)
  -> ResizeableView<T, ptrdiff_t> {
  ResizeableView<T, ptrdiff_t> a{alloc->template allocate<T>(M), M, M};
  a.fill(x);
  return a;
}

template <alloc::FreeAllocator A>
constexpr auto matrix(A a, ptrdiff_t M) -> ManagedArray<
  eltype_t<A>, SquareDims<>,
  containers::PreAllocSquareStorage<eltype_t<A>, SquareDims<>>(), A> {
  return {SquareDims<>{{M}}, a};
}

template <class T, alloc::FreeAllocator A>
constexpr auto matrix(A a, ptrdiff_t M) {
  if constexpr (std::same_as<T, eltype_t<A>>) return matrix(a, M);
  else return matrix(rebound_alloc<A, T>{}, M);
}

template <class T>
constexpr auto matrix(WArena<T> alloc, ptrdiff_t M) -> MutSquarePtrMatrix<T> {
  return {alloc.allocate(M * M), SquareDims<>{{M}}};
}
template <class T, size_t SlabSize, bool BumpUp>
constexpr auto matrix(Arena<SlabSize, BumpUp> *alloc, ptrdiff_t M)
  -> MutSquarePtrMatrix<T> {
  return {alloc->template allocate<T>(M * M), SquareDims<>{{M}}};
}
template <alloc::FreeAllocator A>
constexpr auto matrix(A a, ptrdiff_t M, eltype_t<A> x) -> ManagedArray<
  eltype_t<A>, SquareDims<>,
  containers::PreAllocSquareStorage<eltype_t<A>, SquareDims<>>(), A> {
  return {SquareDims<>{{M}}, x, a};
}
template <class T, alloc::FreeAllocator A>
constexpr auto matrix(A a, ptrdiff_t M, std::type_identity_t<T> x) {
  if constexpr (std::same_as<T, eltype_t<A>>) return matrix(a, M, x);
  else return matrix(rebound_alloc<A, T>{}, M, x);
}
template <class T>
constexpr auto matrix(WArena<T> alloc, ptrdiff_t M, T x)
  -> MutSquarePtrMatrix<T> {
  MutSquarePtrMatrix<T> A{alloc.allocate(M * M), SquareDims<>{{M}}};
  A.fill(x);
  return A;
}
template <class T, size_t SlabSize, bool BumpUp>
constexpr auto matrix(Arena<SlabSize, BumpUp> *alloc, ptrdiff_t M, T x)
  -> MutSquarePtrMatrix<T> {
  MutSquarePtrMatrix<T> A{alloc->template allocate<T>(M * M),
                          SquareDims<>{{M}}};
  A.fill(x);
  return A;
}

template <alloc::FreeAllocator A, ptrdiff_t R, ptrdiff_t C>
constexpr auto matrix(A a, Row<R> M, Col<C> N)
  -> ManagedArray<eltype_t<A>, DenseDims<R, C>,
                  containers::PreAllocStorage<eltype_t<A>, DenseDims<R, C>>(),
                  A> {
  return {DenseDims{M, N}, a};
}
template <class T, alloc::FreeAllocator A, ptrdiff_t R, ptrdiff_t C>
constexpr auto matrix(A a, Row<R> M, Col<C> N) {
  if constexpr (std::same_as<T, eltype_t<A>>) return matrix(a, M, N);
  else return matrix(rebound_alloc<A, T>{}, M, N);
}
template <class T, ptrdiff_t R, ptrdiff_t C>
constexpr auto matrix(WArena<T> alloc, Row<R> M, Col<C> N)
  -> MutArray<T, DenseDims<R, C>> {
  auto memamt = size_t(ptrdiff_t(M) * ptrdiff_t(N));
  return {alloc.allocate(memamt), DenseDims{M, N}};
}
template <class T, size_t SlabSize, bool BumpUp, ptrdiff_t R, ptrdiff_t C>
constexpr auto matrix(Arena<SlabSize, BumpUp> *alloc, Row<R> M, Col<C> N)
  -> MutDensePtrMatrix<T> {
  auto memamt = size_t(ptrdiff_t(M) * ptrdiff_t(N));
  return {alloc->template allocate<T>(memamt), M, N};
}
template <alloc::FreeAllocator A, ptrdiff_t R, ptrdiff_t C>
constexpr auto matrix(A a, Row<R> M, Col<C> N, eltype_t<A> x)
  -> ManagedArray<eltype_t<A>, DenseDims<R, C>,
                  containers::PreAllocStorage<eltype_t<A>, DenseDims<R, C>>(),
                  A> {
  return {DenseDims{M, N}, x, a};
}
template <class T, alloc::FreeAllocator A, ptrdiff_t R, ptrdiff_t C>
constexpr auto matrix(A a, Row<R> M, Col<C> N, std::type_identity_t<T> x) {
  if constexpr (std::same_as<T, eltype_t<A>>) return matrix(a, M, N, x);
  else return matrix(rebound_alloc<A, T>{}, M, N, x);
}
template <class T, ptrdiff_t R, ptrdiff_t C>
constexpr auto matrix(WArena<T> alloc, Row<R> M, Col<C> N, T x)
  -> MutArray<T, DenseDims<R, C>> {
  auto memamt = size_t(ptrdiff_t(M) * ptrdiff_t(N));
  MutDensePtrMatrix<T> A{alloc.allocate(memamt), DenseDims{M, N}};
  A.fill(x);
  return A;
}
template <class T, size_t SlabSize, bool BumpUp, ptrdiff_t R, ptrdiff_t C>
constexpr auto matrix(Arena<SlabSize, BumpUp> *alloc, Row<R> M, Col<C> N, T x)
  -> MutArray<T, DenseDims<R, C>> {
  auto memamt = size_t(ptrdiff_t(M) * ptrdiff_t(N));
  MutDensePtrMatrix<T> A{alloc->template allocate<T>(memamt), DenseDims{M, N}};
  A.fill(x);
  return A;
}

template <alloc::FreeAllocator A>
constexpr auto identity(A a, ptrdiff_t M) -> ManagedArray<
  eltype_t<A>, SquareDims<>,
  containers::PreAllocSquareStorage<eltype_t<A>, SquareDims<>>(), A> {
  ManagedArray<eltype_t<A>, SquareDims<>,
               containers::PreAllocSquareStorage<eltype_t<A>, SquareDims<>>(),
               A>
    B{SquareDims<>{Row<>{M}}, eltype_t<A>{}, a};
  B.diag() << eltype_t<A>{1};
  return B;
}
template <class T, alloc::FreeAllocator A>
constexpr auto identity(A a, ptrdiff_t M) {
  if constexpr (std::same_as<T, eltype_t<A>>) return identity(a, M);
  else return identity(rebound_alloc<A, T>{}, M);
}
template <class T>
constexpr auto identity(WArena<T> alloc, ptrdiff_t M) -> MutSquarePtrMatrix<T> {
  MutSquarePtrMatrix<T> A{matrix(alloc, M, T{})};
  A.diag() << T{1};
  return A;
}
template <class T, size_t SlabSize, bool BumpUp>
constexpr auto identity(Arena<SlabSize, BumpUp> *alloc, ptrdiff_t M)
  -> MutSquarePtrMatrix<T> {
  MutSquarePtrMatrix<T> A{matrix(alloc, M, T{})};
  A.diag() << T{1};
  return A;
}

template <typename T, typename I>
concept Alloc = requires(T t, ptrdiff_t M, Row<> r, Col<> c, I i) {
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
