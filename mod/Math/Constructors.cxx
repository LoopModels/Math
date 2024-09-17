#ifdef USE_MODULE
module;
#else
#pragma once
#endif
#ifndef USE_MODULE
#include <cstddef>
#include <cstring>
#include <type_traits>

#include "Alloc/Arena.cxx"
#include "Math/Array.cxx"
#include "Math/AxisTypes.cxx"
#include "Math/MatrixDimensions.cxx"
#else
export module ArrayConstructors;

import Arena;
import Array;
import AxisTypes;
import MatDim;
import STL;
#endif

using utils::eltype_t;
#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif

template <class T>
constexpr auto vector(alloc::WArena<T> alloc,
                      ptrdiff_t M) -> ResizeableView<T, Length<>> {
  return {alloc.allocate(M), length(M), capacity(M)};
}
template <class T, size_t SlabSize, bool BumpUp>
constexpr auto vector(alloc::Arena<SlabSize, BumpUp> *alloc,
                      ptrdiff_t M) -> ResizeableView<T, Length<>> {
  return {alloc->template allocate<compressed_t<T>>(M), length(M), capacity(M)};
}

template <class T>
constexpr auto vector(alloc::WArena<T> alloc, ptrdiff_t M,
                      T x) -> ResizeableView<T, Length<>> {
  ResizeableView<T, Length<>> a{alloc.allocate(M), length(M), capacity(M)};
  a.fill(x);
  return a;
}
template <class T, size_t SlabSize, bool BumpUp>
constexpr auto vector(alloc::Arena<SlabSize, BumpUp> *alloc, ptrdiff_t M,
                      T x) -> ResizeableView<T, Length<>> {
  ResizeableView<T, Length<>> a{alloc->template allocate<compressed_t<T>>(M),
                                length(M), capacity(M)};
  a.fill(x);
  return a;
}

template <class T>
constexpr auto square_matrix(alloc::WArena<T> alloc,
                             ptrdiff_t M) -> MutSquarePtrMatrix<T> {
  return {alloc.allocate(M * M), SquareDims<>{row(M)}};
}
template <class T, size_t SlabSize, bool BumpUp>
constexpr auto square_matrix(alloc::Arena<SlabSize, BumpUp> *alloc,
                             ptrdiff_t M) -> MutSquarePtrMatrix<T> {
  return {alloc->template allocate<compressed_t<T>>(M * M),
          SquareDims<>{row(M)}};
}
template <class T>
constexpr auto square_matrix(alloc::WArena<T> alloc, ptrdiff_t M,
                             T x) -> MutSquarePtrMatrix<T> {
  MutSquarePtrMatrix<T> A{alloc.allocate(M * M), SquareDims<>{row(M)}};
  A.fill(x);
  return A;
}
template <class T, size_t SlabSize, bool BumpUp>
constexpr auto square_matrix(alloc::Arena<SlabSize, BumpUp> *alloc, ptrdiff_t M,
                             T x) -> MutSquarePtrMatrix<T> {
  MutSquarePtrMatrix<T> A{alloc->template allocate<compressed_t<T>>(M * M),
                          SquareDims<>{row(M)}};
  A.fill(x);
  return A;
}

template <class T, ptrdiff_t R, ptrdiff_t C>
constexpr auto matrix(alloc::WArena<T> alloc, Row<R> M,
                      Col<C> N) -> MutArray<T, DenseDims<R, C>> {
  auto memamt = size_t(ptrdiff_t(M) * ptrdiff_t(N));
  return {alloc.allocate(memamt), DenseDims{M, N}};
}
template <class T, size_t SlabSize, bool BumpUp, ptrdiff_t R, ptrdiff_t C>
constexpr auto matrix(alloc::Arena<SlabSize, BumpUp> *alloc, Row<R> M,
                      Col<C> N) -> MutArray<T, DenseDims<R, C>> {
  auto memamt = size_t(ptrdiff_t(M) * ptrdiff_t(N));
  return {alloc->template allocate<compressed_t<T>>(memamt), M, N};
}
template <class T, size_t SlabSize, bool BumpUp>
constexpr auto
matrix(alloc::Arena<SlabSize, BumpUp> *alloc,
       CartesianIndex<ptrdiff_t, ptrdiff_t> dim) -> MutArray<T, DenseDims<>> {
  Row<> M = Row<>(dim);
  Col<> N = Col<>(dim);
  auto memamt = size_t(ptrdiff_t(M) * ptrdiff_t(N));
  return {alloc->template allocate<compressed_t<T>>(memamt), M, N};
}

template <class T, ptrdiff_t R, ptrdiff_t C>
constexpr auto matrix(alloc::WArena<T> alloc, Row<R> M, Col<C> N,
                      T x) -> MutArray<T, DenseDims<R, C>> {
  auto memamt = size_t(ptrdiff_t(M) * ptrdiff_t(N));
  MutArray<T, DenseDims<R, C>> A{alloc.allocate(memamt), DenseDims{M, N}};
  A.fill(x);
  return A;
}
template <class T, size_t SlabSize, bool BumpUp, ptrdiff_t R, ptrdiff_t C>
constexpr auto matrix(alloc::Arena<SlabSize, BumpUp> *alloc, Row<R> M, Col<C> N,
                      T x) -> MutArray<T, DenseDims<R, C>> {
  auto memamt = size_t(ptrdiff_t(M) * ptrdiff_t(N));
  MutArray<T, DenseDims<R, C>> A{
    alloc->template allocate<compressed_t<T>>(memamt), DenseDims{M, N}};
  A.fill(x);
  return A;
}
template <class T, size_t SlabSize, bool BumpUp, ptrdiff_t R, ptrdiff_t C>
constexpr auto matrix(alloc::Arena<SlabSize, BumpUp> *alloc, Row<R> M, Col<C> N,
                      std::false_type) -> MutArray<T, DenseDims<R, C>> {
  auto memamt = size_t(ptrdiff_t(M) * ptrdiff_t(N));
  MutArray<T, DenseDims<R, C>> A{
    alloc->template allocate<compressed_t<T>>(memamt), DenseDims{M, N}};
  std::memset(A.data(), 0, memamt);
  return A;
}

template <class T>
constexpr auto identity(alloc::WArena<T> alloc,
                        ptrdiff_t M) -> MutSquarePtrMatrix<T> {
  MutSquarePtrMatrix<T> A{square_matrix(alloc, M, T{})};
  A.diag() << T{1};
  return A;
}
template <class T, size_t SlabSize, bool BumpUp>
constexpr auto identity(alloc::Arena<SlabSize, BumpUp> *alloc,
                        ptrdiff_t M) -> MutSquarePtrMatrix<T> {
  MutSquarePtrMatrix<T> A{square_matrix(alloc, M, T{})};
  A.diag() << T{1};
  return A;
}

} // namespace math
