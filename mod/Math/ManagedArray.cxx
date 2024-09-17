#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#include "Owner.hxx"
#ifndef USE_MODULE
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <ostream>
#include <type_traits>

#include "Alloc/Mallocator.cxx"
#include "Containers/Pair.cxx"
#include "Containers/Storage.cxx"
#include "Math/Array.cxx"
#include "Math/ArrayConcepts.cxx"
#include "Math/AxisTypes.cxx"
#include "Math/MatrixDimensions.cxx"
#include "Utilities/ArrayPrint.cxx"
#include "Utilities/TypeCompression.cxx"
#else
export module ManagedArray;
export import Array;
import Allocator;
import ArrayConcepts;
import ArrayPrint;
import AxisTypes;
import MatDim;
import Pair;
import Storage;
import STL;
import TypeCompression;
#endif

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif
#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
template <typename T>
using DefaultAlloc = std::allocator<utils::compressed_t<T>>;
#else
template <typename T>
using DefaultAlloc = alloc::Mallocator<utils::compressed_t<T>>;
#endif

/// Stores memory, then pointer.
/// Thus struct's alignment determines initial alignment
/// of the stack memory.
/// Information related to size is then grouped next to the pointer.
///
/// The Intel compiler + OpenMP appears to memcpy data around,
/// or at least build ManagedArrays bypassing the constructors listed here.
/// This caused invalid frees, as the pointer still pointed to the old
/// stack memory.
template <class T, Dimension S,
          ptrdiff_t StackStorage = containers::PreAllocStorage<T, S>(),
          alloc::FreeAllocator A = DefaultAlloc<T>>
struct MATH_GSL_OWNER ManagedArray : ResizeableView<T, S> {
  // static_assert(std::is_trivially_destructible_v<T>);
  using BaseT = ResizeableView<T, S>;
  using U = containers::default_capacity_type_t<S>;
  using storage_type = typename BaseT::storage_type;
  static constexpr bool trivialelt =
    std::is_trivially_default_constructible_v<T> &&
    std::is_trivially_move_constructible_v<T> &&
    std::is_trivially_destructible_v<T>;
  // We're deliberately not initializing storage.
#if !defined(__clang__) && defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wuninitialized"
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
#endif
  constexpr ManagedArray() noexcept : BaseT{S{}, capacity(StackStorage)} {
#ifndef NDEBUG
    this->ptr = memory_.data();
    if (!StackStorage) return;
    if constexpr (std::numeric_limits<T>::has_signaling_NaN)
      std::fill_n(this->data(), StackStorage,
                  std::numeric_limits<T>::signaling_NaN());
    else if constexpr (std::numeric_limits<T>::is_specialized)
      std::fill_n(this->data(), StackStorage, std::numeric_limits<T>::min());
#endif
  }
  // if `T` is trivial, contents are uninitialized
  // if non-trivial, they are default constructed.
  constexpr ManagedArray(S s) noexcept : BaseT{s, capacity(StackStorage)} {
    this->ptr = memory_.data();
    U len = U(capacity(ptrdiff_t(this->sz)));
    if (len > StackStorage) this->allocateAtLeast(len);
#ifndef NDEBUG
    if (!len) return;
    auto l = ptrdiff_t(len);
    if constexpr (std::numeric_limits<T>::has_signaling_NaN)
      std::fill_n(this->data(), l, std::numeric_limits<T>::signaling_NaN());
    else if constexpr (std::numeric_limits<T>::is_specialized)
      std::fill_n(this->data(), l, std::numeric_limits<T>::min());
#endif
    if constexpr (!trivialelt)
      std::uninitialized_default_construct_n(this->data(), len);
  }
  constexpr ManagedArray(S s, T x) noexcept : BaseT{s, capacity(StackStorage)} {
    this->ptr = memory_.data();
    auto len = ptrdiff_t(this->sz);
    if (len > StackStorage) this->allocateAtLeast(capacity(len));
    if (!len) return;
    if constexpr (trivialelt) std::fill_n(this->data(), len, x);
    else std::uninitialized_fill_n(this->data(), len, std::move(x));
  }
  constexpr ManagedArray(A) noexcept : ManagedArray() {};
  constexpr ManagedArray(S s, A) noexcept : ManagedArray(s) {};
  constexpr ManagedArray(ptrdiff_t s, A) noexcept
  requires(std::same_as<S, SquareDims<>>)
    : ManagedArray(SquareDims<>{row(s)}) {};
  constexpr ManagedArray(S s, T x, A) noexcept : ManagedArray(s, x) {};

  constexpr ManagedArray(T x) noexcept
  requires(std::same_as<S, Length<>>)
    : BaseT{S{}, capacity(StackStorage)} {
    this->ptr = memory_.data();
    if constexpr (StackStorage == 0) this->growUndef(1);
    this->push_back_within_capacity(std::move(x));
  }

  template <class D>
  constexpr ManagedArray(const ManagedArray<T, D, StackStorage, A> &b) noexcept
    : BaseT{S(b.dim()), capacity(StackStorage)} {
    this->ptr = memory_.data();
    auto len = ptrdiff_t(this->sz);
    this->growUndef(len);
    if constexpr (trivialelt) std::copy_n(b.data(), len, this->data());
    else std::uninitialized_copy_n(b.data(), len, this->data());
  }
  template <std::convertible_to<T> Y, class D, class AY>
  constexpr ManagedArray(const ManagedArray<Y, D, StackStorage, AY> &b) noexcept
    : BaseT{S{}, capacity(StackStorage)} {
    this->ptr = memory_.data();
    S d = b.dim();
    auto len = ptrdiff_t(d);
    this->growUndef(len);
    this->sz = d;
    if constexpr (trivialelt) std::copy_n(b.data(), len, this->data());
    else std::uninitialized_copy_n(b.data(), len, this->data());
  }
  template <std::convertible_to<T> Y, size_t M>
  constexpr ManagedArray(std::array<Y, M> il) noexcept
    : BaseT{S{}, capacity(StackStorage)} {
    this->ptr = memory_.data();
    auto len = ptrdiff_t(M);
    this->growUndef(len);
    if constexpr (trivialelt) std::copy_n(il.begin(), len, this->data());
    else std::uninitialized_copy_n(il.begin(), len, this->data());
    this->sz = math::length(ptrdiff_t(M));
  }
  template <std::convertible_to<T> Y, class D, class AY>
  constexpr ManagedArray(const ManagedArray<Y, D, StackStorage, AY> &b,
                         S s) noexcept
    : BaseT{S(s), capacity(StackStorage)} {
    this->ptr = memory_.data();
    auto len = ptrdiff_t(this->sz);
    invariant(len == U(b.size()));
    this->growUndef(len);
    T *p = this->data();
    if constexpr (trivialelt) std::copy_n(b.data(), len, this->data());
    else std::uninitialized_copy_n(b.data(), len, this->data());
  }
  constexpr ManagedArray(const ManagedArray &b) noexcept
    : BaseT{S(b.dim()), capacity(StackStorage)} {
    this->ptr = memory_.data();
    auto len = ptrdiff_t(this->sz);
    this->growUndef(len);
    if constexpr (trivialelt) std::copy_n(b.data(), len, this->data());
    else std::uninitialized_copy_n(b.data(), len, this->data());
  }
  constexpr ManagedArray(const Array<T, S> &b) noexcept
    : BaseT{S(b.dim()), capacity(StackStorage)} {
    this->ptr = memory_.data();
    auto len = ptrdiff_t(this->sz);
    this->growUndef(len);
    if constexpr (trivialelt) std::copy_n(b.data(), len, this->data());
    else std::uninitialized_copy_n(b.data(), len, this->data());
  }
  template <AbstractSimilar<S> V>
  constexpr ManagedArray(const V &b) noexcept
    : BaseT{S(shape(b)), capacity(StackStorage)} {
    this->ptr = memory_.data();
    this->growUndef(ptrdiff_t(this->sz));
    (*this) << b;
  }
  template <class D>
  constexpr ManagedArray(ManagedArray<T, D, StackStorage, A> &&b) noexcept
    : BaseT{b.dim(), U(capacity(StackStorage))} {
    if (!b.isSmall()) { // steal
      this->ptr = b.data();
      this->capacity_ = b.getCapacity();
    } else {
      this->ptr = memory_.data();
      if constexpr (trivialelt)
        std::copy_n(b.data(), ptrdiff_t(b.dim()), this->data());
      else
        std::uninitialized_copy_n(b.data(), ptrdiff_t(b.dim()), this->data());
    }
    b.resetNoFree();
  }
  constexpr ManagedArray(ManagedArray &&b) noexcept
    : BaseT{b.dim(), U(capacity(StackStorage))} {
    if constexpr (StackStorage) {
      if (!b.isSmall()) { // steal
        this->ptr = b.data();
        this->capacity_ = b.getCapacity();
      } else {
        this->ptr = memory_.data();
        if constexpr (trivialelt)
          std::copy_n(b.data(), ptrdiff_t(b.dim()), this->data());
        else
          std::uninitialized_copy_n(b.data(), ptrdiff_t(b.dim()), this->data());
      }
    } else {
      this->ptr = b.ptr;
      this->capacity_ = b.getCapacity();
    }
    b.resetNoFree();
  }
  template <class D>
  constexpr ManagedArray(ManagedArray<T, D, StackStorage, A> &&b, S s) noexcept
    : BaseT{s, U(capacity(StackStorage))} {
    if (!b.isSmall()) { // steal
      this->ptr = b.data();
      this->capacity_ = b.getCapacity();
    } else {
      this->ptr = memory_.data();
      if constexpr (trivialelt)
        std::copy_n(b.data(), ptrdiff_t(b.dim()), this->data());
      else
        std::uninitialized_copy_n(b.data(), ptrdiff_t(b.dim()), this->data());
    }
    b.resetNoFree();
  }
  constexpr ManagedArray(const ColVector auto &v)
  requires(MatrixDimension<S>)
    : BaseT{S(shape(v)), U(capacity(StackStorage))} {
    this->ptr = memory_.data();
    ptrdiff_t M = ptrdiff_t(this->sz);
    this->growUndef(M);
    if constexpr (!trivialelt)
      std::uninitialized_default_construct_n(this->data(), M);
    MutArray<T, decltype(v.dim())>(this->data(), v.dim()) << v;
  }
  constexpr ManagedArray(const RowVector auto &v)
  requires(MatrixDimension<S>)
    : BaseT{S(CartesianIndex(1, v.size())), U(capacity(StackStorage))} {
    this->ptr = memory_.data();
    ptrdiff_t M = ptrdiff_t(this->sz);
    this->growUndef(M);
    if constexpr (!trivialelt)
      std::uninitialized_default_construct_n(this->data(), M);
    MutArray<T, decltype(v.dim())>(this->data(), v.dim()) << v;
  }
#if !defined(__clang__) && defined(__GNUC__)
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
#else
#pragma clang diagnostic pop
#endif

  template <class D>
  constexpr auto operator=(
    const ManagedArray<T, D, StackStorage, A> &b) noexcept -> ManagedArray &
  requires(!std::same_as<S, D>)
  {
    // this condition implies `this->data() == nullptr`
    if (this->data() == b.data()) return *this;
    resizeCopyTo(b);
    return *this;
  }
  template <class D>
  constexpr auto
  operator=(ManagedArray<T, D, StackStorage, A> &&b) noexcept -> ManagedArray &
  requires(!std::same_as<S, D>)
  {
    // this condition implies `this->data() == nullptr`
    if (this->data() == b.data()) return *this;
    // here, we commandeer `b`'s memory
    S d = b.dim();
    // if `b` is small, we need to copy memory
    // no need to shrink our capacity
    if (b.isSmall()) std::copy_n(b.data(), ptrdiff_t(d), this->data());
    else this->maybeDeallocate(b.data(), ptrdiff_t(b.getCapacity()));
    b.resetNoFree();
    this->sz = d;
    return *this;
  }
  constexpr auto operator=(const ManagedArray &b) noexcept -> ManagedArray & {
    if (this == &b) return *this;
    resizeCopyTo(b);
    return *this;
  }
  constexpr auto operator=(ManagedArray &&b) noexcept -> ManagedArray & {
    if (this == &b) return *this;
    // here, we commandeer `b`'s memory
    S d = b.dim();
    if (b.isSmall()) std::copy_n(b.data(), ptrdiff_t(d), this->data());
    else this->maybeDeallocate(b.data(), ptrdiff_t(b.getCapacity()));
    b.resetNoFree();
    this->sz = d;
    return *this;
  }
  constexpr void resetNoFree() {
    this->ptr = memory_.data();
    this->sz = S{};
    this->capacity_ = capacity(StackStorage);
  }
  constexpr ~ManagedArray() noexcept { this->maybeDeallocate(); }

  [[nodiscard]] static constexpr auto identity(ptrdiff_t M) -> ManagedArray
  requires(MatrixDimension<S>)
  {
    ManagedArray B(SquareDims<>{row(M)}, T{0});
    B.diag() << 1;
    return B;
  }
  [[nodiscard]] static constexpr auto identity(Row<> R) -> ManagedArray
  requires(MatrixDimension<S>)
  {
    return identity(ptrdiff_t(R));
  }
  [[nodiscard]] static constexpr auto identity(Col<> C) -> ManagedArray
  requires(MatrixDimension<S>)
  {
    return identity(ptrdiff_t(C));
  }

  constexpr void reserveForGrow1()
  requires(std::same_as<S, Length<>>)
  {
    auto s = ptrdiff_t(this->sz), c = ptrdiff_t(this->capacity_);
    if (s == c) [[unlikely]]
      reserve(length(newCapacity(c)));
  }

  template <class... Args>
  constexpr auto emplace_back(Args &&...args) -> decltype(auto)
  requires(std::same_as<S, Length<>>)
  {
    reserveForGrow1();
    return this->emplace_back_within_capacity(args...);
  }
  constexpr void push_back(T value)
  requires(std::same_as<S, Length<>>)
  {
    reserveForGrow1();
    this->push_back_within_capacity(std::move(value));
  }
  constexpr auto insert(T *p, T x) -> T *requires(std::same_as<S, Length<>>) {
    auto s = ptrdiff_t(this->sz), c = ptrdiff_t(this->capacity_);
    if (s == c) [[unlikely]] {
      ptrdiff_t d = p - this->data();
      reserve(length(newCapacity(c)));
      p = this->data() + d;
    }
    invariant(s == ptrdiff_t(this->sz));
    return this->insert_within_capacity(p, std::move(x));
  }
  // behavior
  // if S is StridedDims, then we copy data.
  // If the new dims are larger in rows or cols, we fill with 0.
  // If the new dims are smaller in rows or cols, we truncate.
  // New memory outside of dims (i.e., stride larger), we leave uninitialized.
  //
  constexpr void resize(S nz) {
    reallocForSize(nz);
    this->sz = nz;
  }
  constexpr void resize(Row<> r)
  requires(std::same_as<S, Length<>> || MatrixDimension<S>)
  {
    if constexpr (std::same_as<S, Length<>>) return resize(S(r));
    else return resize(auto{this->sz}.set(r));
  }
  constexpr void resize(Col<> c)
  requires(std::same_as<S, Length<>> || MatrixDimension<S>)
  {
    if constexpr (std::same_as<S, Length<>>) return resize(S(c));
    else if constexpr (MatrixDimension<S>) return resize(auto{this->sz}.set(c));
  }
  constexpr void resizeForOverwrite(S M) {
    auto nz = ptrdiff_t(M);
    if constexpr (!trivialelt) {
      ptrdiff_t oz = ptrdiff_t(this->sz);
      storage_type *old_ptr = this->data();
      if (nz < oz) std::destroy_n(old_ptr + nz, oz - nz);
      else if (nz > ptrdiff_t(this->capacity_)) {
        auto [new_ptr, new_cap] = alloc::alloc_at_least(A{}, nz);
        invariant(ptrdiff_t(new_cap) >= nz);
        std::uninitialized_move_n(old_ptr, oz, new_ptr);
        std::uninitialized_default_construct_n(new_ptr + oz, nz - oz);
        maybeDeallocate();
        this->ptr = new_ptr;
        this->capacity_ = capacity(new_cap);
      } else if (nz > oz)
        std::uninitialized_default_construct_n(old_ptr + oz, nz - oz);
    } else if (nz > ptrdiff_t(this->sz)) growUndef(nz);
    this->sz = M;
  }
  constexpr void resizeForOverwrite(ptrdiff_t M)
  requires(std::same_as<S, Length<>>)
  {
    resizeForOverwrite(length(M));
  }
  constexpr void resize(ptrdiff_t M)
  requires(std::same_as<S, Length<>>)
  {
    resize(length(M));
  }
  constexpr void reserve(ptrdiff_t M)
  requires(std::same_as<S, Length<>>)
  {
    reserve(length(M));
  }
  constexpr void resizeForOverwrite(Row<> r) {
    if constexpr (std::same_as<S, Length<>>) {
      return resizeForOverwrite(S(r));
    } else if constexpr (MatrixDimension<S>) {
      S nz = this->sz;
      return resizeForOverwrite(nz.set(r));
    }
  }
  constexpr void resizeForOverwrite(Col<> c) {
    if constexpr (std::same_as<S, Length<>>) {
      return resizeForOverwrite(S(c));
    } else if constexpr (MatrixDimension<S>) {
      S nz = this->sz;
      return resizeForOverwrite(nz.set(c));
    }
  }
  static constexpr auto
  reserveCore(S nz, storage_type *op, ptrdiff_t old_len, U oc,
              bool was_allocated) -> containers::Pair<storage_type *, U> {
    auto new_capacity = ptrdiff_t(nz);
    invariant(new_capacity >= 0z);
    if (new_capacity <= oc) return {op, oc};
    // allocate new, copy, deallocate old
    auto [new_ptr, new_cap] = alloc::alloc_at_least(A{}, new_capacity);
    auto nc = ptrdiff_t(new_cap);
    invariant(nc >= new_capacity);
    if (old_len) {
      if constexpr (trivialelt)
        std::memcpy(new_ptr, op, old_len * sizeof(storage_type));
      else std::uninitialized_move_n(op, old_len, new_ptr);
    }
    maybeDeallocate(op, old_len, ptrdiff_t(oc), was_allocated);
    return {new_ptr, math::capacity(nc)};
  }
  constexpr void reserve(S nz) {
    auto [np, nc] = reserveCore(nz, this->data(), ptrdiff_t(this->sz),
                                this->capacity_, wasAllocated());
    this->ptr = np;
    this->capacity_ = nc;
  }
  [[nodiscard]] constexpr auto get_allocator() const noexcept -> A {
    return {};
  }
  // set size and 0.
  constexpr void setSize(Row<> r, Col<> c) {
    resizeForOverwrite({r, c});
    this->fill(0);
  }
  constexpr void resize(Row<> MM, Col<> NN) { resize(DenseDims{MM, NN}); }
  constexpr void reserve(Row<> M, Col<> N) {
    if constexpr (std::is_same_v<S, StridedDims<>>)
      reserve(StridedDims{M, N, max(N, RowStride(this->dim()))});
    else if constexpr (std::is_same_v<S, SquareDims<>>)
      reserve(SquareDims{row(std::max(ptrdiff_t(M), ptrdiff_t(N)))});
    else reserve(DenseDims{M, N});
  }
  constexpr void reserve(Row<> M, RowStride<> X) {
    if constexpr (std::is_same_v<S, StridedDims<>>)
      reserve(S{M, col(ptrdiff_t(X)), X});
    else if constexpr (std::is_same_v<S, SquareDims<>>)
      reserve(SquareDims{row(std::max(ptrdiff_t(M), ptrdiff_t(X)))});
    else reserve(S{M, col(ptrdiff_t(X))});
  }
  constexpr void clearReserve(Row<> M, Col<> N) {
    this->clear();
    reserve(M, N);
  }
  constexpr void clearReserve(Row<> M, RowStride<> X) {
    this->clear();
    reserve(M, X);
  }
  constexpr void resizeForOverwrite(Row<> M, Col<> N, RowStride<> X) {
    invariant(X >= N);
    if constexpr (std::is_same_v<S, StridedDims<>>)
      resizeForOverwrite(S{M, N, X});
    else if constexpr (std::is_same_v<S, SquareDims<>>) {
      invariant(ptrdiff_t(M) == ptrdiff_t(N));
      resizeForOverwrite(S{M});
    } else resizeForOverwrite(S{M, N});
  }
  constexpr void resizeForOverwrite(Row<> M, Col<> N) {
    if constexpr (std::is_same_v<S, StridedDims<>>)
      resizeForOverwrite(S{M, N, {ptrdiff_t(N)}});
    else if constexpr (std::is_same_v<S, SquareDims<>>) {
      invariant(ptrdiff_t(M) == ptrdiff_t(N));
      resizeForOverwrite(S{M});
    } else resizeForOverwrite(S{M, N});
  }

private:
  constexpr void allocateAtLeast(U len) {
    auto l = size_t(ptrdiff_t(len));
    alloc::AllocResult<storage_type> res = alloc::alloc_at_least(A{}, l);
    this->ptr = res.ptr;
    invariant(res.count >= l);
    this->capacity_ = capacity(res.count);
  }
  [[nodiscard]] constexpr auto isSmall() const -> bool {
    invariant(this->capacity_ >= StackStorage);
    return this->capacity_ == StackStorage;
  }
  [[nodiscard]] constexpr auto wasAllocated() const -> bool {
    return !isSmall();
  }
  // this method should only be called from the destructor
  // (and the implementation taking the new ptr and capacity)
  void maybeDeallocate() noexcept {
    if constexpr (StackStorage > 0)
      maybeDeallocate(this->data(), ptrdiff_t(this->sz),
                      ptrdiff_t(this->capacity_), wasAllocated());
    else
      maybeDeallocate(const_cast<storage_type *>(this->ptr),
                      ptrdiff_t(this->sz), ptrdiff_t(this->capacity_),
                      wasAllocated());
  }
  static void maybeDeallocate(storage_type *p, ptrdiff_t sz, ptrdiff_t cap,
                              bool was_allocated) noexcept {
    if constexpr (!std::is_trivially_destructible_v<storage_type>)
      std::destroy_n(p, sz);
    if (was_allocated) A{}.deallocate(p, cap);
  }
  // this method should be called whenever the buffer lives
  // NOTE: it is invalid to reassign `sz` before calling `maybeDeallocate`!
  void maybeDeallocate(storage_type *newPtr, ptrdiff_t newCapacity) noexcept {
    maybeDeallocate(this->data(), ptrdiff_t(this->sz),
                    ptrdiff_t(this->capacity_), wasAllocated());
    this->ptr = newPtr;
    this->capacity_ = capacity(newCapacity);
  }
  // reallocate, discarding old data
  // This only performs the allocation!!
  void growUndef(ptrdiff_t M) {
    invariant(M >= 0);
    if (M <= this->capacity_) return;
    maybeDeallocate();
    // because this doesn't care about the old data,
    // we can allocate after freeing, which may be faster
    this->ptr = A{}.allocate(M);
    this->capacity_ = capacity(M);
    // if constexpr (!trivialelt)
    //   std::uninitialized_default_construct_n(this->data(), M);
#ifndef NDEBUG
    if constexpr (std::numeric_limits<T>::has_signaling_NaN)
      std::fill_n(this->data(), M, std::numeric_limits<T>::signaling_NaN());
    else if constexpr (std::integral<T>)
      std::fill_n(this->data(), M, std::numeric_limits<T>::min());
#endif
  }
  constexpr void reallocForSize(S nz) {
    S oz = this->sz;
    if constexpr (std::same_as<S, Length<>>) {
      auto ozs = ptrdiff_t(oz), nzs = ptrdiff_t(nz);
      storage_type *old_ptr = this->data();
      if (nz <= oz) {
        if constexpr (!std::is_trivially_destructible_v<T>)
          if (nz < oz) std::destroy_n(old_ptr + nzs, ozs - nzs);
        return;
      }
      if (nz > this->capacity_) {
        auto ncu = size_t(nzs);
        auto [new_ptr, new_cap] = alloc::alloc_at_least(A{}, ncu);
        invariant(new_cap >= ncu);
        auto ncs = ptrdiff_t(new_cap);
        invariant(ncs >= nzs);
        if constexpr (trivialelt) {
          if (oz) std::copy_n(old_ptr, ozs, new_ptr);
          std::fill(new_ptr + ozs, new_ptr + nzs, T{});
        } else {
          if (oz) std::uninitialized_move_n(old_ptr, ozs, new_ptr);
          std::uninitialized_default_construct(new_ptr + ozs, new_ptr + nzs);
        }
        maybeDeallocate(new_ptr, ncs);
      } else if (nz > oz) {
        if constexpr (trivialelt) std::fill(old_ptr + ozs, old_ptr + nzs, T{});
        else std::uninitialized_default_construct(old_ptr + ozs, old_ptr + nzs);
      }
    } else {
      static_assert(std::is_trivially_destructible_v<T>,
                    "Resizing matrices holding non-is_trivially_destructible_v "
                    "objects is not yet supported.");
      static_assert(MatrixDimension<S>, "Can only resize 1 or 2d containers.");
      auto len = ptrdiff_t(nz);
      if (len == 0) return;
      auto new_x = ptrdiff_t{RowStride(nz)}, old_x = ptrdiff_t{RowStride(oz)},
           new_n = ptrdiff_t{Col(nz)}, old_n = ptrdiff_t{Col(oz)},
           new_m = ptrdiff_t{Row(nz)}, old_m = ptrdiff_t{Row(oz)};
      bool new_alloc = len > this->capacity_;
      bool in_place = !new_alloc;
      T *npt = this->data();
      if (new_alloc) {
        alloc::AllocResult<T> res = alloc::alloc_at_least(A{}, len);
        npt = res.ptr;
        len = res.count;
      }
      // we can copy forward so long as the new stride is smaller
      // so that the start of the dst range is outside of the src range
      // we can also safely forward copy if we allocated a new ptr
      bool forward_copy = (new_x <= old_x) || new_alloc;
      ptrdiff_t cols_to_copy = std::min(old_n, new_n);
      // we only need to copy if memory shifts position
      bool copy_cols = new_alloc || ((cols_to_copy > 0) && (new_x != old_x));
      // if we're in place, we have 1 less row to copy
      ptrdiff_t rows_to_copy = std::min(old_m, new_m);
      ptrdiff_t fill_count = new_n - cols_to_copy;
      if ((rows_to_copy) && (copy_cols || fill_count)) {
        if (forward_copy) {
          // truncation, we need to copy rows to increase stride
          T *src = this->data();
          T *dst = npt;
          do {
            if (copy_cols && (!in_place)) std::copy_n(src, cols_to_copy, dst);
            if (fill_count) std::fill_n(dst + cols_to_copy, fill_count, T{});
            src += old_x;
            dst += new_x;
            in_place = false;
          } while (--rows_to_copy);
        } else /* [[unlikely]] */ {
          // backwards copy, only needed when we increasing stride but not
          // reallocating, which should be comparatively uncommon.
          // Should probably benchmark or determine actual frequency
          // before adding `[[unlikely]]`.
          invariant(in_place);
          T *src = this->data() + (rows_to_copy + in_place) * old_x;
          T *dst = npt + (rows_to_copy + in_place) * new_x;
          do {
            src -= old_x;
            dst -= new_x;
            if (cols_to_copy && (rows_to_copy > in_place))
              std::copy_backward(src, src + cols_to_copy, dst + cols_to_copy);
            if (fill_count) std::fill_n(dst + cols_to_copy, fill_count, T{});
          } while (--rows_to_copy);
        }
      }
      // zero init remaining rows
      for (ptrdiff_t m = old_m; m < new_m; ++m)
        std::fill_n(npt + m * new_x, new_n, T{});
      if (new_alloc) maybeDeallocate(npt, len);
    }
  }

  // copies and resizes
  void resizeCopyTo(const auto &b) {
    S d = b.dim();
    auto len = ptrdiff_t(d);
    storage_type *old_ptr = this->data();
    const storage_type *bptr = b.data();
    if constexpr (trivialelt) {
      this->growUndef(len);
      std::copy_n(bptr, len, old_ptr);
    } else {
      ptrdiff_t oz = ptrdiff_t(this->sz), nz = ptrdiff_t(d);
      if (nz > ptrdiff_t(this->capacity_)) {
        auto [new_ptr, new_cap] = alloc::alloc_at_least(A{}, nz);
        invariant(new_cap >= nz);
        for (ptrdiff_t i = 0; i < oz; ++i)
          *std::construct_at(new_ptr + i, std::move(old_ptr[i])) = bptr[i];
        std::uninitialized_copy_n(bptr + oz, nz - oz, new_ptr);
        maybeDeallocate();
        this->ptr = new_ptr;
        this->capacity_ = capacity(new_cap);
      } else {
        std::copy_n(bptr, std::min(nz, oz), old_ptr);
        if (nz < oz) std::destroy_n(old_ptr + nz, oz - nz);
        else std::uninitialized_copy_n(bptr + oz, nz - oz, old_ptr);
      }
    }
    this->sz = d;
  }

  friend void PrintTo(const ManagedArray &x, ::std::ostream *os)
  requires(utils::Printable<T>)
  {
    *os << x;
  }

  [[no_unique_address]] containers::Storage<storage_type, StackStorage> memory_;
};

static_assert(std::move_constructible<ManagedArray<int64_t, Length<>>>);
static_assert(std::copyable<ManagedArray<int64_t, Length<>>>);
// Check that `[[no_unique_address]]` is working.
// sizes should be:
// [ptr, dims, capacity, array]
// 8 + 3*4 + 4 + 0 + 64*8 = 24 + 512 = 536
static_assert(sizeof(ManagedArray<int64_t, StridedDims<>, 64,
                                  alloc::Mallocator<int64_t>>) == 552);
// sizes should be:
// [ptr, dims, capacity, array]
// 8 + 2*4 + 8 + 0 + 64*8 = 24 + 512 = 536
static_assert(
  sizeof(ManagedArray<int64_t, DenseDims<>, 64, alloc::Mallocator<int64_t>>) ==
  544);
// sizes should be:
// [ptr, dims, capacity, array]
// 8 + 1*4 + 4 + 0 + 64*8 = 16 + 512 = 528
static_assert(
  sizeof(ManagedArray<int64_t, SquareDims<>, 64, alloc::Mallocator<int64_t>>) ==
  536);

template <class T, ptrdiff_t N = containers::PreAllocStorage<T, Length<>>()>
using Vector = ManagedArray<T, Length<>, N>;

template <class T,
          ptrdiff_t L = containers::PreAllocStorage<T, StridedDims<>>()>
using Matrix = ManagedArray<T, StridedDims<>, L>;
template <class T, ptrdiff_t L = containers::PreAllocStorage<T, DenseDims<>>()>
using DenseMatrix = ManagedArray<T, DenseDims<>, L>;
template <class T, ptrdiff_t L = containers::PreAllocStorage<T, SquareDims<>>()>
using SquareMatrix = ManagedArray<T, SquareDims<>, L>;

// type def defined by allocator
template <alloc::FreeAllocator A,
          ptrdiff_t L =
            containers::PreAllocStorage<utils::eltype_t<A>, SquareDims<>>()>
using SquareMatrixAlloc = ManagedArray<utils::eltype_t<A>, SquareDims<>, L, A>;
template <alloc::FreeAllocator A, ptrdiff_t R, ptrdiff_t C,
          ptrdiff_t L =
            containers::PreAllocStorage<utils::eltype_t<A>, DenseDims<>>()>
using DenseMatrixAlloc =
  ManagedArray<utils::eltype_t<A>, DenseDims<R, C>, L, A>;
template <alloc::FreeAllocator A,
          ptrdiff_t L =
            containers::PreAllocStorage<utils::eltype_t<A>, StridedDims<>>()>
using StrideMatrixAlloc = ManagedArray<utils::eltype_t<A>, StridedDims<>, L, A>;

template <VectorDimension S = ptrdiff_t>
using IntVector = ManagedArray<int64_t, S>;
template <MatrixDimension S = DenseDims<>>
using IntMatrix = ManagedArray<int64_t, S>;

static_assert(sizeof(ManagedArray<int32_t, DenseDims<3, 5>, 15>) ==
              sizeof(int32_t *) + 16 * sizeof(int32_t));
static_assert(sizeof(ManagedArray<int32_t, DenseDims<>, 15>) ==
              sizeof(int32_t *) + 3 * sizeof(ptrdiff_t) + 16 * sizeof(int32_t));
static_assert(AbstractMatrix<ManagedArray<int64_t, SquareDims<>>>);
static_assert(std::is_convertible_v<DenseMatrix<int64_t>, Matrix<int64_t>>);
static_assert(
  std::is_convertible_v<DenseMatrix<int64_t>, DensePtrMatrix<int64_t>>);
static_assert(std::is_convertible_v<DenseMatrix<int64_t>, PtrMatrix<int64_t>>);
static_assert(std::is_convertible_v<SquareMatrix<int64_t>, Matrix<int64_t>>);
static_assert(
  std::is_convertible_v<SquareMatrix<int64_t>, MutPtrMatrix<int64_t>>);
static_assert(std::same_as<IntMatrix<>::value_type, int64_t>);
static_assert(AbstractMatrix<IntMatrix<>>);
static_assert(std::copyable<IntMatrix<>>);
static_assert(std::move_constructible<Vector<int64_t>>);
static_assert(std::copyable<Vector<int64_t>>);
static_assert(AbstractVector<Vector<int64_t>>);
static_assert(!std::is_trivially_copyable_v<Vector<int64_t>>);
static_assert(!std::is_trivially_destructible_v<Vector<int64_t>>);
static_assert(AbstractVector<Vector<int64_t>>,
              "PtrVector<int64_t> isa AbstractVector failed");
static_assert(std::same_as<utils::eltype_t<Matrix<int64_t>>, int64_t>);

template <AbstractMatrix T>
inline auto operator<<(std::ostream &os, const T &A) -> std::ostream & {
  Matrix<std::remove_const_t<typename T::value_type>> B{A};
  return utils::printMatrix(os, B.data(), ptrdiff_t(B.numRow()),
                            ptrdiff_t(B.numCol()), ptrdiff_t(B.rowStride()));
}

} // namespace math
