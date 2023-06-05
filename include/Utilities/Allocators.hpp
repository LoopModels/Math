#pragma once

#include "Utilities/Invariant.hpp"
#include "Utilities/Valid.hpp"
#include <Containers/UnrolledList.hpp>
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <limits>
#include <memory>

#ifndef __has_feature      // Optional of course.
#define __has_feature(x) 0 // Compatibility with non-clang compilers.
#endif
// taken from LLVM, to avoid needing to include
/// \macro LLVM_ADDRESS_SANITIZER_BUILD
/// Whether LLVM itself is built with AddressSanitizer instrumentation.
#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
#define MATH_ADDRESS_SANITIZER_BUILD 1
#if __has_include(<sanitizer/asan_interface.h>)
#include <sanitizer/asan_interface.h>
#else
// These declarations exist to support ASan with MSVC. If MSVC eventually ships
// asan_interface.h in their headers, then we can remove this.
#ifdef __cplusplus
extern "C" {
#endif
void __asan_poison_memory_region(void const volatile *addr, size_t size);
void __asan_unpoison_memory_region(void const volatile *addr, size_t size);
#ifdef __cplusplus
} // extern "C"
#endif
#endif
#else
#define MATH_ADDRESS_SANITIZER_BUILD 0
#define __asan_poison_memory_region(p, size)
#define __asan_unpoison_memory_region(p, size)
#endif

#if __has_feature(memory_sanitizer)
#define MATH_MEMORY_SANITIZER_BUILD 1
#include <sanitizer/msan_interface.h>
#define MATH_NO_SANITIZE_MEMORY_ATTRIBUTE __attribute__((no_sanitize_memory))
#else
#define MATH_MEMORY_SANITIZER_BUILD 0
#define __msan_allocated_memory(p, size)
#define __msan_unpoison(p, size)
#define MATH_NO_SANITIZE_MEMORY_ATTRIBUTE
#endif

namespace poly::utils {

template <typename T> struct AllocResult {
  T *ptr;
  size_t size;
};
template <typename T> AllocResult(T *ptr, size_t size) -> AllocResult<T>;
template <class A, typename T>
constexpr void deallocate(A &&alloc, AllocResult<T> r) {
  alloc.deallocate(r.ptr, r.size);
}

template <size_t SlabSize = 16384, bool BumpUp = false,
          size_t MinAlignment = alignof(std::max_align_t)>
struct BumpAlloc {
  static_assert(std::has_single_bit(MinAlignment));

public:
  static constexpr bool BumpDown = !BumpUp;
  using value_type = void;
  [[gnu::returns_nonnull, gnu::alloc_size(2), gnu::alloc_align(3),
    gnu::malloc]] constexpr auto
  allocate(size_t Size, size_t Align) -> void * {
    if (Size > SlabSize / 2) {
      void *p = std::aligned_alloc(Align, Size);
      pushOldSlab(p);
#ifndef NDEBUG
      std::fill_n((char *)(p), Size, -1);
#endif
      return p;
    }
    auto p = (Align > MinAlignment) ? bumpAlloc(Size, Align) : bumpAlloc(Size);
    __asan_unpoison_memory_region(p, Size);
    __msan_allocated_memory(p, Size);
#ifndef NDEBUG
    if ((MinAlignment >= alignof(int64_t)) && ((Size & 7) == 0)) {
      std::fill_n(static_cast<std::int64_t *>(p), Size >> 3,
                  std::numeric_limits<std::int64_t>::min());
    } else std::fill_n((char *)(p), Size, -1);
#endif
    return p;
  }
  template <typename T>
  [[gnu::returns_nonnull, gnu::flatten]] constexpr auto allocate(size_t N = 1)
    -> T * {
    static_assert(std::is_trivially_destructible_v<T>,
                  "BumpAlloc only supports trivially destructible types.");
    return static_cast<T *>(allocate(N * sizeof(T), alignof(T)));
  }
  template <typename T, class... Args>
  [[gnu::returns_nonnull, gnu::flatten]] constexpr auto create(Args &&...args)
    -> T * {
    static_assert(std::is_trivially_destructible_v<T>,
                  "BumpAlloc only supports trivially destructible types.");
    return std::construct_at(static_cast<T *>(allocate(sizeof(T), alignof(T))),
                             std::forward<Args>(args)...);
  }
  static constexpr auto contains(void *P, void *p) -> bool { return P == p; }
  constexpr void deallocate(void *Ptr, size_t Size) {
    __asan_poison_memory_region(Ptr, Size);
    if constexpr (BumpUp) {
      if (Ptr + align(Size) == slab) slab = Ptr;
    } else if (Ptr == slab) {
      slab = (char *)slab + align(Size);
      return;
    }
#ifdef BUMP_ALLOC_TRY_FREE
    if (size_t numCSlabs = customSlabs.size()) {
      for (size_t i = 0; i < std::min<size_t>(8, customSlabs.size()); ++i) {
        if (contains(customSlabs[customSlabs.size() - 1 - i], Ptr)) {
          std::free(Ptr);
          if (i)
            std::swap(customSlabs[customSlabs.size() - 1],
                      customSlabs[customSlabs.size() - 1 - i]);
          customSlabs.pop_back();
          return;
        }
      }
    }
#endif
  }
  template <typename T> constexpr void deallocate(T *Ptr, size_t N = 1) {
    deallocate((void *)Ptr, N * sizeof(T));
  }
  constexpr auto tryReallocate(void *Ptr, size_t szOld, size_t szNew,
                               size_t Align) -> void * {
    invariant(std::has_single_bit(Align));
    Align = Align > MinAlignment ? Align : MinAlignment;
    if constexpr (BumpUp) {
      if (Ptr == slab - align(szOld)) {
        slab = Ptr + align(szNew);
        if (!outOfSlab()) {
          __asan_unpoison_memory_region((char *)Ptr + szOld, szNew - szOld);
          __msan_allocated_memory((char *)Ptr + szOld, szNew - szOld);
          return Ptr;
        }
      }
    } else if (Ptr == slab) {
      size_t extraSize = align(szNew - szOld, Align);
      slab = (char *)slab - extraSize;
      if (!outOfSlab()) {
        __asan_unpoison_memory_region(slab, extraSize);
        __msan_allocated_memory(SlabCur, extraSize);
        return slab;
      }
    }
    return nullptr;
  }
  template <typename T>
  constexpr auto tryReallocate(T *Ptr, size_t OldSize, size_t NewSize) -> T * {
    return static_cast<T *>(
      tryReallocate(Ptr, OldSize * sizeof(T), NewSize * sizeof(T), alignof(T)));
  }
  /// reallocate<ForOverwrite>(void *Ptr, size_t OldSize, size_t NewSize,
  /// size_t Align) Should be safe with OldSize == 0, as it checks before
  /// copying
  template <bool ForOverwrite = false>
  [[gnu::returns_nonnull, nodiscard]] constexpr auto
  reallocate(void *Ptr, size_t szOld, size_t szNew, size_t Align) -> void * {
    invariant(std::has_single_bit(Align));
    if (szOld >= szNew) return Ptr;
    if (Ptr) {
      if (void *p = tryReallocate(Ptr, szOld, szNew, Align)) {
        if constexpr ((BumpDown) & (!ForOverwrite))
          std::copy_n((char *)Ptr, szOld, (char *)p);
        return p;
      }
      Align = Align > MinAlignment ? Align : MinAlignment;
      if constexpr (BumpUp) {
        if (Ptr == slab - align(szOld)) {
          slab = Ptr + align(szNew);
          if (!outOfSlab()) {
            __asan_unpoison_memory_region((char *)Ptr + szOld, szNew - szOld);
            __msan_allocated_memory((char *)Ptr + szOld, szNew - szOld);
            return Ptr;
          }
        }
      } else if (Ptr == slab) {
        size_t extraSize = align(szNew - szOld, Align);
        slab = (char *)slab - extraSize;
        if (!outOfSlab()) {
          __asan_unpoison_memory_region(slab, extraSize);
          __msan_allocated_memory(SlabCur, extraSize);
          if constexpr (!ForOverwrite)
            std::copy_n((char *)Ptr, szOld, (char *)slab);
          return slab;
        }
      }
    }
    // we need to allocate new memory
    auto newPtr = allocate(szNew, Align);
    if (szOld && Ptr) {
      if constexpr (!ForOverwrite)
        std::copy_n((char *)Ptr, szOld, (char *)newPtr);
      deallocate(Ptr, szOld);
    }
    return newPtr;
  }
  constexpr void reset() {
    resetSlabs();
    initSlab(sEnd);
  }
  template <bool ForOverwrite = false, typename T>
  [[gnu::returns_nonnull, gnu::flatten, nodiscard]] constexpr auto
  reallocate(T *Ptr, size_t OldSize, size_t NewSize) -> T * {
    return static_cast<T *>(reallocate<ForOverwrite>(
      Ptr, OldSize * sizeof(T), NewSize * sizeof(T), alignof(T)));
  }
  constexpr BumpAlloc() {
    initSlab(std::aligned_alloc(MinAlignment, SlabSize));
  }
  constexpr BumpAlloc(BumpAlloc &&other) noexcept
    : slab{other.slab}, sEnd{other.sEnd}, slabs{other.slabs} {
    other.slab = nullptr;
    other.sEnd = nullptr;
    other.slabs = nullptr;
  }
  BumpAlloc(const BumpAlloc &) = delete;
  constexpr ~BumpAlloc() {
    // need a check because of move
    // should we even support moving?
    resetSlabs();
    if (sEnd) std::free(sEnd);
  }
  template <typename T, typename... Args>
  constexpr auto construct(Args &&...args) -> T * {
    auto *p = allocate(sizeof(T), alignof(T));
    return new (p) T(std::forward<Args>(args)...);
  }
  constexpr auto isPointInSlab(void *p) -> bool {
    if constexpr (BumpUp) return (p >= sEnd) && (p < (char *)sEnd + SlabSize);
    else return (p > sEnd) && ((char *)p <= ((char *)sEnd + SlabSize));
  }
  struct CheckPoint {
    constexpr CheckPoint(void *b, void *l) : p(b), e(l) {}
    constexpr auto isInSlab(void *send) const -> bool { return send == e; }
    void *const p;
    void *const e;
  };
  [[nodiscard]] constexpr auto checkpoint() -> CheckPoint {
    return {slab, sEnd};
  }
  constexpr void rollback(CheckPoint p) {
    if (p.isInSlab(sEnd)) {
#if MATH_ADDRESS_SANITIZER_BUILD
      if constexpr (BumpUp)
        __asan_poison_memory_region(p.p, (char *)slab - (char *)p.p);
      else __asan_poison_memory_region(slab, (char *)p.p - (char *)slab);
#endif
      slab = p.p;
    } else initSlab(sEnd);
  }
  /// RAII version of CheckPoint
  struct ScopeLifetime {
    constexpr ScopeLifetime(BumpAlloc &a) : alloc(a), p(a.checkpoint()) {}
    constexpr ~ScopeLifetime() { alloc.rollback(p); }

  private:
    BumpAlloc &alloc;
    CheckPoint p;
  };
  constexpr auto scope() -> ScopeLifetime { return *this; }

private:
  static constexpr auto align(size_t x) -> size_t {
    return (x + MinAlignment - 1) & ~(MinAlignment - 1);
  }
  static constexpr auto align(size_t x, size_t alignment) -> size_t {
    return (x + alignment - 1) & ~(alignment - 1);
  }
  static constexpr auto align(void *p) -> void * {
    uint64_t i = reinterpret_cast<uintptr_t>(p), j = i;
    if constexpr (BumpUp) i += MinAlignment - 1;
    i &= ~(MinAlignment - 1);
    return (char *)p + (i - j);
  }
  static constexpr auto align(void *p, size_t alignment) -> void * {
    invariant(std::has_single_bit(alignment));
    uintptr_t i = reinterpret_cast<uintptr_t>(p), j = i;
    if constexpr (BumpUp) i += alignment - 1;
    i &= ~(alignment - 1);
    return (char *)p + (i - j);
  }
  static constexpr auto bump(void *ptr, size_t N) -> void * {
    if constexpr (BumpUp) return (char *)ptr + N;
    else return (char *)ptr - N;
  }
  static constexpr auto outOfSlab(void *cur, void *lst) -> bool {
    if constexpr (BumpUp) return cur >= lst;
    else return cur < lst;
  }
  constexpr auto outOfSlab() -> bool { return outOfSlab(slab, getEnd()); }
  constexpr auto getEnd() -> void * {
    if constexpr (BumpUp) return static_cast<char *>(sEnd) + SlabSize;
    else return sEnd;
  }
  constexpr void initSlab(void *p) {
    __asan_poison_memory_region(p, SlabSize);
    if constexpr (!BumpUp) {
      slab = (char *)p + SlabSize;
      sEnd = p;
    } else sEnd = slab = p;
  }
  constexpr void newSlab() {
    void *old = sEnd;
    initSlab(std::aligned_alloc(MinAlignment, SlabSize));
    pushOldSlab(old);
  }
  /// stores a slab so we can free it later
  constexpr void pushOldSlab(void *old) {
    if (slabs && (!slabs->isFull())) slabs->pushHasCapacity(old);
    else slabs = create<containers::UList<void *>>(old, slabs);
  }
  // updates SlabCur and returns the allocated pointer
  [[gnu::returns_nonnull]] constexpr auto allocCore(size_t Size, size_t Align)
    -> void * {
#if MATH_ADDRESS_SANITIZER_BUILD
    slab = bump(slab, Align); // poisoned zone
#endif
    if constexpr (BumpUp) {
      slab = align(slab, Align);
      void *old = slab;
      slab = (char *)slab + align(Size);
      return old;
    } else {
      slab = align((char *)slab - Size, Align);
      return slab;
    }
  }
  // updates SlabCur and returns the allocated pointer
  [[gnu::returns_nonnull]] constexpr auto allocCore(size_t Size) -> void * {
    // we know we already have MinAlignment
    // and we need to preserve it.
    // Thus, we align `Size` and offset it.
    invariant((reinterpret_cast<size_t>(slab) % MinAlignment) == 0);
#if MATH_ADDRESS_SANITIZER_BUILD
    slab = bump(slab, MinAlignment); // poisoned zone
#endif
    if constexpr (BumpUp) {
      void *old = slab;
      slab = (char *)slab + align(Size);
      return old;
    } else {
      slab = (char *)slab - align(Size);
      return slab;
    }
  }
  //
  [[gnu::returns_nonnull]] constexpr auto bumpAlloc(size_t Size, size_t Align)
    -> void * {
    invariant(std::has_single_bit(Align));
    void *ret = allocCore(Size, Align);
    if (outOfSlab()) [[unlikely]] {
      newSlab();
      ret = allocCore(Size, Align);
    }
    return ret;
  }
  [[gnu::returns_nonnull]] constexpr auto bumpAlloc(size_t Size) -> void * {
    void *ret = allocCore(Size);
    if (outOfSlab()) [[unlikely]] {
      newSlab();
      ret = allocCore(Size);
    }
    return ret;
  }

  constexpr void resetSlabs() {
    if (slabs) slabs->forEachStack([](auto *s) { std::free(s); });
    slabs = ((void *)slabs == sEnd) ? slabs : nullptr;
  }

  void *slab{nullptr};
  void *sEnd{nullptr};
  containers::UList<void *> *slabs{nullptr};
};
static_assert(sizeof(BumpAlloc<>) == 24);
static_assert(!std::is_trivially_copyable_v<BumpAlloc<>>);
static_assert(!std::is_trivially_destructible_v<BumpAlloc<>>);
static_assert(
  std::same_as<std::allocator_traits<BumpAlloc<>>::size_type, size_t>);

// Alloc wrapper people can pass and store by value
// with a specific value type, so that it can act more like a
// `std::allocator`.
template <typename T, size_t SlabSize = 16384, bool BumpUp = false,
          size_t MinAlignment = alignof(std::max_align_t)>
class WBumpAlloc {
  using Alloc = BumpAlloc<SlabSize, BumpUp, MinAlignment>;
  [[no_unique_address]] NotNull<Alloc> A;

public:
  using value_type = T;
  template <typename U> struct rebind { // NOLINT(readability-identifier-naming)
    using other = WBumpAlloc<U, SlabSize, BumpUp, MinAlignment>;
  };
  constexpr WBumpAlloc(Alloc &alloc) : A(&alloc) {}
  constexpr WBumpAlloc(NotNull<Alloc> alloc) : A(alloc) {}
  constexpr WBumpAlloc(const WBumpAlloc &other) = default;
  template <typename U>
  constexpr WBumpAlloc(WBumpAlloc<U> other) : A(other.get_allocator()) {}
  [[nodiscard]] constexpr auto get_allocator() const -> NotNull<Alloc> {
    return A;
  }
  constexpr void deallocate(T *p, size_t n) { A->deallocate(p, n); }
  [[gnu::returns_nonnull]] constexpr auto allocate(size_t n) -> T * {
    return A->template allocate<T>(n);
  }
  constexpr auto checkpoint() -> typename Alloc::CheckPoint {
    return A->checkpoint();
  }
  constexpr void rollback(typename Alloc::CheckPoint p) { A->rollback(p); }
};
static_assert(std::same_as<
              std::allocator_traits<WBumpAlloc<int64_t *>>::size_type, size_t>);
static_assert(
  std::same_as<std::allocator_traits<WBumpAlloc<int64_t>>::pointer, int64_t *>);

static_assert(std::is_trivially_copyable_v<NotNull<BumpAlloc<>>>);
static_assert(std::is_trivially_copyable_v<WBumpAlloc<int64_t>>);

template <typename A>
concept Allocator = requires(A a) {
  typename A::value_type;
  { a.allocate(1) } -> std::same_as<typename std::allocator_traits<A>::pointer>;
  {
    a.deallocate(std::declval<typename std::allocator_traits<A>::pointer>(), 1)
  };
};
static_assert(Allocator<WBumpAlloc<int64_t>>);
static_assert(Allocator<std::allocator<int64_t>>);

struct NoCheckpoint {};

constexpr auto checkpoint(const auto &) { return NoCheckpoint{}; }
template <class T> constexpr auto checkpoint(WBumpAlloc<T> alloc) {
  return alloc.checkpoint();
}
constexpr auto checkpoint(BumpAlloc<> &alloc) { return alloc.checkpoint(); }

constexpr void rollback(const auto &, NoCheckpoint) {}
template <class T> constexpr void rollback(WBumpAlloc<T> alloc, auto p) {
  alloc.rollback(p);
}
constexpr void rollback(BumpAlloc<> &alloc, auto p) { alloc.rollback(p); }
} // namespace poly::utils
template <size_t SlabSize, bool BumpUp, size_t MinAlignment>
auto operator new(size_t Size,
                  poly::utils::BumpAlloc<SlabSize, BumpUp, MinAlignment> &Alloc)
  -> void * {
  return Alloc.allocate(Size, alignof(std::max_align_t));
}

template <size_t SlabSize, bool BumpUp, size_t MinAlignment>
void operator delete(void *,
                     poly::utils::BumpAlloc<SlabSize, BumpUp, MinAlignment> &) {
}
