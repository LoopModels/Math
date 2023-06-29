#pragma once

#include "Utilities/Invariant.hpp"
#include "Utilities/Valid.hpp"
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

/// An Arena allocator
/// API:
/// 1. Create an OwningArena; this governs the lifetime of the backing memory.
/// 2. To use the arena in a function, receive by
///   - a - value if the allocations internally are to die with the callee. This
///   allows acting like a stack with better support for larger and dynamic
///   allocations.
///   - b - pointer if the allocations may outlive the call.
/// In this way, the type signature indicates how the receiving function is
///  using the allocator. THe callsites also indicate whether they're passing by
///  pointer or value, helping one reason about lifetimes while reading code.
///
/// Use `allocate<T>` and `create<T>` methods for actually performing the
/// allocations.
template <size_t SlabSize = 16384, bool BumpUp = true> class Arena {
  static constexpr size_t Alignment = alignof(std::max_align_t);
  static constexpr auto align(size_t x) -> size_t {
    return (x + Alignment - 1) & ~(Alignment - 1);
  }
  // meta data is always in front, so we don't have to handle
  // the rare accidental case of huge slabs separately.
  static constexpr size_t MetaSize = align(2 * sizeof(void *));
  static_assert(std::has_single_bit(Alignment));

public:
  using value_type = void;
  constexpr Arena(const Arena &) = default;
  constexpr Arena(Arena &&) noexcept = default;
  constexpr auto operator=(const Arena &) -> Arena & = default;
#ifndef NDEBUG
  constexpr void fillWithJunk(void *p, size_t Size) {
    if (((Size & 7) == 0)) {
      std::fill_n(static_cast<std::int64_t *>(p), Size >> 3,
                  std::numeric_limits<std::int64_t>::min());
    } else std::fill_n((char *)(p), Size, -1);
  }
#endif
  [[using gnu: returns_nonnull, alloc_size(2), assume_aligned(Alignment),
    malloc]] constexpr auto
  allocate(size_t Size) -> void * {
    // we allocate this slab and insert it.
    // this is a pretty bad path.
    if (Size > SlabSize - MetaSize) [[unlikely]] {
      void *p = std::malloc(Size + MetaSize);
      void *q = std::malloc(SlabSize);
      void **meta = getMeta(p);
      void **oldMeta = getMeta(sEnd);
      void **newMeta = getMeta(q);
      void *firstSlab = oldMeta[1];
      oldMeta[0] = p;
      meta[0] = q;
      meta[1] = firstSlab;
      newMeta[0] = nullptr;
      newMeta[1] = firstSlab;
      initSlab(q);
#ifndef NDEBUG
      fillWithJunk(static_cast<char *>(p) + MetaSize, Size);
#endif
      return static_cast<char *>(p) + MetaSize;
    }
    auto p = bumpAlloc(Size);
    __asan_unpoison_memory_region(p, Size);
    __msan_allocated_memory(p, Size);
#ifndef NDEBUG
    fillWithJunk(p, Size);
#endif
    return p;
  }
  template <typename T>
  [[gnu::returns_nonnull, gnu::flatten]] constexpr auto allocate(size_t N = 1)
    -> T * {
    static_assert(std::is_trivially_destructible_v<T>,
                  "Arena only supports trivially destructible types.");
    return static_cast<T *>(allocate(N * sizeof(T)));
  }
  template <typename T, class... Args>
  [[gnu::returns_nonnull, gnu::flatten]] constexpr auto create(Args &&...args)
    -> T * {
    static_assert(std::is_trivially_destructible_v<T>,
                  "Arena only supports trivially destructible types.");
    return std::construct_at(static_cast<T *>(allocate(sizeof(T))),
                             std::forward<Args>(args)...);
  }
  constexpr void deallocate(void *Ptr, size_t Size) {
    __asan_poison_memory_region(Ptr, Size);
    if constexpr (BumpUp) {
      if ((char *)Ptr + align(Size) == slab) slab = Ptr;
    } else if (Ptr == slab) {
      slab = (char *)slab + align(Size);
      return;
    }
  }
  template <typename T> constexpr void deallocate(T *Ptr, size_t N = 1) {
    deallocate((void *)Ptr, N * sizeof(T));
  }

  /// reallocate<ForOverwrite>(void *Ptr, ptrdiff_t OldSize, ptrdiff_t NewSize,
  /// size_t Align) Should be safe with OldSize == 0, as it checks before
  /// copying
  template <bool ForOverwrite = false>
  [[gnu::returns_nonnull, nodiscard]] constexpr auto
  reallocateImpl(void *Ptr, size_t capOld, size_t capNew, size_t szOld)
    -> void * {
    if (capOld >= capNew) return Ptr;
    if (Ptr) {
      if (void *p = tryReallocate(Ptr, capOld, capNew)) {
        if constexpr ((!BumpUp) & (!ForOverwrite))
          std::copy_n((char *)Ptr, szOld, (char *)p);
        return p;
      }
      if constexpr (BumpUp) {
        if (Ptr == (char *)slab - align(capOld)) {
          slab = (char *)Ptr + align(capNew);
          if (!outOfSlab()) {
            __asan_unpoison_memory_region((char *)Ptr + capOld,
                                          capNew - capOld);
            __msan_allocated_memory((char *)Ptr + capOld, capNew - capOld);
            return Ptr;
          }
        }
      } else if (Ptr == slab) {
        size_t extraSize = align(capNew - capOld);
        slab = (char *)slab - extraSize;
        if (!outOfSlab()) {
          __asan_unpoison_memory_region(slab, extraSize);
          __msan_allocated_memory(SlabCur, extraSize);
          if constexpr (!ForOverwrite)
            std::copy_n((char *)Ptr, capOld, (char *)slab);
          return slab;
        }
      }
    }
    // we need to allocate new memory
    auto newPtr = allocate(capNew);
    if (szOld && Ptr) {
      if constexpr (!ForOverwrite)
        std::copy_n((char *)Ptr, szOld, (char *)newPtr);
      deallocate(Ptr, capOld);
    }
    return newPtr;
  }
  /// free all memory, but the Arena holds onto it.
  constexpr void reset() {
    initSlab(getFirst(sEnd));
#if MATH_ADDRESS_SANITIZER_BUILD
    for (void *p = sEnd; p; p = getNext(p))
      __asan_poison_memory_region((char *)p + MetaSize, SlabSize - MetaSize);
#endif
  }
  template <bool ForOverwrite = false, typename T>
  [[gnu::returns_nonnull, gnu::flatten, nodiscard]] constexpr auto
  reallocate(T *Ptr, size_t oldCapacity, size_t newCapacity) -> T * {
    return static_cast<T *>(reallocateImpl<ForOverwrite>(
      Ptr, oldCapacity * sizeof(T), newCapacity * sizeof(T),
      oldCapacity * sizeof(T)));
  }
  template <bool ForOverwrite = false, typename T>
  [[gnu::returns_nonnull, gnu::flatten, nodiscard]] constexpr auto
  reallocate(T *Ptr, size_t oldCapacity, size_t newCapacity, size_t oldSize)
    -> T * {
    return static_cast<T *>(reallocateImpl<ForOverwrite>(
      Ptr, oldCapacity * sizeof(T), newCapacity * sizeof(T),
      oldSize * sizeof(T)));
  }
  //   : slab{other.slab}, sEnd{other.sEnd} {
  //   other.slab = nullptr;
  //   other.sEnd = nullptr;
  // }

  template <typename T, typename... Args>
  constexpr auto construct(Args &&...args) -> T * {
    auto *p = allocate(sizeof(T));
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
#if MATH_ADDRESS_SANITIZER_BUILD
    // poison remainder of slab
    if constexpr (BumpUp)
      __asan_poison_memory_region(p.p, (char *)slab - (char *)p.p);
    else __asan_poison_memory_region(slab, (char *)p.p - (char *)slab);
    for (void *m = getNext(p.p); m; m = getNext(m))
      __asan_poison_memory_region((char *)m + MetaSize, SlabSize - MetaSize);
#endif
    slab = p.p;
    sEnd = p.e;
  }
  /// RAII version of CheckPoint
  class ScopeLifetime {
    Arena &alloc;
    CheckPoint p;

  public:
    constexpr ScopeLifetime(Arena &a) : alloc(a), p(a.checkpoint()) {}
    constexpr ~ScopeLifetime() { alloc.rollback(p); }
  };
  constexpr auto scope() -> ScopeLifetime { return *this; }

private:
  constexpr auto tryReallocate(void *Ptr, size_t capOld, size_t capNew)
    -> void * {
    if constexpr (BumpUp) {
      if (Ptr == (char *)slab - align(capOld)) {
        slab = (char *)Ptr + align(capNew);
        if (!outOfSlab()) {
          __asan_unpoison_memory_region((char *)Ptr + capOld, capNew - capOld);
          __msan_allocated_memory((char *)Ptr + capOld, capNew - capOld);
          return Ptr;
        }
      }
    } else if (Ptr == slab) {
      size_t extraSize = align(capNew - capOld);
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
      tryReallocate(Ptr, OldSize * sizeof(T), NewSize * sizeof(T)));
  }
  static constexpr auto bump(void *ptr, ptrdiff_t N) -> void * {
    if constexpr (BumpUp) return (char *)ptr + N;
    else return (char *)ptr - N;
  }
  static constexpr auto outOfSlab(void *current, void *last) -> bool {
    if constexpr (BumpUp) return current >= last;
    else return current < last;
  }
  constexpr auto outOfSlab() -> bool { return outOfSlab(slab, getEnd()); }
  constexpr auto getEnd() -> void * {
    if constexpr (BumpUp) return static_cast<char *>(sEnd) + SlabSize;
    else return static_cast<char *>(sEnd) + MetaSize;
  }
  static constexpr auto initValue(void *p) -> void * {
    if constexpr (BumpUp) return static_cast<char *>(p) + MetaSize;
    else return static_cast<char *>(p) + SlabSize;
  }
  constexpr void initSlab(void *p) {
    sEnd = p;
    slab = initValue(p);
  }
  constexpr void newSlab(void **oldMeta) {
    initNewSlab();
    void **newMeta = getMeta(sEnd);
    oldMeta[0] = sEnd;
    newMeta[0] = nullptr;
    newMeta[1] = oldMeta[1];
  }
  constexpr void nextSlab() {
    void **meta = getMeta(sEnd);
    if (void *p = meta[0]) initSlab(p);
    else newSlab(meta);
  }
  // updates SlabCur and returns the allocated pointer
  [[gnu::returns_nonnull]] constexpr auto allocCore(ptrdiff_t Size) -> void * {
    // we know we already have Alignment
    // and we need to preserve it.
    // Thus, we align `Size` and offset it.
    invariant((reinterpret_cast<ptrdiff_t>(slab) % Alignment) == 0);
#if MATH_ADDRESS_SANITIZER_BUILD
    slab = bump(slab, Alignment); // poisoned zone
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
  [[gnu::returns_nonnull]] constexpr auto bumpAlloc(ptrdiff_t Size) -> void * {
    void *ret = allocCore(Size);
    if (outOfSlab()) [[unlikely]] {
      nextSlab();
      ret = allocCore(Size);
    }
    return ret;
  }

  void *slab;

protected:
  void *sEnd;

  static constexpr auto getMeta(void *p) -> void ** {
    return static_cast<void **>(p);
  }
  static constexpr auto getFirst(void *p) -> void * { return getMeta(p)[1]; }
  static constexpr auto getNext(void *p) -> void * { return getMeta(p)[0]; }
  constexpr void initNewSlab() {
    void *p = std::malloc(SlabSize);
    initSlab(p);
    // poison everything except the meta data
#if MATH_ADDRESS_SANITIZER_BUILD
    __asan_poison_memory_region(static_cast<char *>(p) + MetaSize,
                                SlabSize - MetaSize);
#endif
  }
  // constexpr Arena() : slab(nullptr), sEnd(nullptr){};
  constexpr Arena() = default;
};
static_assert(sizeof(Arena<>) == 16);
static_assert(std::is_trivially_copyable_v<Arena<>>);
static_assert(std::is_trivially_destructible_v<Arena<>>);
static_assert(std::same_as<std::allocator_traits<Arena<>>::size_type, size_t>);

/// RAII type that allocates and deallocates.
template <size_t SlabSize = 16384, bool BumpUp = true>
class OwningArena : public Arena<SlabSize, BumpUp> {
public:
  constexpr explicit OwningArena() {
    this->initNewSlab();
    void **meta = this->getMeta(this->sEnd);
    meta[0] = nullptr;
    meta[1] = this->sEnd; // sEnd is first
  }

  OwningArena(OwningArena &&other) = delete;
  OwningArena(const OwningArena &) = delete;
  constexpr ~OwningArena() {
    void *p = this->getFirst(this->sEnd);
    while (p) {
      void *next = this->getNext(p); // load before free!
      std::free(p);
      p = next;
    }
  }
};
static_assert(sizeof(OwningArena<>) == 16);
static_assert(!std::is_trivially_copyable_v<OwningArena<>>);
static_assert(!std::is_trivially_destructible_v<OwningArena<>>);

// Alloc wrapper people can pass and store by value
// with a specific value type, so that it can act more like a
// `std::allocator`.
template <typename T, size_t SlabSize = 16384, bool BumpUp = true>
class WArena {
  using Alloc = Arena<SlabSize, BumpUp>;
  [[no_unique_address]] NotNull<Alloc> A;

public:
  using value_type = T;
  template <typename U> struct rebind { // NOLINT(readability-identifier-naming)
    using other = WArena<U, SlabSize, BumpUp>;
  };
  constexpr explicit WArena(Alloc &alloc) : A(&alloc) {}
  constexpr explicit WArena(NotNull<Alloc> alloc) : A(alloc) {}
  constexpr explicit WArena(const WArena &other) = default;
  template <typename U>
  constexpr explicit WArena(WArena<U> other) : A(other.get_allocator()) {}
  [[nodiscard]] constexpr auto get_allocator() const -> NotNull<Alloc> {
    return A;
  }
  constexpr void deallocate(T *p, ptrdiff_t n) { A->deallocate(p, n); }
  [[gnu::returns_nonnull]] constexpr auto allocate(ptrdiff_t n) -> T * {
    return A->template allocate<T>(n);
  }
  constexpr auto checkpoint() -> typename Alloc::CheckPoint {
    return A->checkpoint();
  }
  constexpr void rollback(typename Alloc::CheckPoint p) { A->rollback(p); }
};
template <typename T, size_t SlabSize, bool BumpUp>
constexpr auto wrap(NotNull<Arena<SlabSize, BumpUp>> a)
  -> WArena<T, SlabSize, BumpUp> {
  return WArena<T, SlabSize, BumpUp>(a);
}

static_assert(
  std::same_as<std::allocator_traits<WArena<int64_t *>>::size_type, size_t>);
static_assert(
  std::same_as<std::allocator_traits<WArena<int64_t>>::pointer, int64_t *>);

static_assert(std::is_trivially_copyable_v<NotNull<Arena<>>>);
static_assert(std::is_trivially_copyable_v<WArena<int64_t>>);

template <typename A>
concept Allocator = requires(A a) {
  typename A::value_type;
  { a.allocate(1) } -> std::same_as<typename std::allocator_traits<A>::pointer>;
  {
    a.deallocate(std::declval<typename std::allocator_traits<A>::pointer>(), 1)
  };
};
static_assert(Allocator<WArena<int64_t>>);
static_assert(Allocator<std::allocator<int64_t>>);

struct NoCheckpoint {};

constexpr auto checkpoint(const auto &) { return NoCheckpoint{}; }
template <class T> constexpr auto checkpoint(WArena<T> alloc) {
  return alloc.checkpoint();
}
constexpr auto checkpoint(Arena<> *alloc) -> Arena<>::CheckPoint {
  return alloc->checkpoint();
}

constexpr void rollback(const auto &, NoCheckpoint) {}
template <class T> constexpr void rollback(WArena<T> alloc, auto p) {
  alloc.rollback(p);
}
constexpr void rollback(Arena<> *alloc, auto p) { alloc->rollback(p); }
} // namespace poly::utils
template <size_t SlabSize, bool BumpUp>
auto operator new(size_t Size, poly::utils::Arena<SlabSize, BumpUp> &Alloc)
  -> void * {
  return Alloc.allocate(Size);
}

template <size_t SlabSize, bool BumpUp>
void operator delete(void *, poly::utils::Arena<SlabSize, BumpUp> &) {}
