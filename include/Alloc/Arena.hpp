#pragma once

#include "Alloc/Mallocator.hpp"
#include "Utilities/Invariant.hpp"
#include "Utilities/Valid.hpp"
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <utility>
#include <version>

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
extern "C" {
void __asan_poison_memory_region(void const volatile *addr, size_t size);
void __asan_unpoison_memory_region(void const volatile *addr, size_t size);
} // extern "C"
#endif
#else
#define MATH_ADDRESS_SANITIZER_BUILD 0
#endif

#if __has_feature(memory_sanitizer)
#define MATH_MEMORY_SANITIZER_BUILD 1
#include <sanitizer/msan_interface.h>
#define MATH_NO_SANITIZE_MEMORY_ATTRIBUTE __attribute__((no_sanitize_memory))
#else
#define MATH_MEMORY_SANITIZER_BUILD 0
#define MATH_NO_SANITIZE_MEMORY_ATTRIBUTE
#endif

namespace poly::alloc {

using utils::invariant, utils::Valid;

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
/// Contains two fields: slab, sEnd.
/// Optionally bumps either up or down.
/// Memory layout when `BumpUp == true`:
/// [end, first slab], region, [next slab, first slab]
/// Here, first/next slab point to the start, while `sEnd` points to the end.
/// Memory layout when `BumpUp == false`:
/// [next slab, first slab], region, [start, first slab]
/// Here, first/next slab point to the end, while `sEnd` points to the start.
/// In both cases, the start/end references are to the region
/// So, meta is indexed with negative/positive indices depending on size.
template <size_t SlabSize = 16384, bool BumpUp = true> class Arena {
  static constexpr size_t Alignment = alignof(std::max_align_t);
  static constexpr auto align(size_t x) -> size_t {
    return (x + Alignment - 1) & ~(Alignment - 1);
  }

protected:
  // meta data is always in front, so we don't have to handle
  // the rare accidental case of huge slabs separately.
  static constexpr size_t MetaSize = align(2 * sizeof(void *));
  static_assert(std::has_single_bit(Alignment));
  static_assert(__STDCPP_DEFAULT_NEW_ALIGNMENT__ >= Alignment);

public:
  using value_type = void;
  constexpr Arena(const Arena &) = default;
  constexpr Arena(Arena &&) noexcept = default;
  constexpr auto operator=(const Arena &) -> Arena & = default;
#ifndef NDEBUG
  constexpr void fillWithJunk(void *p, size_t Size) {
    if ((Size & 7) == 0) {
      std::fill_n(static_cast<std::int64_t *>(p), Size >> 3,
                  std::numeric_limits<std::int64_t>::min());
    } else std::fill_n(static_cast<char *>(p), Size, -1);
  }
#endif
  constexpr void bigAlloc(size_t Size) {
    void **oldMeta = getMetaEnd(sEnd);
    void *next = oldMeta[0];
    if (next && Size <= (SlabSize - 2 * MetaSize)) return initSlab(next);
    initNewSlab(align(Size) + SlabSize);
    oldMeta[0] = slab;
    void *firstSlab = oldMeta[1];
    getMetaStart(slab)[1] = firstSlab;
    getMetaEnd(sEnd)[0] = next;
    getMetaEnd(sEnd)[1] = firstSlab;
  }
  [[using gnu: returns_nonnull, alloc_size(2), assume_aligned(Alignment),
    malloc]] constexpr auto
  allocate(size_t Size) -> void * {
    void *p = allocCore(Size);
    if (outOfSlab()) [[unlikely]] {
      bigAlloc(Size);
      p = allocCore(Size);
    }
#if MATH_ADDRESS_SANITIZER_BUILD
    __asan_unpoison_memory_region(p, Size);
#endif
#if MATH_MEMORY_SANITIZER_BUILD
    __msan_allocated_memory(p, Size);
#endif
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
#if MATH_ADDRESS_SANITIZER_BUILD
    __asan_poison_memory_region(Ptr, Size);
#endif
    if constexpr (BumpUp) {
      if (static_cast<char *>(Ptr) + align(Size) == slab) slab = Ptr;
    } else if (Ptr == slab) {
      slab = static_cast<char *>(slab) + align(Size);
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
    char *cptr = static_cast<char *>(Ptr);
    if (Ptr) {
      if (void *p = tryReallocate(Ptr, capOld, capNew)) {
        if constexpr ((!BumpUp) & (!ForOverwrite))
          std::copy_n(cptr, szOld, static_cast<char *>(p));
        return p;
      }
      if constexpr (BumpUp) {
        if (Ptr == (char *)slab - align(capOld)) {
          slab = cptr + align(capNew);
          if (!outOfSlab()) {
#if MATH_ADDRESS_SANITIZER_BUILD
            __asan_unpoison_memory_region(cptr + capOld, capNew - capOld);
#endif
#if MATH_MEMORY_SANITIZER_BUILD
            __msan_allocated_memory(cptr + capOld, capNew - capOld);
#endif
            return Ptr;
          }
        }
      } else if (Ptr == slab) {
        size_t extraSize = align(capNew - capOld);
        slab = static_cast<char *>(slab) - extraSize;
        if (!outOfSlab()) {
#if MATH_ADDRESS_SANITIZER_BUILD
          __asan_unpoison_memory_region(slab, extraSize);
#endif
#if MATH_MEMORY_SANITIZER_BUILD
          __msan_allocated_memory(SlabCur, extraSize);
#endif
          if constexpr (!ForOverwrite) std::copy_n(cptr, capOld, (char *)slab);
          return slab;
        }
      }
    }
    // we need to allocate new memory
    auto newPtr = allocate(capNew);
    if (szOld && Ptr) {
      if constexpr (!ForOverwrite) std::copy_n(cptr, szOld, (char *)newPtr);
      deallocate(Ptr, capOld);
    }
    return newPtr;
  }
#if MATH_ADDRESS_SANITIZER_BUILD
  constexpr void poison_slabs(void *s0, void *e0) {
    for (char *s = static_cast<char *>(s0), *e = static_cast<char *>(e0); s;) {
      if constexpr (BumpUp) __asan_poison_memory_region(s, e - s);
      else __asan_poison_memory_region(e, s - e);
      s = static_cast<char *>(getNext(e));
      if (!s) break;
      e = static_cast<char *>(getEnd(s));
    }
  }
#endif
  /// free all memory, but the Arena holds onto it.
  constexpr void reset() {
    initSlab(getFirstEnd(sEnd));
#if MATH_ADDRESS_SANITIZER_BUILD
    poison_slabs(slab, sEnd);
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

  template <typename T, typename... Args>
  constexpr auto construct(Args &&...args) -> T * {
    auto *p = allocate(sizeof(T));
    return new (p) T(std::forward<Args>(args)...);
  }
  struct CheckPoint {
    void *const p;
    void *const e;
  };
  [[nodiscard]] constexpr auto checkpoint() -> CheckPoint {
    return {slab, sEnd};
  }
  constexpr void rollback(CheckPoint p) {
#if MATH_ADDRESS_SANITIZER_BUILD
    poison_slabs(p.p, p.e);
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
#if MATH_ADDRESS_SANITIZER_BUILD
          __asan_unpoison_memory_region((char *)Ptr + capOld, capNew - capOld);
#endif
#if MATH_MEMORY_SANITIZER_BUILD
          __msan_allocated_memory((char *)Ptr + capOld, capNew - capOld);
#endif
          return Ptr;
        }
      }
    } else if (Ptr == slab) {
      size_t extraSize = align(capNew - capOld);
      slab = (char *)slab - extraSize;
      if (!outOfSlab()) {
#if MATH_ADDRESS_SANITIZER_BUILD
        __asan_unpoison_memory_region(slab, extraSize);
#endif
#if MATH_MEMORY_SANITIZER_BUILD
        __msan_allocated_memory(SlabCur, extraSize);
#endif
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
    if constexpr (BumpUp) return static_cast<char *>(ptr) + N;
    else return static_cast<char *>(ptr) - N;
  }
  static constexpr auto outOfSlab(void *current, void *last) -> bool {
    if constexpr (BumpUp) return current >= last;
    else return current < last;
  }
  constexpr auto outOfSlab() -> bool { return outOfSlab(slab, sEnd); }
  constexpr void initSlab(void *p) {
    slab = p;
    sEnd = getEnd(p);
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
    void *old = slab;
    slab = bump(old, align(Size));
    if constexpr (BumpUp) return old;
    else return slab;
  }

protected:
  void *slab;
  void *sEnd;
  /// meta layout:
  /// Either an array of length 2.
  /// 0: sEnd
  /// 1: first slab
  static constexpr auto getMetaStart(void *p) -> void ** {
    if constexpr (BumpUp) return static_cast<void **>(p) - 2;
    else return static_cast<void **>(p);
  }
  /// meta layout:
  /// Either an array of length 2.
  /// 0: next slab
  /// 1: first slab
  static constexpr auto getMetaEnd(void *p) -> void ** {
    if constexpr (BumpUp) return static_cast<void **>(p);
    else return static_cast<void **>(p) - 2;
  }
  static constexpr auto getFirstStart(void *p) -> void * {
    return getMetaStart(p)[1];
  }
  static constexpr auto getFirstEnd(void *p) -> void * {
    return getMetaEnd(p)[1];
  }
  static constexpr auto getNext(void *p) -> void * { return getMetaEnd(p)[0]; }
  static constexpr auto getEnd(void *p) -> void * { return getMetaStart(p)[0]; }

  constexpr void initNewSlab(size_t size) {
    auto [p, sz] = alloc_at_least(size);
    char *c = static_cast<char *>(p) + MetaSize;
    char *q = static_cast<char *>(p) + sz - MetaSize;
    void **metac = static_cast<void **>(p);
    void **metaq = static_cast<void **>(static_cast<void *>(q));
    if constexpr (BumpUp) {
      metac[0] = q;
      slab = c;
      sEnd = q;
    } else {
      metaq[0] = c;
      slab = q;
      sEnd = c;
    }
#ifndef NDEBUG
    fillWithJunk(c, sz - 2 * MetaSize);
#endif
    // poison everything except the meta data
#if MATH_ADDRESS_SANITIZER_BUILD
    __asan_poison_memory_region(c, q - c);
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
    this->initNewSlab(SlabSize);
    this->getMetaStart(this->slab)[1] = this->slab;
    this->getMetaEnd(this->sEnd)[0] = nullptr;
    this->getMetaEnd(this->sEnd)[1] = this->slab;
  }

  OwningArena(OwningArena &&other) = delete;
  OwningArena(const OwningArena &) = delete;
  constexpr ~OwningArena() {
    char *p = static_cast<char *>(this->getFirstEnd(this->sEnd));
    constexpr size_t m = Arena<SlabSize, BumpUp>::MetaSize;
    constexpr size_t mpad = 2 * m;
    while (p) {
      char *end = static_cast<char *>(this->getEnd(p));
      char *next = static_cast<char *>(this->getNext(end)); // load before free!
      if constexpr (BumpUp) free(p - m, end - p + mpad);
      else free(end - m, p - end + mpad);
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
  [[no_unique_address]] Valid<Alloc> A;

public:
  using value_type = T;
  template <typename U> struct rebind { // NOLINT(readability-identifier-naming)
    using other = WArena<U, SlabSize, BumpUp>;
  };
  constexpr explicit WArena(Alloc *alloc) : A(alloc) {}
  constexpr explicit WArena(Alloc &alloc) : A(&alloc) {}
  constexpr explicit WArena(Valid<Alloc> alloc) : A(alloc) {}
  constexpr WArena(const WArena &other) = default;
  template <typename U>
  constexpr WArena(WArena<U> other) : A(other.get_allocator()) {}
  [[nodiscard, gnu::returns_nonnull]] constexpr auto get_allocator() const
    -> Alloc * {
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
constexpr auto wrap(Valid<Arena<SlabSize, BumpUp>> a)
  -> WArena<T, SlabSize, BumpUp> {
  return WArena<T, SlabSize, BumpUp>(a);
}

static_assert(
  std::same_as<std::allocator_traits<WArena<int64_t *>>::size_type, size_t>);
static_assert(
  std::same_as<std::allocator_traits<WArena<int64_t>>::pointer, int64_t *>);

static_assert(std::is_trivially_copyable_v<Valid<Arena<>>>);
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

template <typename F, typename... T>
concept ArenaPtrCallable = requires(F f) {
  { f(std::declval<Arena<> *>(), std::declval<T>()...) };
};
template <typename F, typename... T>
concept ArenaValCallable = requires(F f) {
  { f(std::declval<Arena<>>(), std::declval<T>()...) };
};
/// If we receive a pointer, we check if we can use either an arena pointer
/// or an arena value. The pointer and value imply, respectively,
/// that the lifetime of some objects are going to outlive or are
/// bounded by the function call `f`.
template <typename F, typename... T>
constexpr auto call(Arena<> *alloc, const F &f, T &&...t) {
  if constexpr (ArenaPtrCallable<F, T...>)
    return f(alloc, std::forward<T>(t)...);
  else if constexpr (ArenaValCallable<F, T...>)
    return f(*alloc, std::forward<T>(t)...);
  else return f(std::forward<T>(t)...);
}
/// If we require that the lifetime of anything allocated by the arena
/// are bounded by the function call `f`, we can use a value.
template <typename F, typename... T>
constexpr auto call(Arena<> alloc, const F &f, T &&...t) {
  if constexpr (ArenaValCallable<F, T...>)
    return f(alloc, std::forward<T>(t)...);
  else return f(std::forward<T>(t)...);
}

} // namespace poly::alloc

template <size_t SlabSize, bool BumpUp>
auto operator new(size_t Size, poly::alloc::Arena<SlabSize, BumpUp> &Alloc)
  -> void * {
  return Alloc.allocate(Size);
}

template <size_t SlabSize, bool BumpUp>
void operator delete(void *, poly::alloc::Arena<SlabSize, BumpUp> &) {}
