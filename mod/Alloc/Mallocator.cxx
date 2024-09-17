#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifndef USE_MODULE
#include <concepts>
#include <cstddef>
#include <memory>
#include <new>
#include <version>
#endif

#ifndef __has_feature      // Optional of course.
#define __has_feature(x) 0 // Compatibility with non-clang compilers.
#endif
#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
#define USE_MIMALLOC 0
#define USE_JEMALLOC 0
#else
#ifdef USING_MIMALLOC
#define USE_MIMALLOC 1
#define USE_JEMALLOC 0
#elifdef USING_JEMALLOC
#define USE_MIMALLOC 0
#define USE_JEMALLOC 1
#else
#define USE_MIMALLOC 0
#define USE_JEMALLOC 0
#endif
#endif
#if USE_MIMALLOC
// #include <mimalloc-new-delete.h>
#include <mimalloc.h>
#elif USE_JEMALLOC
// #include "jemalloc-new-delete.hpp"
#include <jemalloc/jemalloc.h>
#elifndef USE_MODULE
#include <cstdlib>
#endif
#ifndef USE_MODULE
#ifdef __cpp_lib_allocate_at_least
#include <memory>
#endif

#include "Utilities/Invariant.cxx"
#else
export module Allocator;

import STL;
import Invariant;
#endif

#ifdef USE_MODULE
export namespace alloc {
#else
namespace alloc {
#endif

#ifdef __cpp_lib_allocate_at_least
template <class Pointer> using AllocResult = std::allocation_result<Pointer *>;
#else
template <class Pointer> struct AllocResult {
  Pointer *ptr;
  size_t count;
};
#endif

inline auto good_malloc_size(size_t n) -> size_t {
#if USE_MIMALLOC
  return mi_good_size(n);
#elif USE_JEMALLOC
  return nallocx(n, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__));
#else
  return n;
#endif
}

[[using gnu: returns_nonnull, malloc, alloc_size(1)]] inline auto
malloc(size_t n) -> void * {
#if USE_MIMALLOC
  return mi_malloc(n);
#elif USE_JEMALLOC
  return mallocx(n, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__));
#else
  return std::malloc(n);
#endif
}
[[using gnu: returns_nonnull, malloc, alloc_size(1),
  alloc_align(2)]] inline auto
malloc(size_t n, std::align_val_t al) -> void * {
  auto a = static_cast<size_t>(al);
#if USE_MIMALLOC
  return mi_malloc_aligned(n, a);
#elif USE_JEMALLOC
  return mallocx(n, MALLOCX_ALIGN(a));
#else
  return std::aligned_alloc(a, n);
#endif
}

[[using gnu: returns_nonnull, malloc, alloc_size(1)]] inline auto
zalloc(size_t n) -> void * {
#if USE_MIMALLOC
  return mi_zalloc(n);
#elif USE_JEMALLOC
  return mallocx(n, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__) |
                      MALLOCX_ZERO);
#else
  return std::malloc(n);
#endif
}
[[using gnu: returns_nonnull, malloc, alloc_size(1),
  alloc_align(2)]] inline auto
zalloc(size_t n, std::align_val_t al) -> void * {
  auto a = static_cast<size_t>(al);
#if USE_MIMALLOC
  return mi_zalloc_aligned(n, a);
#elif USE_JEMALLOC
  return mallocx(n, MALLOCX_ALIGN(a) | MALLOCX_ZERO);
#else
  return std::aligned_alloc(a, n);
#endif
}

inline auto realloc(void *p, size_t n) -> void * {
#if USE_MIMALLOC
  return mi_realloc(p, n);
#elif USE_JEMALLOC
  return rallocx(p, n, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__)));
#else
  return std::realloc(p, n);
#endif
}
// expanding in place
// returns `nullptr` on failure
inline auto exalloc(void *p, size_t n) -> void * {
#if USE_MIMALLOC
  return mi_expand(p, n);
#elif USE_JEMALLOC
  return xallocx(p, n, 0, MALLOCX_ALIGN(a));
#else
  (void)p;
  (void)n;
  return nullptr;
#endif
}

inline auto realloc(void *p, size_t n, std::align_val_t al) -> void * {
  auto a = static_cast<size_t>(al);
#if USE_MIMALLOC
  return mi_realloc_aligned(p, n, a);
#elif USE_JEMALLOC
  return rallocx(p, n MALLOCX_ALIGN(a));
#else
  (void)a; // FIXME
  return std::realloc(p, n);
#endif
}

inline void free(void *p) {
#if USE_MIMALLOC
  return mi_free(p);
#elif USE_JEMALLOC
  return dallocx(p, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__)));
#else
  return std::free(p);
#endif
};

inline void free(void *p, size_t n) {
#if USE_MIMALLOC
  return mi_free_size(p, n);
#elif USE_JEMALLOC
  return sdallocx(p, n, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__)));
#else
  (void)n;
  return std::free(p);
#endif
};

inline void free(void *p, std::align_val_t al) {
  auto a = static_cast<size_t>(al);
#if USE_MIMALLOC
  return mi_free_aligned(p, a);
#elif USE_JEMALLOC
  return dallocx(p, MALLOCX_ALIGN(a)));
#else
  (void)a;
  return std::free(p);
#endif
};

inline void free(void *p, size_t n, std::align_val_t al) {
  auto a = static_cast<size_t>(al);
#if USE_MIMALLOC
  return mi_free_size_aligned(p, n, a);
#elif USE_JEMALLOC
  return sdallocx(p, n, MALLOCX_ALIGN(a)));
#else
  (void)a;
  (void)n;
  return std::free(p);
#endif
};

inline auto alloc_at_least(size_t n) -> AllocResult<void> {
  size_t newn = good_malloc_size(n);
  utils::invariant(newn >= n);
  return {malloc(newn), newn};
}

inline auto alloc_at_least(size_t n, std::align_val_t al) -> AllocResult<void> {
  size_t newn = good_malloc_size(n);
  utils::invariant(newn >= n);
  return {malloc(newn, al), newn};
}

template <class T> struct Mallocator {
  using value_type = T;
  template <class U> struct rebind { // NOLINT(readability-identifier-naming)
    using other = Mallocator<U>;
  };
// https://github.com/microsoft/mimalloc/issues/199#issuecomment-596023203
// https://github.com/jemalloc/jemalloc/issues/1533#issuecomment-507915829
#if USE_MIMALLOC || USE_JEMALLOC
  static constexpr bool overalign = alignof(T) > 16 ||
                                    (alignof(T) > 8 && sizeof(T) <= 8);
#else
  static constexpr bool overalign = alignof(T) > 16;
#endif
  static auto allocate(size_t n) -> T * {
    if constexpr (overalign)
      return static_cast<T *>(
        malloc(n * sizeof(T), std::align_val_t{alignof(T)}));
    else return static_cast<T *>(malloc(n * sizeof(T)));
  };
  static auto allocate(size_t n, std::align_val_t al) -> T * {
    return static_cast<T *>(malloc(n * sizeof(T), al));
  };
  // NOLINTNEXTLINE(readability-identifier-naming)
  static auto allocate_at_least(size_t n) {
    T *t;
    size_t m;
    if constexpr (overalign) {
      auto [p, l] =
        allocate_at_least(n * sizeof(T), std::align_val_t{alignof(T)});
      t = static_cast<T *>(p);
      m = l / sizeof(T);
    } else {
      auto [p, l] = alloc_at_least(n * sizeof(T));
      t = static_cast<T *>(p);
      m = l / sizeof(T);
    }
    utils::invariant(m >= n);
    return AllocResult<T>{t, m};
  }
  // NOLINTNEXTLINE(readability-identifier-naming)
  static auto allocate_at_least(size_t n, std::align_val_t al) {
    auto [p, l] = alloc_at_least(n * sizeof(T), al);
    size_t m = l / sizeof(T);
    utils::invariant(m >= n);
    return AllocResult<T>{static_cast<T *>(p), m};
  }
  static void deallocate(T *p, size_t n) {
    if constexpr (overalign)
      free(p, n * sizeof(T), std::align_val_t{alignof(T)});
    else free(p, n * sizeof(T));
  };
  static void deallocate(T *p, size_t n, std::align_val_t al) {
    free(p, n * sizeof(T), al);
  };
  constexpr auto operator==(const Mallocator &) { return true; };
  template <class U> constexpr operator Mallocator<U>() { return {}; }
};
template <class A>
concept CanAllocAtLeast = requires(A a) {
  {
    a.allocate_at_least(1)
  } -> std::same_as<AllocResult<typename A::value_type>>;
};
static_assert(CanAllocAtLeast<Mallocator<ptrdiff_t>>);

template <class A>
[[gnu::always_inline]] inline auto
alloc_at_least(A a, size_t n) -> AllocResult<typename A::value_type> {
  if constexpr (CanAllocAtLeast<A>) return a.allocate_at_least(n);
  else return {a.allocate(n), n};
}

template <typename A>
concept Allocator = requires(A a) {
  typename A::value_type;
  { a.allocate(1) } -> std::same_as<typename std::allocator_traits<A>::pointer>;
  {
    a.deallocate(std::declval<typename std::allocator_traits<A>::pointer>(), 1)
  };
};
template <typename A>
concept FreeAllocator = Allocator<A> && std::is_empty_v<A>;

static_assert(FreeAllocator<Mallocator<int>>);

} // namespace alloc
