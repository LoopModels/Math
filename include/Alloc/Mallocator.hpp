// #pragma once

#include <cstddef>
#ifdef USING_MIMALLOC
// #include <mimalloc-new-delete.h>
#include <mimalloc.h>
#elifdef USING_JEMALLOC
// #include "jemalloc-new-delete.hpp"
#include <jemalloc/jemalloc.h>
#else
#include <cstdlib>
#endif
namespace poly::alloc {

#ifdef __cpp_lib_allocate_at_least
template <class Pointer> using AllocResult = std::allocation_result<Pointer *>;
#else
template <class Pointer> struct AllocResult {
  Pointer *ptr;
  size_t count;
};
#endif

inline auto good_malloc_size(size_t n) -> size_t {
#ifdef USING_MIMALLOC
  return mi_good_size(n);
#elifdef USING_JEMALLOC
  return nallocx(n, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__));
#else
  return n;
#endif
}

inline auto malloc(size_t n) -> void * {
#ifdef USING_MIMALLOC
  return mi_malloc(n);
#elifdef USING_JEMALLOC
  return mallocx(n, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__));
#else
  return std::malloc(n);
#endif
}
inline auto malloc(size_t n, std::align_val_t al) -> void * {
  auto a = static_cast<size_t>(al);
#ifdef USING_MIMALLOC
  return mi_malloc_aligned(n, a);
#elifdef USING_JEMALLOC
  return mallocx(n, MALLOCX_ALIGN(a));
#else
  return std::aligned_alloc(a, n);
#endif
}

inline auto zalloc(size_t n) -> void * {
#ifdef USING_MIMALLOC
  return mi_zalloc(n);
#elifdef USING_JEMALLOC
  return mallocx(n, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__) |
                      MALLOCX_ZERO);
#else
  return std::malloc(n);
#endif
}
inline auto zalloc(size_t n, std::align_val_t al) -> void * {
  auto a = static_cast<size_t>(al);
#ifdef USING_MIMALLOC
  return mi_zalloc_aligned(n, a);
#elifdef USING_JEMALLOC
  return mallocx(n, MALLOCX_ALIGN(a) | MALLOCX_ZERO);
#else
  return std::aligned_alloc(a, n);
#endif
}

inline auto realloc(void *p, size_t n) -> void * {
#ifdef USING_MIMALLOC
  return mi_realloc(p, n);
#elifdef USING_JEMALLOC
  return rallocx(p, n, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__)));
#else
  return std::realloc(p, n);
#endif
}
// expanding in place
// returns `nullptr` on failure
inline auto exalloc(void *p, size_t n) -> void * {
#ifdef USING_MIMALLOC
  return mi_expand(p, n);
#elifdef USING_JEMALLOC
  return xallocx(p, n, 0, MALLOCX_ALIGN(a));
#else
  (void)p;
  (void)n;
  return nullptr;
#endif
}

inline auto realloc(void *p, size_t n, std::align_val_t al) -> void * {
  auto a = static_cast<size_t>(al);
#ifdef USING_MIMALLOC
  return mi_realloc_aligned(p, n, a);
#elifdef USING_JEMALLOC
  return rallocx(p, n MALLOCX_ALIGN(a));
#else
  (void)a; // FIXME
  return std::realloc(p, n);
#endif
}

inline void free(void *p) {
#ifdef USING_MIMALLOC
  return mi_free(p);
#elifdef USING_JEMALLOC
  return dallocx(p, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__)));
#else
  return std::free(p);
#endif
};

inline void free(void *p, size_t n) {
#ifdef USING_MIMALLOC
  return mi_free_size(p, n);
#elifdef USING_JEMALLOC
  return sdallocx(p, n, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__)));
#else
  (void)n;
  return std::free(p);
#endif
};

inline void free(void *p, std::align_val_t al) {
  auto a = static_cast<size_t>(al);
#ifdef USING_MIMALLOC
  return mi_free_aligned(p, a);
#elifdef USING_JEMALLOC
  return dallocx(p, MALLOCX_ALIGN(a)));
#else
  (void)a;
  return std::free(p);
#endif
};

inline void free(void *p, size_t n, std::align_val_t al) {
  auto a = static_cast<size_t>(al);
#ifdef USING_MIMALLOC
  return mi_free_size_aligned(p, n, a);
#elifdef USING_JEMALLOC
  return sdallocx(p, n, MALLOCX_ALIGN(a)));
#else
  (void)a;
  (void)n;
  return std::free(p);
#endif
};

inline auto alloc_at_least(size_t n) -> AllocResult<void> {
  n = good_malloc_size(n);
  return {malloc(n), n};
}

inline auto alloc_at_least(size_t n, std::align_val_t al) -> AllocResult<void> {
  n = good_malloc_size(n);
  return {malloc(n, al), n};
}

template <class T> struct Mallocator {
  using value_type = T;
  template <class U> struct rebind { // NOLINT(readability-identifier-naming)
    using other = Mallocator<U>;
  };
// https://github.com/microsoft/mimalloc/issues/199#issuecomment-596023203
// https://github.com/jemalloc/jemalloc/issues/1533#issuecomment-507915829
#if defined(USING_MIMALLOC) || defined(USING_JEMALLOC)
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
    return AllocResult<T>{t, m};
  }
  // NOLINTNEXTLINE(readability-identifier-naming)
  static auto allocate_at_least(size_t n, std::align_val_t al) {
    T *t;
    size_t m;
    auto [p, l] = alloc_at_least(n * sizeof(T), al);
    t = static_cast<T *>(p);
    m = l / sizeof(T);
    return AllocResult<T>{t, m};
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
[[gnu::always_inline]] inline auto alloc_at_least(A a, size_t n)
  -> AllocResult<typename A::value_type> {
  if constexpr (CanAllocAtLeast<A>) return a.allocate_at_least(n);
  else return {a.allocate(n), n};
}
} // namespace poly::alloc
