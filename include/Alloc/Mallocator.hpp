#pragma once

#include <cstddef>
#ifdef USING_MIMALLOC
#include <mimalloc-new-delete.h>
#elif USING_JEMALLOC
#include "jemalloc-new-delete.hpp"
#else
#include <cstdlib>
#endif
namespace poly::alloc {

template <class Pointer, class SizeType = size_t> struct AllocResult {
  Pointer *ptr;
  SizeType count;
};

inline auto good_malloc_size(size_t n) -> size_t {
#ifdef USING_MIMALLOC
  return mi_good_size(n);
#elif USING_JEMALLOC
  return nallocx(n, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__));
#else
  return n;
#endif
}

inline auto malloc(size_t n) -> void * {
#ifdef USING_MIMALLOC
  return mi_malloc(n);
#elif USING_JEMALLOC
  return mallocx(n, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__));
#else
  return std::malloc(n);
#endif
}
inline auto malloc(size_t n, std::align_val_t al) -> void * {
  auto a = static_cast<size_t>(al);
#ifdef USING_MIMALLOC
  return mi_malloc_aligned(n, a);
#elif USING_JEMALLOC
  return mallocx(n, MALLOCX_ALIGN(a));
#else
  return std::aligned_alloc(a, n);
#endif
}

inline auto zalloc(size_t n) -> void * {
#ifdef USING_MIMALLOC
  return mi_zalloc(n);
#elif USING_JEMALLOC
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
#elif USING_JEMALLOC
  return mallocx(n, MALLOCX_ALIGN(a) | MALLOCX_ZERO);
#else
  return std::aligned_alloc(a, n);
#endif
}

inline auto realloc(void *p, size_t n) -> void * {
#ifdef USING_MIMALLOC
  return mi_realloc(p, n);
#elif USING_JEMALLOC
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
#elif USING_JEMALLOC
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
  return mi_realloc(p, n, a);
#elif USING_JEMALLOC
  return rallocx(p, n MALLOCX_ALIGN(a));
#else
  (void)a; // FIXME
  return std::realloc(p, n);
#endif
}

inline void free(void *p) {
#ifdef USING_MIMALLOC
  return mi_free(p);
#elif USING_JEMALLOC
  return dallocx(p, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__)));
#else
  return std::free(p);
#endif
};

inline void free(void *p, size_t n) {
#ifdef USING_MIMALLOC
  return mi_free_size(p, n);
#elif USING_JEMALLOC
  return sdallocx(p, n, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__)));
#else
  (void)n;
  return std::free(p);
#endif
};

inline void free(void *p, std::align_val_t al) {
  auto a = static_cast<size_t>(al);
#ifdef USING_MIMALLOC
  return mi_free_aligned(p);
#elif USING_JEMALLOC
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
#elif USING_JEMALLOC
  return sdallocx(p, n, MALLOCX_ALIGN(a)));
#else
  (void)a;
  (void)n;
  return std::free(p);
#endif
};

inline auto allocAtLeast(size_t n) -> AllocResult<void> {
  n = good_malloc_size(n);
  return {malloc(n), n};
}

inline auto allocAtLeast(size_t n, std::align_val_t al) -> AllocResult<void> {
  n = good_malloc_size(n);
  return {malloc(n, al), n};
}

} // namespace poly::alloc
