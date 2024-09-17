#pragma once

// Based on the implementation from mimalloc
// https://github.com/microsoft/mimalloc/blob/master/include/mimalloc-new-delete.h

#include <cstddef>
#include <jemalloc/jemalloc.h>
#include <new>

void operator delete(void *p) noexcept {
  dallocx(p, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__));
};
void operator delete[](void *p) noexcept {
  dallocx(p, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__));
};

void operator delete(void *p, const std::nothrow_t &) noexcept {
  dallocx(p, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__));
}
void operator delete[](void *p, const std::nothrow_t &) noexcept {
  dallocx(p, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__));
}

[[using gnu: returns_nonnull, alloc_size(1)]] void *
operator new(std::size_t n) noexcept(false) {
  return mallocx(n, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__));
}
[[using gnu: returns_nonnull, alloc_size(1)]] void *
operator new[](std::size_t n) noexcept(false) {
  return mallocx(n, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__));
}

[[using gnu: returns_nonnull, alloc_size(1)]] void *
operator new(std::size_t n, const std::nothrow_t &tag) noexcept {
  (void)(tag);
  return mallocx(n, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__));
}
[[using gnu: returns_nonnull, alloc_size(1)]] void *
operator new[](std::size_t n, const std::nothrow_t &tag) noexcept {
  (void)(tag);
  return mallocx(n, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__));
}

#if (__cplusplus >= 201402L || _MSC_VER >= 1916)
void operator delete(void *p, std::size_t n) noexcept {
  sdallocx(p, n, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__));
};
void operator delete[](void *p, std::size_t n) noexcept {
  sdallocx(p, n, MALLOCX_ALIGN(__STDCPP_DEFAULT_NEW_ALIGNMENT__));
};
#endif

#if (__cplusplus > 201402L || defined(__cpp_aligned_new))
void operator delete(void *p, std::align_val_t al) noexcept {
  dallocx(p, MALLOCX_ALIGN(static_cast<size_t>(al)));
}
void operator delete[](void *p, std::align_val_t al) noexcept {
  dallocx(p, MALLOCX_ALIGN(static_cast<size_t>(al)));
}
void operator delete(void *p, std::size_t n, std::align_val_t al) noexcept {
  sdallocx(p, n, MALLOCX_ALIGN(static_cast<size_t>(al)));
};
void operator delete[](void *p, std::size_t n, std::align_val_t al) noexcept {
  sdallocx(p, n, MALLOCX_ALIGN(static_cast<size_t>(al)));
};
void operator delete(void *p, std::align_val_t al,
                     const std::nothrow_t &) noexcept {
  dallocx(p, MALLOCX_ALIGN(static_cast<size_t>(al)));
}
void operator delete[](void *p, std::align_val_t al,
                       const std::nothrow_t &) noexcept {
  dallocx(p, MALLOCX_ALIGN(static_cast<size_t>(al)));
}

[[using gnu: returns_nonnull, alloc_size(1), alloc_align(2)]] void *
operator new(std::size_t n, std::align_val_t al) noexcept(false) {
  return mallocx(n, MALLOCX_ALIGN(static_cast<size_t>(al)));
}
[[using gnu: returns_nonnull, alloc_size(1), alloc_align(2)]] void *
operator new[](std::size_t n, std::align_val_t al) noexcept(false) {
  return mallocx(n, MALLOCX_ALIGN(static_cast<size_t>(al)));
}
[[using gnu: returns_nonnull, alloc_size(1), alloc_align(2)]] void *
operator new(std::size_t n, std::align_val_t al,
             const std::nothrow_t &) noexcept {
  return mallocx(n, MALLOCX_ALIGN(static_cast<size_t>(al)));
}
[[using gnu: returns_nonnull, alloc_size(1), alloc_align(2)]] void *
operator new[](std::size_t n, std::align_val_t al,
               const std::nothrow_t &) noexcept {
  return mallocx(n, MALLOCX_ALIGN(static_cast<size_t>(al)));
}
#endif
