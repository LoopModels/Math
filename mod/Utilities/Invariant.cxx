#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifndef USE_MODULE
#ifdef NDEBUG
#ifdef __cpp_lib_unreachable
#include <utility>
#include <version>
#endif
#endif
#include <cstdlib>
#endif
#if __has_builtin(__builtin_trap)
#define TRAP() __builtin_trap()
#else
#define TRAP() std::abort()
#endif

#ifdef USE_MODULE
export module Invariant;
import STL;
#endif

#ifdef USE_MODULE
export namespace utils {
#else
namespace utils {
#endif

#ifndef NDEBUG
[[gnu::artificial, gnu::always_inline]] constexpr inline void
invariant(bool condition) {
  if (!condition) [[unlikely]]
    TRAP();
}
template <typename T>
[[gnu::artificial, gnu::always_inline]] constexpr inline void invariant(T x,
                                                                        T y) {
  if (x != y) [[unlikely]]
    TRAP();
}
#else
[[gnu::artificial, gnu::always_inline]] constexpr inline void invariant(bool) {}
template <typename T>
[[gnu::artificial, gnu::always_inline]] constexpr inline void invariant(T, T) {}
#endif

[[gnu::artificial, gnu::always_inline]] constexpr inline void
assume(bool condition) {
#ifndef NDEBUG
  if (!condition) [[unlikely]]
    TRAP();
#else
#if defined(__has_cpp_attribute) && __has_cpp_attribute(assume)
  [[assume(condition)]];
#else
  if (!condition) {
#ifdef __cpp_lib_unreachable
    std::unreachable();
#else
#ifdef __has_builtin
#if __has_builtin(__builtin_unreachable)
    __builtin_unreachable();
#endif
#endif
#endif // __cpp_lib_unreachable
  }
#endif // assume
#endif // NDEBUG
}
template <typename T>
[[gnu::artificial, gnu::always_inline]] constexpr inline void assumeeq(T x,
                                                                       T y) {
#ifndef NDEBUG
  if (x != y) [[unlikely]]
    TRAP();
#else
#if defined(__has_cpp_attribute) && __has_cpp_attribute(assume)
  [[assume(x == y)]];
#else
  if (x != y) {
#ifdef __cpp_lib_unreachable
    std::unreachable();
#else
#ifdef __has_builtin
#if __has_builtin(__builtin_unreachable)
    __builtin_unreachable();
#endif
#endif
#endif // __cpp_lib_unreachable
  }
#endif // assume
#endif // NDEBUG
}

} // namespace utils
