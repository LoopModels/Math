#pragma once

#ifndef NDEBUG
#include <iostream>
#include <ostream>
#include <source_location>
#include <version>

namespace poly::utils {

[[gnu::noinline]] static void errorReport(std::source_location location) {
  std::cout << "invariant violation\nfile: " << location.file_name() << ":"
            << location.line() << ":" << location.column() << " `"
            << location.function_name() << "`\n";
  __builtin_trap();
}
template <typename T>
[[gnu::noinline]] static void errorReport(const T &x, const T &y,
                                          std::source_location location) {
  std::cout << x << " != " << y << "\n";
  errorReport(location);
}

[[gnu::artificial, gnu::always_inline]] constexpr inline void
invariant(bool condition,
          std::source_location location = std::source_location::current()) {
  if (!condition) [[unlikely]]
    errorReport(location);
}
template <typename T>
[[gnu::artificial, gnu::always_inline]] constexpr inline void
invariant(const T &x, const T &y,
          std::source_location location = std::source_location::current()) {
  if (x != y) [[unlikely]]
    errorReport(x, y, location);
}
// we want gdb-friendly builtin trap
#define ASSERT(condition) ::poly::utils::invariant(condition)
} // namespace poly::utils
#else // ifdef NDEBUG
#ifdef __cpp_lib_unreachable
#include <utility>
#endif
namespace poly::utils {
[[gnu::artificial]] constexpr inline void invariant(bool condition) {

#ifdef __has_cpp_attribute
#if __has_cpp_attribute(assume)
  [[assume(condition)]];
#endif
#endif
  if (!condition) {
#ifdef __cpp_lib_unreachable
    std::unreachable();
#else
#ifdef __has_builtin
#if __has_builtin(__builtin_unreachable)
    __builtin_unreachable();
#endif
#endif
#endif
  }
}
template <typename T>
[[gnu::artificial]] constexpr inline void invariant(const T &x, const T &y) {
#ifdef __has_cpp_attribute
#if __has_cpp_attribute(assume)
  [[assume(x == y)]];
#endif
#endif
  if (x != y) {
#ifdef __cpp_lib_unreachable
    std::unreachable();
#else
#ifdef __has_builtin
#if __has_builtin(__builtin_unreachable)
    __builtin_unreachable();
#endif
#endif
#endif
  }
}
#define ASSERT(condition) ((void)0)
} // namespace poly::utils
#endif // ifdef NDEBUG
