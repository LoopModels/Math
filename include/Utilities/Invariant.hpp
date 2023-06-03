#pragma once

#ifndef NDEBUG
#include <ostream>
#include <source_location>
[[gnu::artificial]] constexpr inline void
invariant(bool condition,
          std::source_location location = std::source_location::current()) {
  if (!condition) {
    std::cout << "invariant violation\nfile: " << location.file_name() << ":"
              << location.line() << ":" << location.column() << " `"
              << location.function_name() << "`\n";
    __builtin_trap();
  }
}
template <typename T>
[[gnu::artificial]] constexpr inline void
invariant(const T &x, const T &y,
          std::source_location location = std::source_location::current()) {
  if (x != y) {
    std::cout << "invariant violation: " << x << " != " << y
              << "\nfile: " << location.file_name() << ":" << location.line()
              << ":" << location.column() << " `" << location.function_name()
              << "`\n";
    __builtin_trap();
  }
}
// we want gdb-friendly builtin trap
#define ASSERT(condition) invariant(condition)

#else // ifdef NDEBUG
#if __cplusplus >= 202202L
#include <utility>
#endif
[[gnu::artificial]] constexpr inline void invariant(bool condition) {
  if (!condition) {
#if __cplusplus >= 202202L
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
  if (x != y) {
#if __cplusplus >= 202202L
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
#endif // ifdef NDEBUG
