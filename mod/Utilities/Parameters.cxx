#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifndef USE_MODULE
#include <type_traits>
#else
export module Param;
import STL;
#endif

#ifdef USE_MODULE
export namespace utils {
#else
namespace utils {
#endif
template <typename T>
concept TriviallyCopyable =
  std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>;
} // namespace utils

template <typename T> struct InParameter {
  using type = const T &;
};
template <utils::TriviallyCopyable T> struct InParameter<T> {
  using type = T;
};

#ifdef USE_MODULE
export namespace utils {
#else
namespace utils {
#endif

/// This can be used like
/// auto foo_impl(inparam_t<T> x, ...);
/// template <typename T>
/// [[gnu::always_inline]] inline auto foo(const T& x){
///  return foo_impl<T>(x); // inparam_t blocks deduction
/// }
template <typename T> using inparam_t = typename InParameter<T>::type;

} // namespace utils
