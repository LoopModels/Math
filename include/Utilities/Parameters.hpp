#pragma once
#ifndef Parameters_hpp_INCLUDED
#define Parameters_hpp_INCLUDED

#include <type_traits>

template <typename T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>;
template <typename T> struct InParameter {
  using type = const T &;
};
template <TriviallyCopyable T> struct InParameter<T> {
  using type = T;
};

/// This can be used like
/// auto foo_impl(inparam_t<T> x, ...);
/// template <typename T>
/// [[gnu::always_inline]] inline auto foo(const T& x){
///  return foo_impl<T>(x); // inparam_t blocks deduction
/// }
template <typename T> using inparam_t = typename InParameter<T>::type;

#endif // Parameters_hpp_INCLUDED
