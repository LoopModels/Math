#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifndef USE_MODULE
#include <compare>
#include <concepts>
#include <limits>
#include <ostream>
#include <type_traits>

#include "Utilities/Invariant.cxx"
#else
export module Int8;

import Invariant;
import STL;
#endif

#ifdef USE_MODULE
export namespace numbers {
#else
namespace numbers {
#endif

template <std::integral I, bool nowrap = false, int alias = 0>
struct IntWrapper {
  enum class strong : I {};

private:
  static constexpr bool issigned = std::is_signed_v<I>;
  using T = std::conditional_t<nowrap, std::make_signed_t<I>, I>;
  [[gnu::always_inline, gnu::artificial]] inline static constexpr auto
  create(std::integral auto x) -> strong {
    if constexpr (nowrap) {
      utils::invariant(x >= std::numeric_limits<I>::min());
      utils::invariant(x <= std::numeric_limits<I>::max());
    }
    return x;
  }

  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator++(strong &x) -> strong & {
    x = static_cast<strong>(static_cast<T>(x) + T{1});
    return x;
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator++(strong &&x) -> decltype(auto) {
    x = static_cast<strong>(static_cast<T>(x) + T{1});
    return x;
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator--(strong &x) -> strong & {
    x = static_cast<strong>(static_cast<T>(x) - T{1});
    return x;
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator--(strong &&x) -> decltype(auto) {
    x = static_cast<strong>(static_cast<T>(x) - T{1});
    return x;
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator++(strong &x, int) -> strong {
    strong y = x;
    x = static_cast<strong>(static_cast<T>(x) + T{1});
    return y;
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator--(strong &x, int) -> strong {
    strong y = x;
    x = static_cast<strong>(static_cast<T>(x) - T{1});
    return y;
  }

  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator++(strong &&x, int) -> strong {
    strong y = x;
    x = static_cast<strong>(static_cast<T>(x) + T{1});
    return y;
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator--(strong &&x, int) -> strong {
    strong y = x;
    x = static_cast<strong>(static_cast<T>(x) - T{1});
    return y;
  }

  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator==(strong x, J y) -> bool
  requires((issigned == std::is_signed_v<J>) ||
           (!issigned && (sizeof(J) > sizeof(I))))
  {
    if constexpr (sizeof(J) >= sizeof(I)) return static_cast<J>(x) == y;
    else return static_cast<I>(x) == static_cast<I>(y);
  }
  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator==(J y, strong x) -> bool
  requires((issigned == std::is_signed_v<J>) ||
           (!issigned && (sizeof(J) > sizeof(I))))
  {
    if constexpr (sizeof(J) >= sizeof(I)) return y == static_cast<J>(x);
    else return static_cast<I>(x) == static_cast<I>(y);
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator==(strong x, strong y) -> bool {
    return static_cast<I>(x) == static_cast<I>(y);
  }

  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator<=>(strong x, J y) -> std::strong_ordering
  requires((issigned == std::is_signed_v<J>) ||
           (!issigned && (sizeof(J) > sizeof(I))))
  {
    if constexpr (sizeof(J) >= sizeof(I)) return static_cast<J>(x) <=> y;
    else return static_cast<I>(x) <=> static_cast<I>(y);
  }
  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator<=>(J y, strong x) -> std::strong_ordering
  requires((issigned == std::is_signed_v<J>) ||
           (!issigned && (sizeof(J) > sizeof(I))))
  {
    if constexpr (sizeof(J) >= sizeof(I)) return y <=> static_cast<J>(x);
    else return static_cast<I>(x) <=> static_cast<I>(y);
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator<=>(strong x, strong y) -> std::strong_ordering {
    return static_cast<I>(x) <=> static_cast<I>(y);
  }

  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator+(strong x, J y) -> strong
  requires(issigned == std::is_signed_v<J>)
  {
    if constexpr (sizeof(J) > sizeof(I)) return create(static_cast<J>(x) + y);
    else return static_cast<strong>(static_cast<T>(x) + static_cast<T>(y));
  }
  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator+(J y, strong x) -> strong
  requires(issigned == std::is_signed_v<J>)
  {
    if constexpr (sizeof(J) > sizeof(I)) return create(y + static_cast<J>(x));
    else return static_cast<strong>(static_cast<T>(x) + static_cast<T>(y));
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator+(strong x, strong y) -> strong {
    return static_cast<strong>(static_cast<T>(x) + static_cast<T>(y));
  }

  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator-(strong x, J y) -> strong
  requires(issigned == std::is_signed_v<J>)
  {
    if constexpr (sizeof(J) > sizeof(I)) return create(static_cast<J>(x) - y);
    else return static_cast<strong>(static_cast<T>(x) - static_cast<T>(y));
  }
  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator-(J y, strong x) -> strong
  requires(issigned == std::is_signed_v<J>)
  {
    if constexpr (sizeof(J) > sizeof(I)) return create(y - static_cast<J>(x));
    else return static_cast<strong>(static_cast<T>(y) - static_cast<T>(x));
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator-(strong x, strong y) -> strong {
    return static_cast<strong>(static_cast<T>(x) - static_cast<T>(y));
  }

  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator*(strong x, J y) -> strong
  requires(issigned == std::is_signed_v<J>)
  {
    if constexpr (sizeof(J) > sizeof(I)) return create(static_cast<J>(x) * y);
    else return static_cast<strong>(static_cast<T>(x) * static_cast<T>(y));
  }
  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator*(J y, strong x) -> strong
  requires(issigned == std::is_signed_v<J>)
  {
    if constexpr (sizeof(J) > sizeof(I)) return create(y * static_cast<J>(x));
    else return static_cast<strong>(static_cast<T>(y) * static_cast<T>(x));
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator*(strong x, strong y) -> strong {
    return static_cast<strong>(static_cast<T>(x) * static_cast<T>(y));
  }

  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator/(strong x, J y) -> strong
  requires(issigned == std::is_signed_v<J>)
  {
    if constexpr (sizeof(J) > sizeof(I)) return create(static_cast<J>(x) / y);
    else return static_cast<strong>(static_cast<I>(x) / static_cast<I>(y));
  }
  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator/(J y, strong x) -> strong
  requires(issigned == std::is_signed_v<J>)
  {
    if constexpr (sizeof(J) > sizeof(I)) return create(y / static_cast<J>(x));
    else return static_cast<strong>(static_cast<I>(y) / static_cast<I>(x));
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator/(strong x, strong y) -> strong {
    return static_cast<strong>(static_cast<I>(x) / static_cast<I>(y));
  }

  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator|(strong x, J y) -> strong
  requires(issigned == std::is_signed_v<J>)
  {
    if constexpr (sizeof(J) > sizeof(I)) return create(static_cast<J>(x) | y);
    else return static_cast<strong>(static_cast<T>(x) | static_cast<T>(y));
  }
  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator|(J y, strong x) -> strong
  requires(issigned == std::is_signed_v<J>)
  {
    if constexpr (sizeof(J) > sizeof(I)) return create(y | static_cast<J>(x));
    else return static_cast<strong>(static_cast<T>(y) | static_cast<T>(x));
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator|(strong x, strong y) -> strong {
    return static_cast<strong>(static_cast<T>(x) | static_cast<T>(y));
  }

  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator&(strong x, J y) -> strong
  requires(issigned == std::is_signed_v<J>)
  {
    if constexpr (sizeof(J) > sizeof(I)) return static_cast<J>(x) & y;
    else return static_cast<strong>(static_cast<T>(x) & static_cast<T>(y));
  }
  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator&(J y, strong x) -> strong
  requires(issigned == std::is_signed_v<J>)
  {
    if constexpr (sizeof(J) > sizeof(I)) return y & static_cast<J>(x);
    else return static_cast<strong>(static_cast<T>(y) & static_cast<T>(x));
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator&(strong x, strong y) -> strong {
    return static_cast<strong>(static_cast<T>(x) & static_cast<T>(y));
  }

  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator^(strong x, J y) -> strong
  requires(issigned == std::is_signed_v<J>)
  {
    if constexpr (sizeof(J) > sizeof(I)) return create(static_cast<J>(x) ^ y);
    else return static_cast<strong>(static_cast<T>(x) ^ static_cast<T>(y));
  }
  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator^(J y, strong x) -> strong
  requires(issigned == std::is_signed_v<J>)
  {
    if constexpr (sizeof(J) > sizeof(I)) return create(y ^ static_cast<J>(x));
    else return static_cast<strong>(static_cast<T>(y) ^ static_cast<T>(x));
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator^(strong x, strong y) -> strong {
    return static_cast<strong>(static_cast<T>(x) ^ static_cast<T>(y));
  }

  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator%(strong x, J y) -> strong
  requires(issigned == std::is_signed_v<J>)
  {
    if constexpr (sizeof(J) > sizeof(I)) return static_cast<J>(x) % y;
    else return static_cast<strong>(static_cast<I>(x) % static_cast<I>(y));
  }
  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator%(J y, strong x) -> strong
  requires(issigned == std::is_signed_v<J>)
  {
    if constexpr (sizeof(J) > sizeof(I)) return y % static_cast<J>(x);
    else return static_cast<strong>(static_cast<I>(y) % static_cast<I>(x));
  }
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator%(strong x, strong y) -> strong {
    return static_cast<strong>(static_cast<I>(x) % static_cast<I>(y));
  }

  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator+=(strong &x,
             J y) -> strong &requires(issigned == std::is_signed_v<J>) {
    return x = x + y;
  }

  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator+=(strong &x, strong y) -> strong & {
    return x = x + y;
  }
  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator-=(strong &x,
             J y) -> strong &requires(issigned == std::is_signed_v<J>) {
    return x = x - y;
  }

  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator-=(strong &x, strong y) -> strong & {
    return x = x - y;
  }
  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator*=(strong &x,
             J y) -> strong &requires(issigned == std::is_signed_v<J>) {
    return x = x * y;
  }

  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator*=(strong &x, strong y) -> strong & {
    return x = x * y;
  }
  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator/=(strong &x,
             J y) -> strong &requires(issigned == std::is_signed_v<J>) {
    return x = x / y;
  }

  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator/=(strong &x, strong y) -> strong & {
    return x = x / y;
  }
  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator%=(strong &x,
             J y) -> strong &requires(issigned == std::is_signed_v<J>) {
    return x = x % y;
  }

  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator%=(strong &x, strong y) -> strong & {
    return x = x % y;
  }
  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator&=(strong &x,
             J y) -> strong &requires(issigned == std::is_signed_v<J>) {
    return x = x & y;
  }

  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator&=(strong &x, strong y) -> strong & {
    return x = x & y;
  }
  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator|=(strong &x,
             J y) -> strong &requires(issigned == std::is_signed_v<J>) {
    return x = x | y;
  }

  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator|=(strong &x, strong y) -> strong & {
    return x = x | y;
  }
  template <std::integral J>
  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator^=(strong &x,
             J y) -> strong &requires(issigned == std::is_signed_v<J>) {
    return x = x ^ y;
  }

  [[gnu::always_inline, gnu::artificial]] friend inline constexpr auto
  operator^=(strong &x, strong y) -> strong & {
    return x = x ^ y;
  }
  friend auto operator<<(std::ostream &os, strong x) -> std::ostream & {
    return os << static_cast<I>(x);
  }
};
static_assert(++static_cast<IntWrapper<int>::strong>(3) == 4);

using i8 = IntWrapper<signed char>::strong;
using u8 = IntWrapper<unsigned char>::strong;
using Flag8 = IntWrapper<unsigned char, false, 1>::strong;

static_assert(!bool(i8{}));
static_assert(!bool(u8{}));

} // namespace numbers
