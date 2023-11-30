#pragma once
#ifndef SOA_hpp_INCLUDED
#define SOA_hpp_INCLUDED

#include "Containers/Tuple.hpp"
#include "Math/MatrixDimensions.hpp"
#include <Matrix>
#include <type_traits>
#include <utility>
namespace poly::math {

template <typename... T> struct Types {};

static_assert(std::tuple_size_v<containers::Tuple<int, double>> == 2);

template <typename T,
          typename I = std::make_index_sequence<std::tuple_size_v<T>>>
struct TupleTypes {
  using type = void;
};

template <typename T, size_t... I>
struct TupleTypes<T, std::index_sequence<I...>> {
  using type = Types<std::tuple_element_t<I, T>...>;
};

template <typename T> using TupleTypes_t = typename TupleTypes<T>::type;

static_assert(std::is_same_v<TupleTypes_t<containers::Tuple<int, double>>,
                             Types<int, double>>);

template <typename T, typename U = TupleTypes<T>::type>
struct CanConstructFromMembers {
  static constexpr bool value = false;
};
template <typename T, typename... Elts>
struct CanConstructFromMembers<T, Types<Elts...>> {
  static constexpr bool value = std::is_constructible_v<T, Elts...>;
};

static_assert(CanConstructFromMembers<containers::Tuple<int, double>>::value);

namespace CapacityCalculators {

struct Length {
  constexpr auto operator()(auto sz) -> ptrdiff_t { return ptrdiff_t(sz); }
};
struct NextPow2 {
  constexpr auto operator()(auto sz) -> ptrdiff_t {
    return ptrdiff_t(std::bit_ceil(size_t(ptrdiff_t(sz))));
  }
};

} // namespace CapacityCalculators

template <size_t I, typename T> struct CumSizeOf {
  static constexpr size_t value = 0;
};

template <size_t I, typename T, typename... Ts>
struct CumSizeOf<I, Types<T, Ts...>> {
  static constexpr size_t value =
    sizeof(T) + CumSizeOf<I - 1, Types<Ts...>>::value;
};
template <typename... Ts> struct CumSizeOf<0, Types<Ts...>> {
  static constexpr size_t value = 0;
};
template <size_t I, typename T>
inline constexpr size_t CumSizeOf_v = CumSizeOf<I, T>::value;

template <typename T, typename S,
          typename C =
            std::conditional_t<MatrixDimension<S>, CapacityCalculators::Length,
                               CapacityCalculators::NextPow2>,
          typename TT = TupleTypes_t<T>,
          typename II = std::make_index_sequence<std::tuple_size_v<T>>>
struct SOA {};
template <typename T, typename S, typename C, typename... Elts, size_t... II>
requires(CanConstructFromMembers<T>::value)
struct SOA<T, S, C, Types<Elts...>, std::index_sequence<II...>> {
  char *data;
  [[no_unique_address]] S sz;
  [[no_unique_address]] C capacity;
  auto operator[](ptrdiff_t i) const -> T {
    ptrdiff_t stride = capacity(sz);
    return T(*reinterpret_cast<std::tuple_element_t<II, T> *>(
      reinterpret_cast<unsigned char *>(data) + CumSizeOf_v<II, T> * stride +
      sizeof(std::tuple_element_t<II, T>) * i)...);
  }
};

} // namespace poly::math
#endif // SOA_hpp_INCLUDED
