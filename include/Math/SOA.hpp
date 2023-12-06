#pragma once
#ifndef SOA_hpp_INCLUDED
#define SOA_hpp_INCLUDED

#include "Alloc/Mallocator.hpp"
#include "Containers/Tuple.hpp"
#include "Math/Matrix.hpp"
#include "Math/MatrixDimensions.hpp"
#include <memory>
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
template <typename T, typename... Ts> struct CumSizeOf<0, Types<T, Ts...>> {
  static constexpr size_t value = 0;
};
template <> struct CumSizeOf<0, Types<>> {
  static constexpr size_t value = 0;
};
template <size_t I, typename T>
inline constexpr size_t CumSizeOf_v = CumSizeOf<I, TupleTypes_t<T>>::value;

/// requires 16 byte alignment of allocated pointer
template <typename T, typename S = ptrdiff_t,
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
  struct Reference {
    char *ptr;
    ptrdiff_t stride;
    ptrdiff_t i;
    operator T() const {
      char *p = std::assume_aligned<16>(ptr);
      return T(*reinterpret_cast<std::tuple_element_t<II, T> *>(
        p + CumSizeOf_v<II, T> * stride +
        sizeof(std::tuple_element_t<II, T>) * i)...);
    }
    template <size_t I> void assign(const T &x) {
      char *p = std::assume_aligned<16>(ptr);
      *reinterpret_cast<std::tuple_element_t<I, T> *>(
        p + CumSizeOf_v<I, T> * stride +
        sizeof(std::tuple_element_t<I, T>) * i) = x.template get<I>();
    }
    auto operator=(const T &x) -> Reference & {
      ((void)assign<II>(x), ...);
      // assign<II...>(x);
      return *this;
    }
  };
  auto operator[](ptrdiff_t i) const -> T {
    char *p = std::assume_aligned<16>(data);
    ptrdiff_t stride = capacity(sz);
    return T(*reinterpret_cast<std::tuple_element_t<II, T> *>(
      p + CumSizeOf_v<II, T> * stride +
      sizeof(std::tuple_element_t<II, T>) * i)...);
  }
  auto operator[](ptrdiff_t i) -> Reference { return {data, capacity(sz), i}; }
  static constexpr auto totalSizePer() -> size_t {
    return CumSizeOf_v<sizeof...(II), T>;
  }
};

template <typename T, typename S = ptrdiff_t,
          typename C =
            std::conditional_t<MatrixDimension<S>, CapacityCalculators::Length,
                               CapacityCalculators::NextPow2>,
          typename TT = TupleTypes_t<T>,
          typename II = std::make_index_sequence<std::tuple_size_v<T>>,
          class A = alloc::Mallocator<char>>
struct ManagedSOA : public SOA<T, S, C, TT, II> {
  /// uninitialized allocation
  ManagedSOA(S nsz) {
    this->sz = nsz;
    this->capacity = C{};
    ptrdiff_t stride = this->capacity(this->sz);
    this->data = A::allocate(stride * this->totalSizePer());
    // this->data =
    //   A::allocate(stride * this->totalSizePer(), std::align_val_t{16});
  }
  ManagedSOA(std::type_identity<T>, S nsz) : ManagedSOA(nsz) {}
  ~ManagedSOA() {
    ptrdiff_t stride = this->capacity(this->sz);
    A::deallocate(this->data, stride * this->totalSizePer());
  }
};
template <typename T, typename S>
ManagedSOA(std::type_identity<T>, S) -> ManagedSOA<T, S>;

} // namespace poly::math
#endif // SOA_hpp_INCLUDED
