#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifndef USE_MODULE
#include <concepts>
#include <functional>
#include <type_traits>

#include "Math/MatrixDimensions.cxx"
#else
export module ScalarizeViaCast;

import MatDim;
import STL;
#endif

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif
template <typename T> struct ScalarizeViaCast {
  using type = void;
};
template <typename T>
using scalarize_via_cast_t = typename ScalarizeViaCast<T>::type;

template <typename T> struct IsDualImpl : std::false_type {};
template <typename T>
concept IsDual = IsDualImpl<T>::value;

// We want to support casting compressed `Dual` arrays to `double`
// when possible as a performance optimization.
// This is possible with
// 1. Dual<T,N> (+/-) Dual<T,N>
// 2. Dual<T,N> * double or double * Dual<T,N>
// 3. Dual<T,N> / double
// 4. Simple copies
template <typename T> struct ScalarizeEltViaCast {
  using type = void;
};
template <typename T>
using scalarize_elt_cast_t = typename ScalarizeEltViaCast<T>::type;

template <typename T>
concept AdditiveOp =
  std::same_as<T, std::plus<>> || std::same_as<T, std::minus<>>;
template <typename T>
concept MultiplicativeOp =
  std::same_as<T, std::multiplies<>> || std::same_as<T, std::divides<>>;
template <typename T>
concept EltIsDual = IsDual<utils::eltype_t<T>>;

template <typename T>
concept EltCastableDual =
  EltIsDual<T> && std::same_as<scalarize_via_cast_t<T>, double>;

} // namespace math
