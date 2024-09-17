#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifndef USE_MODULE
#include <cstddef>
#include <type_traits>

#else
export module SIMD:Vec;
import STL;
#endif

template <ptrdiff_t W, typename T>
using Vec_ [[gnu::vector_size(W * sizeof(T))]] = T;

#ifdef USE_MODULE
export namespace simd {
#else
namespace simd {
#endif
template <ptrdiff_t W, typename T>
using Vec = std::conditional_t<W == 1, T, Vec_<W, T>>;
#ifdef __x86_64__
#ifdef __AVX512F__
inline constexpr ptrdiff_t REGISTERS = 32;
inline constexpr ptrdiff_t VECTORWIDTH = 64;
#else // not __AVX512F__
inline constexpr ptrdiff_t REGISTERS = 16;
#ifdef __AVX__
inline constexpr ptrdiff_t VECTORWIDTH = 32;
#else  // no AVX
inline constexpr ptrdiff_t VECTORWIDTH = 16;
#endif // no AVX
#endif
#else  // not __x86_64__
inline constexpr ptrdiff_t REGISTERS = 32;
inline constexpr ptrdiff_t VECTORWIDTH = 16;
#endif // __x86_64__

} // namespace simd
