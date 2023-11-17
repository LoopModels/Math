#pragma once
#ifndef MATH_SIMD_VEC_HPP_INCLUDED
#define MATH_SIMD_VEC_HPP_INCLUDED

namespace poly::simd {
template <ptrdiff_t W, typename T>
using Vec_ [[gnu::vector_size(W * sizeof(T))]] = T;

template <ptrdiff_t W, typename T>
using Vec = std::conditional_t<W == 1, T, Vec_<W, T>>;

} // namespace poly::simd

#endif
