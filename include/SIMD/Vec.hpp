#pragma once
#ifndef MATH_SIMD_VEC_HPP_INCLUDED
#define MATH_SIMD_VEC_HPP_INCLUDED

namespace poly::simd {
template <ptrdiff_t W, typename T>
using Vec [[gnu::vector_size(W * sizeof(T))]] = T;
} // namespace poly::simd

#endif
