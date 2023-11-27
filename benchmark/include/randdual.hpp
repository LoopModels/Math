#pragma once
#ifndef randdual_hpp_INCLUDED
#define randdual_hpp_INCLUDED

#include <Math/Dual.hpp>
#include <cstddef>
#include <random>

namespace poly::math {
template <class T> struct URand {};

template <class T, ptrdiff_t N> struct URand<Dual<T, N>> {
  auto operator()(std::mt19937_64 &rng) -> Dual<T, N> {
    Dual<T, N> x{URand<T>{}(rng)};
    for (size_t i = 0; i < N; ++i) x.gradient()[i] = URand<T>{}(rng);
    return x;
  }
};
template <> struct URand<double> {
  auto operator()(std::mt19937_64 &rng) -> double {
    return std::uniform_real_distribution<double>(-2, 2)(rng);
  }
};
} // namespace poly::math

#endif // randdual_hpp_INCLUDED
