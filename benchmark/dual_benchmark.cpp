
#include "include/randdual.hpp"
#include <Math/Array.hpp>
#include <Math/Dual.hpp>
#include <Math/LinearAlgebra.hpp>
#include <Math/Matrix.hpp>
#include <Math/StaticArrays.hpp>
#include <Utilities/Invariant.hpp>
#include <algorithm>
#include <array>
#include <benchmark/benchmark.h>
#include <concepts>
#include <cstdint>
#include <random>
#include <ranges>

using poly::math::Dual, poly::math::SquareMatrix, poly::math::SquareDims,
  poly::math::I, poly::math::URand;

[[gnu::noinline]] void prod(auto &c, const auto &a, const auto &b) {
  c = a * b;
}

static void BM_dual8x2prod(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<Dual<double, 8>, 2>;
  D a = URand<D>{}(rng0), b = URand<D>{}(rng0), c;
  for (auto _ : state) {
    prod(c, a, b);
    benchmark::DoNotOptimize(c);
  }
}
BENCHMARK(BM_dual8x2prod);

template <typename T, ptrdiff_t N, bool SIMDArry = false> struct ManualDual {
  T value;
  poly::math::SVector<T, N> partials;
};

template <std::floating_point T, ptrdiff_t N> struct ManualDual<T, N, false> {
  T value;
  poly::simd::Vec<N, T> partials;
};
template <std::floating_point T, ptrdiff_t N> struct ManualDual<T, N, true> {
  T value;
  poly::math::StaticArray<T, 1, N,
                          alignof(poly::simd::Vec<poly::math::VecLen<N, T>, T>)>
    partials;
};
template <typename T, ptrdiff_t N, bool B>
[[gnu::always_inline]] constexpr auto operator*(const ManualDual<T, N, B> &a,
                                                const ManualDual<T, N, B> &b)
  -> ManualDual<T, N, B> {
  return {a.value * b.value, a.value * b.partials + b.value * a.partials};
}
template <typename T, ptrdiff_t N, bool B>
[[gnu::always_inline]] constexpr auto operator*(const ManualDual<T, N, B> &a,
                                                const T &b)
  -> ManualDual<T, N, B> {
  return {a.value * b, b * a.partials};
}
template <typename T, ptrdiff_t N, bool B>
[[gnu::always_inline]] constexpr auto operator*(const T &a,
                                                const ManualDual<T, N, B> &b)
  -> ManualDual<T, N, B> {
  return {b.value * a, a * b.partials};
}
template <typename T, ptrdiff_t N, bool B>
[[gnu::always_inline]] constexpr auto operator+(const ManualDual<T, N, B> &a,
                                                const ManualDual<T, N, B> &b)
  -> ManualDual<T, N, B> {
  return {a.value + b.value, a.partials + b.partials};
}
template <typename T, ptrdiff_t N, bool B>
[[gnu::always_inline]] constexpr auto operator+(const ManualDual<T, N, B> &a,
                                                const T &b)
  -> ManualDual<T, N, B> {
  return {a.value + b, a.partials};
}
template <typename T, ptrdiff_t N, bool B>
[[gnu::always_inline]] constexpr auto operator+(const T &a,
                                                const ManualDual<T, N, B> &b)
  -> ManualDual<T, N, B> {
  return {b.value + a, b.partials};
}
// template <typename T, ptrdiff_t M, ptrdiff_t N>
// [[gnu::noinline]] void prod_manual(ManualDual<ManualDual<T, M>, N> &c,
//                                    const ManualDual<ManualDual<T, M>, N> &a,
//                                    const ManualDual<ManualDual<T, M>, N> &b)
//                                    {
//   // return {val * other.val, val * other.partials + other.val * partials};
//   c.value= a.value* b.value;
//   c.partials = a.value* b.partials+ b.value* a.partials;
// }

static void BM_dual8x2prod_manual(benchmark::State &state) {
  std::mt19937_64 rng0;
  constexpr ptrdiff_t M = 8;
  constexpr ptrdiff_t N = 2;
  using D = ManualDual<ManualDual<double, M>, N>;
  D a, b, c;
  a.value.value = URand<double>{}(rng0);
  b.value.value = URand<double>{}(rng0);
  for (ptrdiff_t j = 0; j < M; ++j) {
    a.value.partials[j] = URand<double>{}(rng0);
    b.value.partials[j] = URand<double>{}(rng0);
  }
  for (ptrdiff_t i = 0; i < N; ++i) {
    a.partials[i].value = URand<double>{}(rng0);
    b.partials[i].value = URand<double>{}(rng0);
    for (ptrdiff_t j = 0; j < M; ++j) {
      a.partials[i].partials[j] = URand<double>{}(rng0);
      b.partials[i].partials[j] = URand<double>{}(rng0);
    }
  }
  for (auto _ : state) {
    // prod_manual(c, a, b);
    prod(c, a, b);
    benchmark::DoNotOptimize(c);
  }
}
BENCHMARK(BM_dual8x2prod_manual);

static void BM_dual8x2prod_simdarray(benchmark::State &state) {
  std::mt19937_64 rng0;
  constexpr ptrdiff_t M = 8;
  constexpr ptrdiff_t N = 2;
  using D = ManualDual<ManualDual<double, M, true>, N>;
  D a, b, c;
  a.value.value = URand<double>{}(rng0);
  b.value.value = URand<double>{}(rng0);
  for (ptrdiff_t j = 0; j < M; ++j) {
    a.value.partials[j] = URand<double>{}(rng0);
    b.value.partials[j] = URand<double>{}(rng0);
  }
  for (ptrdiff_t i = 0; i < N; ++i) {
    a.partials[i].value = URand<double>{}(rng0);
    b.partials[i].value = URand<double>{}(rng0);
    for (ptrdiff_t j = 0; j < M; ++j) {
      a.partials[i].partials[j] = URand<double>{}(rng0);
      b.partials[i].partials[j] = URand<double>{}(rng0);
    }
  }
  for (auto _ : state) {
    // prod_manual(c, a, b);
    prod(c, a, b);
    benchmark::DoNotOptimize(c);
  }
}
BENCHMARK(BM_dual8x2prod_simdarray);

