
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
  poly::math::StaticArray<T, 1, N, false> partials;
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

template <ptrdiff_t M, ptrdiff_t N, bool SIMDArray> auto setup_manual() {
  using D = ManualDual<ManualDual<double, M, SIMDArray>, N>;
  std::mt19937_64 rng0;
  D a{}, b{}, c{};
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
  return std::array<D, 3>{a, b, c};
}

static void BM_dual8x2prod_manual(benchmark::State &state) {
  auto [a, b, c] = setup_manual<8, 2, false>();
  for (auto _ : state) {
    prod(c, a, b);
    benchmark::DoNotOptimize(c);
  }
}
BENCHMARK(BM_dual8x2prod_manual);

static void BM_dual8x2prod_simdarray(benchmark::State &state) {
  auto [a, b, c] = setup_manual<8, 2, true>();
  for (auto _ : state) {
    prod(c, a, b);
    benchmark::DoNotOptimize(c);
  }
}
BENCHMARK(BM_dual8x2prod_simdarray);

static void BM_dual7x2prod(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<Dual<double, 7>, 2>;
  static_assert(std::same_as<double, poly::math::scalarize_via_cast_to_t<
                                       Dual<Dual<double, 7, true>, 2, true>>>);
  // static_assert(sizeof(D) == sizeof(Dual<Dual<double, 8>, 2>));
  static_assert(poly::utils::Compressible<D>);
  D a = URand<D>{}(rng0), b = URand<D>{}(rng0), c;
  for (auto _ : state) {
    prod(c, a, b);
    benchmark::DoNotOptimize(c);
  }
}
BENCHMARK(BM_dual7x2prod);

static void BM_dual7x2prod_manual(benchmark::State &state) {
  auto [a, b, c] = setup_manual<7, 2, false>();
  for (auto _ : state) {
    prod(c, a, b);
    benchmark::DoNotOptimize(c);
  }
}
BENCHMARK(BM_dual7x2prod_manual);

static void BM_dual7x2prod_simdarray(benchmark::State &state) {
  auto [a, b, c] = setup_manual<7, 2, true>();
  for (auto _ : state) {
    prod(c, a, b);
    benchmark::DoNotOptimize(c);
  }
}
BENCHMARK(BM_dual7x2prod_simdarray);

static void BM_dual6x2prod(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<Dual<double, 6>, 2>;
  // static_assert(sizeof(D) == sizeof(Dual<Dual<double, 8>, 2>));
  static_assert(poly::utils::Compressible<D>);
  D a = URand<D>{}(rng0), b = URand<D>{}(rng0), c;
  for (auto _ : state) {
    prod(c, a, b);
    benchmark::DoNotOptimize(c);
  }
}
BENCHMARK(BM_dual6x2prod);

static void BM_dual6x2prod_manual(benchmark::State &state) {
  auto [a, b, c] = setup_manual<6, 2, false>();
  for (auto _ : state) {
    prod(c, a, b);
    benchmark::DoNotOptimize(c);
  }
}
BENCHMARK(BM_dual6x2prod_manual);

static void BM_dual6x2prod_simdarray(benchmark::State &state) {
  auto [a, b, c] = setup_manual<6, 2, true>();
  for (auto _ : state) {
    prod(c, a, b);
    benchmark::DoNotOptimize(c);
  }
}
BENCHMARK(BM_dual6x2prod_simdarray);

