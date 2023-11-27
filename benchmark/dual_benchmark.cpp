
#include "include/randdual.hpp"
#include <Containers/Tuple.hpp>
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

namespace {
using benchmark::State;
using poly::math::Dual, poly::math::SquareMatrix, poly::math::URand;

[[gnu::noinline]] void prod(auto &c, const auto &a, const auto &b) {
  c = a * b;
}

template <ptrdiff_t M, ptrdiff_t N> void BM_dualprod(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<Dual<double, M>, N>;
  D a = URand<D>{}(rng0), b = URand<D>{}(rng0), c;
  for (auto _ : state) {
    prod(c, a, b);
    benchmark::DoNotOptimize(c);
  }
}

template <typename T, ptrdiff_t N, bool SIMDArray = false> struct ManualDual {
  T value;
  poly::math::SVector<T, N> partials;
  auto grad() -> poly::math::SVector<T, N> & { return partials; }
};
template <std::floating_point T, ptrdiff_t N, bool SIMDArray>
struct ManualDual<ManualDual<T, N, SIMDArray>, 2, false> {
  using V = ManualDual<T, N, SIMDArray>;
  V value;
  poly::containers::Tuple<V, V> partials{V{}, V{}};
  struct Gradient {
    poly::containers::Tuple<V, V> &partials;
    auto operator[](ptrdiff_t i) -> V & {
      poly::utils::invariant(i == 0 || i == 1);
      if (i == 0) return partials.head;
      return partials.tail.head;
    }
  };
  constexpr auto grad() -> Gradient { return {partials}; }
  [[nodiscard]] constexpr auto grad() const -> std::array<V, 2> {
    return {partials.head, partials.tail.head};
  }
};

template <std::floating_point T, ptrdiff_t N> struct ManualDual<T, N, false> {
  using P = poly::simd::Vec<ptrdiff_t(std::bit_ceil(size_t(N))), T>;
  T value;
  P partials;
  auto grad() -> P & { return partials; }
};
template <std::floating_point T, ptrdiff_t N> struct ManualDual<T, N, true> {
  using P = poly::math::StaticArray<T, 1, N, false>;
  T value;
  P partials;
  auto grad() -> P & { return partials; }
};
template <typename T, ptrdiff_t N, bool B>
[[gnu::always_inline]] constexpr auto operator*(const ManualDual<T, N, B> &a,
                                                const ManualDual<T, N, B> &b)
  -> ManualDual<T, N, B> {
  if constexpr ((!B) && (!std::floating_point<T>)&&(N == 2))
    return {a.value * b.value,
            {a.value * b.grad()[0] + b.value + a.grad()[0],
             a.value * b.grad()[1] + b.value * a.grad()[1]}};
  else return {a.value * b.value, a.value * b.partials + b.value * a.partials};
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

template <ptrdiff_t M, ptrdiff_t N, bool SIMDArray, bool Outer>
auto setup_manual() {
  using D = ManualDual<ManualDual<double, M, SIMDArray>, N, Outer>;
  std::mt19937_64 rng0;
  D a{}, b{}, c{};
  a.value.value = URand<double>{}(rng0);
  b.value.value = URand<double>{}(rng0);
  for (ptrdiff_t j = 0; j < M; ++j) {
    a.value.partials[j] = URand<double>{}(rng0);
    b.value.partials[j] = URand<double>{}(rng0);
  }
  for (ptrdiff_t i = 0; i < N; ++i) {
    a.grad()[i].value = URand<double>{}(rng0);
    b.grad()[i].value = URand<double>{}(rng0);
    for (ptrdiff_t j = 0; j < M; ++j) {
      a.grad()[i].partials[j] = URand<double>{}(rng0);
      b.grad()[i].partials[j] = URand<double>{}(rng0);
    }
  }
  return std::array<D, 3>{a, b, c};
}

template <ptrdiff_t M, ptrdiff_t N>
void BM_dualprod_manual(benchmark::State &state) {
  auto [a, b, c] = setup_manual<M, N, false, true>();
  for (auto _ : state) {
    prod(c, a, b);
    benchmark::DoNotOptimize(c);
  }
}
template <ptrdiff_t M, ptrdiff_t N> void BM_dualprod_simdarray(State &state) {
  auto [a, b, c] = setup_manual<M, N, true, true>();
  for (auto _ : state) {
    prod(c, a, b);
    benchmark::DoNotOptimize(c);
  }
}
template <ptrdiff_t M, ptrdiff_t N>
void BM_dualprod_manual_tuple(State &state) {
  auto [a, b, c] = setup_manual<M, N, false, false>();
  for (auto _ : state) {
    prod(c, a, b);
    benchmark::DoNotOptimize(c);
  }
}
template <ptrdiff_t M, ptrdiff_t N>
void BM_dualprod_simdarray_tuple(benchmark::State &state) {
  auto [a, b, c] = setup_manual<M, N, true, false>();
  for (auto _ : state) {
    prod(c, a, b);
    benchmark::DoNotOptimize(c);
  }
}
} // namespace
BENCHMARK(BM_dualprod<6, 2>);
BENCHMARK(BM_dualprod<7, 2>);
BENCHMARK(BM_dualprod<8, 2>);

BENCHMARK(BM_dualprod_manual<6, 2>);
BENCHMARK(BM_dualprod_manual<7, 2>);
BENCHMARK(BM_dualprod_manual<8, 2>);

BENCHMARK(BM_dualprod_simdarray<6, 2>);
BENCHMARK(BM_dualprod_simdarray<7, 2>);
BENCHMARK(BM_dualprod_simdarray<8, 2>);

BENCHMARK(BM_dualprod_manual_tuple<6, 2>);
BENCHMARK(BM_dualprod_manual_tuple<7, 2>);
BENCHMARK(BM_dualprod_manual_tuple<8, 2>);

BENCHMARK(BM_dualprod_simdarray_tuple<6, 2>);
BENCHMARK(BM_dualprod_simdarray_tuple<7, 2>);
BENCHMARK(BM_dualprod_simdarray_tuple<8, 2>);

