#include "include/expm.hpp"

static void BM_expm_dual7x2(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<Dual<double, 7>, 2>;
  SquareMatrix<D> A{SquareDims{math::row(state.range(0))}};
#ifndef NDEBUG
  static_assert(
    std::same_as<math::ElementwiseBinaryOp<math::Array<D, SquareDims<>, true>,
                                           math::Array<D, SquareDims<>, true>,
                                           std::plus<>>,
                 decltype(A + A)>);
  static_assert(
    std::same_as<
      double, math::scalarize_via_cast_t<math::Array<D, SquareDims<>, true>>>);
  static_assert(
    std::same_as<double, math::scalarize_via_cast_t<decltype(A + A)>>);
  static_assert(
    std::same_as<math::ElementwiseBinaryOp<math::Array<D, SquareDims<>, true>,
                                           double, std::multiplies<>>,
                 decltype(A * 2.3)>);
  static_assert(
    std::same_as<double, math::scalarize_via_cast_t<decltype(A * 2.3)>>);
  static_assert(
    std::same_as<double, math::scalarize_via_cast_t<decltype(2.3 * A)>>);
  static_assert(
    math::ScalarizeViaCastTo<math::scalarize_via_cast_t<decltype(view(A))>,
                             decltype(2.3 * A)>());
#endif
  for (auto &&a : A) a = URand<D>{}(rng0);
  for (auto b : state) expbench(A);
}
BENCHMARK(BM_expm_dual7x2)->DenseRange(2, 10, 1);
