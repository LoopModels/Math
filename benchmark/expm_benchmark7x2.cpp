#include "include/expm.hpp"

static void BM_expm_dual7x2(benchmark::State &state) {
  std::mt19937_64 rng0;
  using D = Dual<Dual<double, 7>, 2>;
  SquareMatrix<D> A{SquareDims{{state.range(0)}}};
  static_assert(
    std::same_as<poly::math::ElementwiseBinaryOp<
                   poly::math::Array<D, SquareDims<>, true>,
                   poly::math::Array<D, SquareDims<>, true>, std::plus<>>,
                 decltype(A + A)>);
  static_assert(
    std::same_as<double, poly::math::scalarize_via_cast_t<
                           poly::math::Array<D, SquareDims<>, true>>>);
  static_assert(
    std::same_as<double, poly::math::scalarize_via_cast_t<decltype(A + A)>>);
  static_assert(
    std::same_as<
      poly::math::ElementwiseBinaryOp<poly::math::Array<D, SquareDims<>, true>,
                                      double, std::multiplies<>>,
      decltype(A * 2.3)>);
  static_assert(
    std::same_as<double, poly::math::scalarize_via_cast_t<decltype(A * 2.3)>>);
  static_assert(
    std::same_as<double, poly::math::scalarize_via_cast_t<decltype(2.3 * A)>>);
  static_assert(poly::math::ScalarizeViaCastTo<
                poly::math::scalarize_via_cast_t<decltype(view(A))>,
                decltype(2.3 * A)>());
  for (auto &&a : A) a = URand<D>{}(rng0);
  for (auto b : state) expbench(A);
}
BENCHMARK(BM_expm_dual7x2)->DenseRange(2, 10, 1);
