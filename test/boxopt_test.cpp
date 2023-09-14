#include "Math/Array.hpp"
#include "Math/BoxOpt.hpp"
#include "Math/BoxOptInt.hpp"
#include "Math/Exp.hpp"
#include <gtest/gtest.h>
#include <iostream>

constexpr auto fcore(auto u1, auto u2) {
  return (2.0 * u1 + u2 + u1 * u2) / (u1 * u2);
}
constexpr auto gcore(auto u1, auto u2) { return u1 + u1 * u2 - 31; }

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(BoxOptTest, BasicAssertions) {
  // opt1 = BoxOptNewton.minimize(fsoft, (2, 2), (1, 1), (32, 32))
  // @test SVector(opt1) ≈ SVector(3.4567718680186568, 7.799906157078232) rtol =
  //   1e-6
  // opt2 = BoxOptNewton.minimize(fsoft, (2, 2), (1, 1), (3, 32))
  // @test SVector(opt2) ≈ SVector(3.0, 9.132451832031007) rtol = 1e-6
  poly::math::BoxTransform box(2, 1, 32);
  poly::math::MutPtrVector<double> x0{box.getRaw()};
  x0 << -3.4; // approx 2 after transform
  constexpr auto fsoft = [](auto x) {
    auto u0 = x[0];
    auto u1 = x[1];
    return fcore(u0, u1) + 0.25 * poly::math::softplus(8.0 * gcore(u0, u1));
  };
  poly::utils::OwningArena<> arena;
  double opt0 = poly::math::minimize(&arena, box, fsoft);
  double u0 = box.transformed()[0];
  double u1 = box.transformed()[1];
  std::cout << "u0 = " << u0 << "; u1 = " << u1 << '\n';
  EXPECT_LT(std::abs(3.45128 - u0), 1e-3);
  EXPECT_LT(std::abs(7.78878 - u1), 1e-3);
  box.decreaseUpperBound(0, 3);
  double opt1 = poly::math::minimize(&arena, box, fsoft);
  EXPECT_LT(opt0, opt1);
  double u01 = box.transformed()[0];
  double u11 = box.transformed()[1];
  std::cout << "u01 = " << u01 << "; u11 = " << u11 << '\n';
  EXPECT_EQ(u01, 3.0);
  EXPECT_LT(std::abs(9.10293 - u11), 1e-3);

  poly::math::Vector<int32_t> r{0, 0};
  double opti = poly::math::minimizeIntSol(&arena, r, 1, 32, fsoft);
  EXPECT_GT(opti, opt1);
  EXPECT_EQ(r[0], 3);
  EXPECT_EQ(r[1], 9);
};
