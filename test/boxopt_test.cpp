#include "Math/Array.hpp"
#include "Math/BoxOpt.hpp"
#include "Math/Exp.hpp"
#include <gtest/gtest.h>

constexpr auto fcore(auto u1, auto u2) {
  return (2 * u1 + u2 + u1 * u2) / (u1 * u2);
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
  poly::math::Vector<double> x0{-3.4, -3.4}; // approx 2 after transform
  constexpr auto fsoft = [](auto x) {
    auto u0 = x[0];
    auto u1 = x[1];
    return fcore(u0, u1) + 0.125 * poly::math::softplus(8.0 * gcore(u0, u1));
  };
  poly::utils::OwningArena<> arena;
  double opt0 = poly::math::minimize(&arena, x0, box, fsoft);
  double u0 = poly::math::BoxTransformVector{x0.view(), box}[0];
  double u1 = poly::math::BoxTransformVector{x0.view(), box}[1];
  EXPECT_LT(std::abs(3.4567718680186568 - u0), 1e-6);
  EXPECT_LT(std::abs(7.799906157078232 - u1), 1e-6);
  box.decreaseUpperBound(x0, 0, 3);
  double opt1 = poly::math::minimize(&arena, x0, box, fsoft);
  EXPECT_LT(opt0, opt1);
  double u01 = poly::math::BoxTransformVector{x0.view(), box}[0];
  double u11 = poly::math::BoxTransformVector{x0.view(), box}[1];
  EXPECT_EQ(u01, 3.0);
  EXPECT_LT(std::abs(9.132451832031007 - u11), 1e-6);
};
