#include "Math/Constructors.hpp"
#include "Math/Dual.hpp"

#include <gtest/gtest.h>
#include <random>

using namespace poly::math;

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(DualTest, BasicAssertions) {
  OwningArena arena;

  std::mt19937 gen(0);
  std::uniform_real_distribution<double> dist(-1, 1);
  SquareMatrix<double> A(15);
  Vector<double> x(15);
  for (auto &a : A) a = dist(gen);
  for (auto &xx : x) xx = dist(gen);
  SquareMatrix<double> B = A + A.transpose();
  const auto halfquadform = [&](const auto &y) {
    return 0.5 * (y.transpose() * (B * y));
  };
  Vector<double> g = B * x;
  auto f = halfquadform(x);

  auto [fx, gx] = gradient(&arena, x, halfquadform);
  auto [fxx, gxx, hxx] = hessian(&arena, x, halfquadform);
  EXPECT_TRUE(std::abs(fx - f) < 1e-10);
  EXPECT_TRUE(std::abs(fxx - f) < 1e-10);
  EXPECT_TRUE(norm2(g - gx) < 1e-10);
  EXPECT_TRUE(norm2(g - gxx) < 1e-10);
  for (ptrdiff_t i = 1; i < hxx.numRow(); ++i)
    for (ptrdiff_t j = 0; j < i; ++j) hxx(i, j) = hxx(j, i);
  // std::cout << "B = " << B << "\nhxx = " << hxx << std::endl;
  EXPECT_TRUE(norm2(B - hxx) < 1e-10);
};
