#include <gtest/gtest.h>
#ifndef USE_MODULE
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "Alloc/Arena.cxx"
#include "Math/Array.cxx"
#include "Math/ArrayConcepts.cxx"
#include "Math/BoxOpt.cxx"
#include "Math/BoxOptInt.cxx"
#include "Math/Dual.cxx"
#include "Math/ElementarySIMD.cxx"
#include "Math/ManagedArray.cxx"
#else

import Arena;
import Array;
import ArrayConcepts;
import BoxOpt;
import BoxOptInt;
import Dual;
import Elementary;
import ManagedArray;
import STL;
#endif

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
  math::BoxTransform box(2, 1, 32);
  EXPECT_EQ(box.getLowerBounds().size(), 2);
  EXPECT_EQ(box.getUpperBounds().size(), 2);
  math::BoxTransformVector<math::MutPtrVector<double>> trf{box.transformed()};
  trf[1] = 3;
  box.transformed()[0] = 4;
  EXPECT_NEAR(box.transformed()[0], 4.0, 1e-14);
  EXPECT_NEAR(box.transformed()[1], 3.0, 1e-14);
  math::MutPtrVector<double> x0{box.getRaw()};
  x0 << -3.4; // approx 2 after transform
  constexpr auto fsoft = [](auto x) {
    auto u0 = x[0];
    auto u1 = x[1];
    return fcore(u0, u1) + 0.25 * math::softplus(8.0 * gcore(u0, u1));
  };
  alloc::OwningArena<> arena;
  double opt0 = math::minimize(&arena, box, fsoft);
  double u0 = box.transformed()[0];
  double u1 = box.transformed()[1];
  std::cout << "u0 = " << u0 << "; u1 = " << u1 << '\n';
  EXPECT_LT(std::abs(3.45128 - u0), 1e-3);
  EXPECT_LT(std::abs(7.78878 - u1), 1e-3);
  box.decreaseUpperBound(0, 3);
  double opt1 = math::minimize(&arena, box, fsoft);
  EXPECT_LT(opt0, opt1);
  double u01 = box.transformed()[0];
  double u11 = box.transformed()[1];
  std::cout << "u01 = " << u01 << "; u11 = " << u11 << '\n';
  EXPECT_EQ(u01, 3.0);
  EXPECT_LT(std::abs(9.09724 - u11), 1e-3);

  math::Vector<int32_t> r{std::array{0, 0}};
  double opti = math::minimizeIntSol(&arena, r, 1, 32, fsoft);
  EXPECT_GT(opti, opt1);
  EXPECT_EQ(r[0], 3);
  EXPECT_EQ(r[1], 9);
};

// constexpr int32_t KiB = 1 << 10;
constexpr int32_t MiB = 1 << 20;
constexpr int32_t GiB = 1 << 30;
// constexpr int32_t L1c = 32 * KiB;
constexpr int32_t L2c = 1 * MiB;
constexpr int32_t L3c = 14417920;
// constexpr double L1b = 430.0 * GiB;
constexpr double L2b = 150.0 * GiB;
constexpr double L3b = 40.0 * GiB;
constexpr double RAMb = 15.0 * GiB;

constexpr int32_t m_r = 24;
constexpr int32_t n_r = 9;

constexpr auto cld(int32_t n, int32_t d) -> int32_t { return (n + d - 1) / d; }

auto l1_use(auto k_c) { return (m_r + n_r) * k_c + m_r * n_r; }
auto l2_use(auto m_c, auto k_c) { return (m_c + n_r) * k_c + m_c * n_r; }
auto l3_use(auto m_c, auto k_c, auto n_c) {
  return (m_c + n_c) * k_c + m_c * n_c;
}

struct MatOpt {
  double KN;
  double MKN;
  constexpr MatOpt(int32_t M, int32_t K, int32_t N)
    : KN(double(K) * double(N)), MKN(double(M) * KN) {}
  auto ram_to_l3_datavolume(auto k_c, auto n_c) const {
    // Our l3 multiplies C[m_c, n_c] = A[m_c, k_c] * B[k_c, n_c]
    // Here, we consider the tiling loops that iterate over all three
    // for (auto n_c : Ntiles)
    //   for (auto k_c : Ktiles)
    //     for (auto m_c : Mtiles)
    // dvA = m_c * k_c
    // dvB = k_c * n_c;
    // dvC = m_c * n_c;
    // freqA = (M/m_c) * (K/k_c) * (N/n_c)
    // freqB = (K/k_c) * (N/n_c)
    // freqC = (M/m_c) * (K/k_c) * (N/n_c) * 2 // load & store
    // totalA = M*K*(N/n_c)
    // totalB = K*N
    // totalC = M*(K/k_c)*N * 2 // load & store
    return MKN / n_c + KN + (2 * MKN) / k_c;
  }
  auto l3_to_l2_datavolume(auto m_c, auto k_c, auto n_c) const {
    // Our l2 multiplies C[m_c, n_r] = A[m_c, k_c] * B[k_c, n_r]
    // for (auto n_r : n_c)
    // dvA = m_c * k_c
    // dvB = k_c * n_r;
    // dvC = m_c * n_r;
    // baseFreq = (M / m_c) * (K / k_c) * (N / n_c)
    // freqA = baseFreq
    // freqB = baseFreq * (n_c / n_r)
    // freqC = baseFreq * (n_c / n_r) * 2 // load & store
    // totalA = M*K*(N/n_c)
    // totalB = (M / m_c) * K * N
    // totalC = M * (K / k_c) * N * 2
    return MKN / n_c + MKN / m_c + (2 * MKN) / k_c;
  }
  auto l2_to_l1_datavolume(auto m_c, auto k_c) const {
    // Our l1 multiplies C[m_r, n_r] = A[m_r, k_c] * B[k_c, n_r]
    // for (auto m_r : m_c)
    // dvA = m_r * k_c
    // dvB = k_c * n_r;
    // dvC = m_r * n_r;
    // baseFreq = (M / m_c) * (K / k_c) * (N / n_r)
    // freqA = baseFreq * (m_c / m_r)
    // freqB = baseFreq
    // freqC = baseFreq * (m_c / m_r) * 2 // load & store
    // totalA = M*K*(N/n_r)
    // totalB = (M / m_c) * K * N
    // totalC = M * (K / k_c) * N * 2
    return (MKN / n_r) + MKN / m_c + (2 * MKN) / k_c;
  }

  inline auto operator()(const math::AbstractVector auto &x) const {
    auto m_c = x[0] * m_r;
    auto k_c = x[1];
    auto n_c = x[2] * n_r;
    // TODO: smarter penalty scaling
    auto violation_penalty =
      math::smax<>(0.0, l2_use(m_c, k_c) - (0.9 / sizeof(double)) * L2c) +
      math::smax<>(0.0, l3_use(m_c, k_c, n_c) - (0.9 / sizeof(double)) * L3c);
    auto r_to_l3 = ram_to_l3_datavolume(k_c, n_c) * (sizeof(double) / RAMb);
    auto l3_to_l2 = l3_to_l2_datavolume(m_c, k_c, n_c) * (sizeof(double) / L3b);
    auto l2_to_l1 = l2_to_l1_datavolume(m_c, k_c) * (sizeof(double) / L2b);
    auto res =
      math::smax<>(r_to_l3, l3_to_l2, l2_to_l1) + 1e3 * violation_penalty;
    // std::cout << "m_c = " << math::value(1.0 * m_c)
    //           << "\nk_c = " << math::value(1.0 * k_c)
    //           << "\nn_c = " << math::value(1.0 * n_c) << "\nres = " <<
    //           res
    //           << "\n";
    return res;
    // return math::smax<>(r_to_l3, l3_to_l2, l2_to_l1) +
    //        1000.0 * violation_penalty;
  }
};

auto optimizeFloat(int32_t M, int32_t K, int32_t N) -> std::array<double, 4> {

  math::BoxTransform box(std::array<int32_t, 3>{1, 1, 1},
                         std::array<int32_t, 3>{int32_t(cld(M, m_r)),
                                                int32_t(K),
                                                int32_t(cld(N, n_r))});
  { // init, we set `m_c = 3*m_r` and then use l2 and l3 sizes for rest
    box.transformed()[0] = 8;
    double m_c = 8 * m_r,
           k_c = std::min(double(K), (0.7 * L2c) / (sizeof(double) * m_c));
    box.transformed()[1] = k_c;
    double n_c =
      std::min(double(N), (0.7 * L3c) / (sizeof(double) * k_c)) / n_r;
    box.transformed()[2] = n_c;
  }

  alloc::OwningArena<> arena;
  double opt = math::minimize(&arena, box, MatOpt{M, K, N});
  return {opt, m_r * box.transformed()[0], box.transformed()[1],
          n_r * box.transformed()[2]};
}
auto optimize(int32_t M, int32_t K, int32_t N) -> std::array<int32_t, 3> {

  math::BoxTransform box(std::array<int32_t, 3>{1, 1, 1},
                         std::array<int32_t, 3>{int32_t(cld(M, m_r)),
                                                int32_t(K),
                                                int32_t(cld(N, n_r))});
  { // init, we set `m_c = 3*m_r` and then use l2 and l3 sizes for rest
    box.transformed()[0] = 4;
    double m_c = 4 * m_r,
           k_c = std::min(double(K), (0.7 * L2c) / (sizeof(double) * m_c));
    box.transformed()[1] = k_c;
    double n_c =
      std::min(double(N), (0.7 * L3c) / (sizeof(double) * k_c)) / n_r;
    box.transformed()[2] = n_c;
  }

  alloc::OwningArena<> arena;
  math::Vector<int32_t> r{math::length(3)};
  math::minimizeIntSol(&arena, r, box, MatOpt{M, K, N});
  return {m_r * r[0], r[1], n_r * r[2]};
}

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(BoxOptMatmulTest, BasicAssertions) {
  int32_t M = 1000, K = 2000, N = 1000;
  auto [opt, m_cf, k_cf, n_cf] = optimizeFloat(M, K, N);
  std::cout << "opt result = " << opt << "\nm_cf = " << m_cf
            << "\nk_cf = " << k_cf << "\nn_cf = " << n_cf << "\n";
  // auto [m_c, k_c, n_c] = optimize(M, K, N);
  // EXPECT_EQ(m_c, 144);
  // EXPECT_EQ(k_c, 762);
  // EXPECT_EQ(n_c, 1008);
}
