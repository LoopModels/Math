#include "Containers/TinyVector.hpp"
#include "Math/Array.hpp"
#include "Math/Constructors.hpp"
#include "Math/Dual.hpp"
#include "Math/Exp.hpp"
#include "Math/LinearAlgebra.hpp"
#include "Math/Matrix.hpp"
#include "Utilities/TypePromotion.hpp"
#include <gtest/gtest.h>
#include <random>

using namespace poly::math;
using poly::utils::eltype_t, poly::math::transpose;

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(DualTest, BasicAssertions) {
  OwningArena arena;

  std::mt19937 gen(0);
  std::uniform_real_distribution<double> dist(-1, 1);
  SquareMatrix<double> A(15);
  Vector<double> x(15);
  for (auto &a : A) a = dist(gen);
  for (auto &xx : x) xx = dist(gen);
  SquareMatrix<double> B = A + A.t();
  const auto halfquadform = [&](const auto &y) {
    return 0.5 * ((y * B) * transpose(y));
  };
  Vector<double> g = x * B;
  auto f = halfquadform(x);

  auto [fx, gx] = gradient(&arena, x, halfquadform);
  auto [fxx, gxx, hxx] = hessian(&arena, x, halfquadform);
  EXPECT_TRUE(std::abs(fx - f) < 1e-10);
  EXPECT_TRUE(std::abs(fxx - f) < 1e-10);
  EXPECT_TRUE(norm2(g - gx) < 1e-10);
  EXPECT_TRUE(norm2(g - gxx) < 1e-10);
  std::cout << "g = " << g << "\ngxx = " << gxx << '\n';
  std::cout << "B = " << B << "\nhxx = " << hxx << '\n';
  for (ptrdiff_t i = 0; i < hxx.numRow(); ++i)
    for (ptrdiff_t j = i + 1; j < hxx.numCol(); ++j) hxx[i, j] = hxx[j, i];
  std::cout << "hxx = " << hxx << '\n';
  EXPECT_TRUE(norm2(B - hxx) < 1e-10);
};

template <typename T>
constexpr void evalpoly(MutSquarePtrMatrix<T> B, MutSquarePtrMatrix<T> A,
                        SquarePtrMatrix<T> C, const auto &p) {
  ptrdiff_t N = p.size();
  invariant(N > 0);
  invariant(ptrdiff_t(B.numRow()), ptrdiff_t(C.numRow()));
  if (N & 1) std::swap(A, B);
  B << p[0] * C + p[1] * I;
  for (ptrdiff_t i = 2; i < N; ++i) {
    std::swap(A, B);
    B << A * C + p[i] * I;
  }
}

template <AbstractMatrix T> constexpr auto opnorm1(const T &A) {
  using S = decltype(value(std::declval<eltype_t<T>>()));
  auto [M, N] = shape(A);
  invariant(M > 0);
  invariant(N > 0);
  S a{};
  for (ptrdiff_t n = 0; n < N; ++n) {
    S s{};
    for (ptrdiff_t m = 0; m < M; ++m) s += std::abs(value(A[m, n]));
    a = std::max(a, s);
  }
  return a;
}

/// computes ceil(log2(x)) for x >= 1
constexpr auto log2ceil(double x) -> unsigned {
  invariant(x >= 1);
  uint64_t u = std::bit_cast<uint64_t>(x) - 1;
  return (u >> 52) - 1022;
}

template <typename T> constexpr void expmimpl(MutSquarePtrMatrix<T> A) {
  ptrdiff_t n = ptrdiff_t(A.numRow()), s = 0;
  SquareMatrix<T> A2{SquareDims<>{{n}}}, U_{SquareDims<>{{n}}};
  MutSquarePtrMatrix<T> U{U_};
  if (double nA = opnorm1(A); nA <= 0.015) {
    A2 << A * A;
    U << A * (A2 + 60.0 * I);
    A << 12.0 * A2 + 120.0 * I;
  } else {
    SquareMatrix<T> B{SquareDims<>{{n}}};
    if (nA <= 2.1) {
      A2 << A * A;
      poly::containers::TinyVector<double, 5> p0, p1;
      if (nA > 0.95) {
        p0 = {1.0, 3960.0, 2162160.0, 302702400.0, 8821612800.0};
        p1 = {90.0, 110880.0, 3.027024e7, 2.0756736e9, 1.76432256e10};
      } else if (nA > 0.25) {
        p0 = {1.0, 1512.0, 277200.0, 8.64864e6};
        p1 = {56.0, 25200.0, 1.99584e6, 1.729728e7};
      } else {
        p0 = {1.0, 420.0, 15120.0};
        p1 = {30.0, 3360.0, 30240.0};
      }
      evalpoly(B, U, A2, p0);
      U << A * B;
      evalpoly(A, B, A2, p1);
    } else {
      // s = std::max(unsigned(std::ceil(std::log2(nA / 5.4))), 0);
      s = nA > 5.4 ? log2ceil(nA / 5.4) : 0;
      if (s & 1) {       // we'll swap `U` and `A` an odd number of times
        std::swap(A, U); // so let them switch places
        A << U * exp2(-s);
      } else if (s > 0) A *= exp2(-s);
      A2 << A * A;
      // here we take an estrin (instead of horner) approach to cut down flops
      SquareMatrix<T> A4{A2 * A2}, A6{A2 * A4};
      B << A6 * (A6 + 16380 * A4 + 40840800 * A2) +
             (33522128640 * A6 + 10559470521600 * A4 + 1187353796428800 * A2) +
             32382376266240000 * I;
      U << A * B;
      A << A6 * (182 * A6 + 960960 * A4 + 1323241920 * A2) +
             (670442572800 * A6 + 129060195264000 * A4 +
              7771770303897600 * A2) +
             64764752532480000 * I;
    }
  }
  poly::containers::tie(A, U) << poly::containers::Tuple(A + U, A - U);
  LU::ldiv(U, MutPtrMatrix<T>(A));
  for (; s--; std::swap(A, U)) U << A * A;
}

template <typename T> constexpr auto expm(SquarePtrMatrix<T> A) {
  SquareMatrix<T> V{SquareDims{A.numRow()}};
  V << A;
  expmimpl(V);
  return V;
}
constexpr auto dualDeltaCmp(double x, double y) -> bool { return x < y; }
template <typename T, ptrdiff_t N>
constexpr auto dualDeltaCmp(Dual<T, N> x, double y) -> bool {
  if (!dualDeltaCmp(x.value(), y)) return false;
  for (ptrdiff_t i = 0; i < N; ++i)
    if (!dualDeltaCmp(x.gradient()[i], y)) return false;
  return true;
}

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(ExpMatTest, BasicAssertions) {
  SquareMatrix<double> A(4);
  A[0, 0] = 0.13809508135032297;
  A[0, 1] = -0.10597225613986219;
  A[0, 2] = -0.5623996136438215;
  A[0, 3] = 1.099556072129511;
  A[1, 0] = 0.7571409301354933;
  A[1, 1] = -0.0725924122459707;
  A[1, 2] = -0.5732592019339723;
  A[1, 3] = 0.4216707913809331;
  A[2, 0] = 0.9551223749499392;
  A[2, 1] = 1.0628072168060698;
  A[2, 2] = -1.0919065664748313;
  A[2, 3] = 0.9125836172498181;
  A[3, 0] = -1.4826804140146677;
  A[3, 1] = 0.6550780207685463;
  A[3, 2] = -0.6227535845719466;
  A[3, 3] = 0.2280514374580733;
  SquareMatrix<double> B(4);
  B[0, 0] = 0.2051199361909877;
  B[0, 1] = -0.049831094437687434;
  B[0, 2] = -0.3980657896416266;
  B[0, 3] = 0.6706580677244947;
  B[1, 0] = 0.14286173961464693;
  B[1, 1] = 0.7798855526203928;
  B[1, 2] = -0.468198822464851;
  B[1, 3] = 0.3841802990849566;
  B[2, 0] = 0.04397695538798724;
  B[2, 1] = 0.7186280674937524;
  B[2, 2] = -0.10256382048628668;
  B[2, 3] = 0.8710856559160713;
  B[3, 0] = -1.2481676162786608;
  B[3, 1] = 0.4472989810307132;
  B[3, 2] = -0.11106692926404803;
  B[3, 3] = 0.3930685232252409;
  EXPECT_LE(norm2(B - expm(A)), 1e-10);

  static_assert(poly::utils::Compressible<Dual<double, 2>>);
  SquareMatrix<Dual<double, 2>> Ad(4);
  Ad[0, 0] = Dual<double, 2>{
    0.13809508135032297,
    SVector<double, 2>{0.23145585885555967, 0.6736099502056541}};
  Ad[0, 1] = Dual<double, 2>{
    -0.10597225613986219,
    SVector<double, 2>{-1.697508191315446, -1.0754726189887889}};
  Ad[0, 2] = Dual<double, 2>{
    -0.5623996136438215,
    SVector<double, 2>{-0.1193943011160536, -0.26291486453222596}};
  Ad[0, 3] =
    Dual<double, 2>{1.099556072129511,
                    SVector<double, 2>{1.387881995203853, 3.1737580685186204}};
  Ad[1, 0] =
    Dual<double, 2>{0.7571409301354933,
                    SVector<double, 2>{-1.7465876925755974, -3.0307435092366}};
  Ad[1, 1] = Dual<double, 2>{
    -0.0725924122459707,
    SVector<double, 2>{0.3308073172514302, 0.07339531064172199}};
  Ad[1, 2] = Dual<double, 2>{
    -0.5732592019339723,
    SVector<double, 2>{-0.6947698845697236, -1.0672845574608438}};
  Ad[1, 3] = Dual<double, 2>{
    0.4216707913809331,
    SVector<double, 2>{-0.37974214526732475, 0.23640427332714647}};
  Ad[2, 0] =
    Dual<double, 2>{0.9551223749499392,
                    SVector<double, 2>{0.6139406820054037, 1.5323159670356596}};
  Ad[2, 1] = Dual<double, 2>{
    1.0628072168060698,
    SVector<double, 2>{-1.9568771179939957, -0.4461533904276679}};
  Ad[2, 2] = Dual<double, 2>{
    -1.0919065664748313,
    SVector<double, 2>{1.1814457776729768, -1.8787170755674358}};
  Ad[2, 3] =
    Dual<double, 2>{0.9125836172498181,
                    SVector<double, 2>{1.2012117345451894, -2.031045206544551}};
  Ad[3, 0] = Dual<double, 2>{
    -1.4826804140146677,
    SVector<double, 2>{1.8971407850439679, 0.05411248868441574}};
  Ad[3, 1] =
    Dual<double, 2>{0.6550780207685463,
                    SVector<double, 2>{-1.121131948533708, 0.3543625073973555}};
  Ad[3, 2] =
    Dual<double, 2>{-0.6227535845719466,
                    SVector<double, 2>{1.195008337302074, -1.4233657256047536}};
  Ad[3, 3] = Dual<double, 2>{
    0.2280514374580733,
    SVector<double, 2>{-1.2001994532706792, 0.03274459682369542}};
  SquareMatrix<Dual<double, 2>> Bd(4);
  Bd[0, 0] = Dual<double, 2>{
    0.20511993619098767,
    SVector<double, 2>{0.09648410552837837, -2.2538795735050865}};
  Bd[0, 1] = Dual<double, 2>{
    -0.04983109443768741,
    SVector<double, 2>{-0.9642251558357876, 0.19359255059179328}};
  Bd[0, 2] = Dual<double, 2>{
    -0.39806578964162675,
    SVector<double, 2>{0.26655194154076056, -0.44550440724595763}};
  Bd[0, 3] =
    Dual<double, 2>{0.6706580677244948,
                    SVector<double, 2>{0.1094503631997486, 2.088335208353239}};
  Bd[1, 0] = Dual<double, 2>{
    0.1428617396146469,
    SVector<double, 2>{-0.5809206484565365, -3.1420312577464244}};
  Bd[1, 1] = Dual<double, 2>{
    0.7798855526203933,
    SVector<double, 2>{-0.1381973048177799, 0.08427369552387146}};
  Bd[1, 2] = Dual<double, 2>{
    -0.46819882246485106,
    SVector<double, 2>{0.0698482347780988, 0.24309332281112794}};
  Bd[1, 3] = Dual<double, 2>{
    0.3841802990849566,
    SVector<double, 2>{-1.4946422537617146, -0.5928606061328459}};
  Bd[2, 0] = Dual<double, 2>{
    0.043976955387987425,
    SVector<double, 2>{-0.1226602268238547, -0.5881911245335547}};
  Bd[2, 1] =
    Dual<double, 2>{0.7186280674937524, SVector<double, 2>{-0.9224382403836392,
                                                           -1.117883803285957}};
  Bd[2, 2] = Dual<double, 2>{
    -0.1025638204862866,
    SVector<double, 2>{0.5621700920192116, -0.7417763253026113}};
  Bd[2, 3] = Dual<double, 2>{
    0.8710856559160713,
    SVector<double, 2>{0.15338641454990598, -0.9911471735749906}};
  Bd[3, 0] = Dual<double, 2>{
    -1.2481676162786608,
    SVector<double, 2>{2.1160878512755934, -0.30475289132078964}};
  Bd[3, 1] =
    Dual<double, 2>{0.4472989810307133, SVector<double, 2>{0.43406926847671146,
                                                           0.2778024646943355}};
  Bd[3, 2] = Dual<double, 2>{
    -0.11106692926404803,
    SVector<double, 2>{0.18787583104988448, -0.09522217722177315}};
  Bd[3, 3] = Dual<double, 2>{
    0.3930685232252409,
    SVector<double, 2>{-0.9491225558721068, -2.5673833776578996}};

  EXPECT_TRUE(dualDeltaCmp(norm2(Bd - expm(Ad)), 1e-10));
  {
    auto x = Bd[3, 3] - 0.35;
    auto y = -1 * (0.35 - Bd[3, 3]);
    EXPECT_NEAR(x.value(), y.value(), 1e-14);
    EXPECT_NEAR(x.gradient()[0], y.gradient()[0], 1e-14);
    EXPECT_NEAR(x.gradient()[1], y.gradient()[1], 1e-14);
  }
  EXPECT_EQ(poly::math::smax(Bd[3, 3], 0.35), poly::math::smax(0.35, Bd[3, 3]));
}
