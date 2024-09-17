#include <gtest/gtest.h>
#ifndef USE_MODULE
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

#include "Alloc/Arena.cxx"
#include "Containers/TinyVector.cxx"
#include "Containers/Tuple.cxx"
#include "Math/Array.cxx"
#include "Math/ArrayConcepts.cxx"
#include "Math/Dual.cxx"
#include "Math/ExpressionTemplates.cxx"
#include "Math/LinearAlgebra.cxx"
#include "Math/ManagedArray.cxx"
#include "Math/MatrixDimensions.cxx"
#include "Math/Reductions.cxx"
#include "Math/StaticArrays.cxx"
#include "Math/UniformScaling.cxx"
#include "Utilities/TypeCompression.cxx"
#else

import Arena;
import Array;
import ArrayConcepts;
import Dual;
import ExprTemplates;
import LinearAlgebra;
import ManagedArray;
import MatDim;
import Reductions;
import StaticArray;
import STL;
import TinyVector;
import Tuple;
import TypeCompression;
import UniformScaling;
#endif

using namespace math;
using utils::eltype_t, math::transpose;

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(DualTest, BasicAssertions) {
  alloc::OwningArena arena;

  std::mt19937 gen(0);
  std::uniform_real_distribution<double> dist(-1, 1);
  SquareMatrix<double> A(SquareDims<>{math::row(15)});
  Vector<double> x(length(15));
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
  using S = decltype(extractvalue(std::declval<eltype_t<T>>()));
  auto [M, N] = shape(A);
  invariant(M > 0);
  invariant(N > 0);
  S a{};
  for (ptrdiff_t n = 0; n < N; ++n) {
    S s{};
    for (ptrdiff_t m = 0; m < M; ++m) s += std::abs(extractvalue(A[m, n]));
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
  SquareMatrix<T> A2{SquareDims<>{row(n)}}, U_{SquareDims<>{row(n)}};
  MutSquarePtrMatrix<T> U{U_};
  if (double nA = opnorm1(A); nA <= 0.015) {
    A2 << A * A;
    U << A * (A2 + 60.0 * I);
    A << 12.0 * A2 + 120.0 * I;
  } else {
    SquareMatrix<T> B{SquareDims<>{row(n)}};
    if (nA <= 2.1) {
      A2 << A * A;
      containers::TinyVector<double, 5> p0, p1;
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
  containers::tie(A, U) << containers::Tuple(A + U, A - U);
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
  SquareMatrix<double> A(SquareDims<>{math::row(4)});
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
  SquareMatrix<double> B(SquareDims<>{math::row(4)});
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

  static_assert(utils::Compressible<Dual<double, 2>>);
  SquareMatrix<Dual<double, 2>> Ad(SquareDims<>{math::row(4)});
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
  SquareMatrix<Dual<double, 2>> Bd(SquareDims<>{math::row(4)});
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
  EXPECT_EQ(math::smax(Bd[3, 3], 0.35), math::smax(0.35, Bd[3, 3]));
}

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(IntDivDualTest, BasicAssertions) {

  Dual<double, 2> x{
    -0.1025638204862866,
    SVector<double, 2>{0.5621700920192116, -0.7417763253026113}};
  int32_t a = 4, b = 3;
  Dual<double, 2> y = a / x + b;
  EXPECT_NEAR(y.value(), -36.00010726038452, 1e-14);
  EXPECT_NEAR(y.gradient()[0], -213.76635331423668, 1e-14);
  EXPECT_NEAR(y.gradient()[1], 282.061999181122, 1e-14);

  Dual<Dual<double, 8>, 2> w{
    Dual<double, 8>{
      0.474029877977747,
      SVector<double, 8>{0.3086698530598352, 0.2473835557435392,
                         0.3323869313428053, 0.5171334596793957,
                         0.38702317404526887, 0.00047786444101627357,
                         0.5631545736256198, 0.43922995203510906}},
    SVector<Dual<double, 8>, 2>{
      Dual<double, 8>{
        0.364700717714898,
        SVector<double, 8>{0.49478829029794746, 0.8045632402683945,
                           0.14233875934752005, 0.6248974091625752,
                           0.4100454368750559, 0.36093334233891017,
                           0.7299307580759404, 0.9599831981794166}},
      Dual<double, 8>{
        0.8935518055743592,
        SVector<double, 8>{0.1802316112143557, 0.7998299168331432,
                           0.6885713578218868, 0.16063226225861582,
                           0.8882724638859483, 0.3121292626973291,
                           0.8389228620948505, 0.8982533827333361}}}};
  Dual<Dual<double, 8>, 2> z = a / w + b;
  EXPECT_NEAR(z.value().value(), 11.438286668900178, 1e-14);
  EXPECT_NEAR(z.value().gradient()[0], -5.494684675315883, 1e-14);
  EXPECT_NEAR(z.value().gradient()[1], -4.403716848906783, 1e-14);
  EXPECT_NEAR(z.value().gradient()[2], -5.916876429038721, 1e-14);
  EXPECT_NEAR(z.value().gradient()[3], -9.205580874924776, 1e-14);
  EXPECT_NEAR(z.value().gradient()[4], -6.88946549958806, 1e-14);
  EXPECT_NEAR(z.value().gradient()[5], -0.008506546379252392, 1e-14);
  EXPECT_NEAR(z.value().gradient()[6], -10.024810569806137, 1e-14);
  EXPECT_NEAR(z.value().gradient()[7], -7.818807254621021, 1e-14);
  EXPECT_NEAR(z.gradient()[0].value(), -6.4920996489938965, 1e-14);
  EXPECT_NEAR(z.gradient()[0].gradient()[0], -0.353004214108692, 1e-14);
  EXPECT_NEAR(z.gradient()[0].gradient()[1], -7.546059942645349, 1e-14);
  EXPECT_NEAR(z.gradient()[0].gradient()[2], 6.570646809055814, 1e-14);
  EXPECT_NEAR(z.gradient()[0].gradient()[3], 3.040948459025321, 1e-14);
  EXPECT_NEAR(z.gradient()[0].gradient()[4], 3.301701335355337, 1e-14);
  EXPECT_NEAR(z.gradient()[0].gradient()[5], -6.4119467254828155, 1e-14);
  EXPECT_NEAR(z.gradient()[0].gradient()[6], 2.431800795666499, 1e-14);
  EXPECT_NEAR(z.gradient()[0].gradient()[7], -5.057833482826592, 1e-14);
  EXPECT_NEAR(z.gradient()[1].value(), -15.906268020733835, 1e-14);
  EXPECT_NEAR(z.gradient()[1].gradient()[0], 17.50675476102715, 1e-14);
  EXPECT_NEAR(z.gradient()[1].gradient()[1], 2.3642057402333787, 1e-14);
  EXPECT_NEAR(z.gradient()[1].gradient()[2], 10.049384954558796, 1e-14);
  EXPECT_NEAR(z.gradient()[1].gradient()[3], 31.8458106722888, 1e-14);
  EXPECT_NEAR(z.gradient()[1].gradient()[4], 10.161154827174284, 1e-14);
  EXPECT_NEAR(z.gradient()[1].gradient()[5], -5.524196339292094, 1e-14);
  EXPECT_NEAR(z.gradient()[1].gradient()[6], 22.859958982249076, 1e-14);
  EXPECT_NEAR(z.gradient()[1].gradient()[7], 13.487122714859662, 1e-14);
}
