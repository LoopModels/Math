#ifdef USE_MODULE
module;
#else
#pragma once
#endif
#ifndef USE_MODULE
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "Bit/Float.cxx"
#include "SIMD/SIMD.cxx"
#else
export module Elementary;

import BitHack;
import SIMD;
import STL;
#endif

// NOLINTNEXTLINE(modernize-avoid-c-arrays)
constexpr double J_TABLE[256] = {1.0,
                                 1.0027112750502025,
                                 1.0054299011128027,
                                 1.0081558981184175,
                                 1.0108892860517005,
                                 1.0136300849514894,
                                 1.016378314910953,
                                 1.019133996077738,
                                 1.0218971486541166,
                                 1.0246677928971357,
                                 1.0274459491187637,
                                 1.030231637686041,
                                 1.0330248790212284,
                                 1.0358256936019572,
                                 1.0386341019613787,
                                 1.041450124688316,
                                 1.0442737824274138,
                                 1.0471050958792898,
                                 1.0499440858006872,
                                 1.0527907730046264,
                                 1.0556451783605572,
                                 1.0585073227945128,
                                 1.061377227289262,
                                 1.0642549128844645,
                                 1.0671404006768237,
                                 1.0700337118202419,
                                 1.0729348675259756,
                                 1.075843889062791,
                                 1.0787607977571199,
                                 1.0816856149932152,
                                 1.0846183622133092,
                                 1.0875590609177697,
                                 1.0905077326652577,
                                 1.0934643990728858,
                                 1.0964290818163769,
                                 1.099401802630222,
                                 1.102382583307841,
                                 1.1053714457017412,
                                 1.1083684117236787,
                                 1.1113735033448175,
                                 1.1143867425958924,
                                 1.1174081515673693,
                                 1.1204377524096067,
                                 1.12347556733302,
                                 1.1265216186082418,
                                 1.129575928566288,
                                 1.1326385195987192,
                                 1.1357094141578055,
                                 1.1387886347566916,
                                 1.1418762039695616,
                                 1.1449721444318042,
                                 1.148076478840179,
                                 1.1511892299529827,
                                 1.154310420590216,
                                 1.1574400736337511,
                                 1.1605782120274988,
                                 1.1637248587775775,
                                 1.1668800369524817,
                                 1.1700437696832502,
                                 1.1732160801636373,
                                 1.1763969916502812,
                                 1.1795865274628758,
                                 1.182784710984341,
                                 1.1859915656609938,
                                 1.189207115002721,
                                 1.1924313825831512,
                                 1.1956643920398273,
                                 1.1989061670743806,
                                 1.202156731452703,
                                 1.2054161090051239,
                                 1.2086843236265816,
                                 1.2119613992768012,
                                 1.215247359980469,
                                 1.2185422298274085,
                                 1.2218460329727576,
                                 1.2251587936371455,
                                 1.22848053610687,
                                 1.2318112847340759,
                                 1.2351510639369334,
                                 1.2384998981998165,
                                 1.241857812073484,
                                 1.245224830175258,
                                 1.2486009771892048,
                                 1.2519862778663162,
                                 1.255380757024691,
                                 1.2587844395497165,
                                 1.2621973503942507,
                                 1.2656195145788063,
                                 1.2690509571917332,
                                 1.2724917033894028,
                                 1.275941778396392,
                                 1.2794012075056693,
                                 1.2828700160787783,
                                 1.2863482295460256,
                                 1.2898358734066657,
                                 1.2933329732290895,
                                 1.2968395546510096,
                                 1.3003556433796506,
                                 1.3038812651919358,
                                 1.3074164459346773,
                                 1.3109612115247644,
                                 1.3145155879493546,
                                 1.318079601266064,
                                 1.3216532776031575,
                                 1.3252366431597413,
                                 1.3288297242059544,
                                 1.3324325470831615,
                                 1.3360451382041458,
                                 1.339667524053303,
                                 1.3432997311868353,
                                 1.3469417862329458,
                                 1.3505937158920345,
                                 1.3542555469368927,
                                 1.3579273062129011,
                                 1.3616090206382248,
                                 1.365300717204012,
                                 1.3690024229745905,
                                 1.3727141650876684,
                                 1.3764359707545302,
                                 1.380167867260238,
                                 1.383909881963832,
                                 1.387662042298529,
                                 1.3914243757719262,
                                 1.3951969099662003,
                                 1.3989796725383112,
                                 1.4027726912202048,
                                 1.4065759938190154,
                                 1.4103896082172707,
                                 1.4142135623730951,
                                 1.4180478843204152,
                                 1.4218926021691656,
                                 1.4257477441054942,
                                 1.42961333839197,
                                 1.433489413367789,
                                 1.4373759974489824,
                                 1.4412731191286257,
                                 1.4451808069770467,
                                 1.449099089642035,
                                 1.4530279958490526,
                                 1.4569675544014438,
                                 1.460917794180647,
                                 1.4648787441464057,
                                 1.4688504333369818,
                                 1.4728328908693675,
                                 1.4768261459394993,
                                 1.4808302278224719,
                                 1.4848451658727524,
                                 1.488870989524397,
                                 1.4929077282912648,
                                 1.4969554117672355,
                                 1.5010140696264256,
                                 1.5050837316234065,
                                 1.5091644275934228,
                                 1.5132561874526098,
                                 1.5173590411982147,
                                 1.5214730189088146,
                                 1.5255981507445384,
                                 1.529734466947287,
                                 1.533881997840956,
                                 1.5380407738316568,
                                 1.5422108254079407,
                                 1.5463921831410214,
                                 1.550584877685,
                                 1.5547889397770887,
                                 1.559004400237837,
                                 1.5632312899713576,
                                 1.567469639965553,
                                 1.5717194812923414,
                                 1.5759808451078865,
                                 1.5802537626528246,
                                 1.5845382652524937,
                                 1.588834384317164,
                                 1.593142151342267,
                                 1.597461597908627,
                                 1.6017927556826934,
                                 1.606135656416771,
                                 1.6104903319492543,
                                 1.6148568142048607,
                                 1.6192351351948637,
                                 1.6236253270173289,
                                 1.6280274218573478,
                                 1.632441451987275,
                                 1.6368674497669644,
                                 1.6413054476440063,
                                 1.645755478153965,
                                 1.6502175739206177,
                                 1.6546917676561943,
                                 1.6591780921616162,
                                 1.6636765803267364,
                                 1.6681872651305825,
                                 1.6727101796415966,
                                 1.6772453570178785,
                                 1.681792830507429,
                                 1.6863526334483934,
                                 1.6909247992693053,
                                 1.6955093614893326,
                                 1.7001063537185235,
                                 1.7047158096580513,
                                 1.709337763100463,
                                 1.713972247929926,
                                 1.718619298122478,
                                 1.723278947746274,
                                 1.7279512309618377,
                                 1.732636182022311,
                                 1.7373338352737062,
                                 1.7420442251551564,
                                 1.746767386199169,
                                 1.7515033530318782,
                                 1.7562521603732995,
                                 1.761013843037584,
                                 1.7657884359332727,
                                 1.7705759740635547,
                                 1.7753764925265212,
                                 1.7801900265154245,
                                 1.785016611318935,
                                 1.789856282321401,
                                 1.7947090750031072,
                                 1.7995750249405351,
                                 1.804454167806624,
                                 1.809346539371032,
                                 1.8142521755003989,
                                 1.8191711121586085,
                                 1.8241033854070534,
                                 1.8290490314048973,
                                 1.8340080864093424,
                                 1.8389805867758937,
                                 1.843966568958626,
                                 1.8489660695104508,
                                 1.8539791250833855,
                                 1.8590057724288205,
                                 1.864046048397789,
                                 1.8690999899412386,
                                 1.8741676341103,
                                 1.8792490180565602,
                                 1.8843441790323345,
                                 1.8894531543909392,
                                 1.8945759815869656,
                                 1.8997126981765553,
                                 1.9048633418176741,
                                 1.9100279502703899,
                                 1.9152065613971474,
                                 1.9203992131630474,
                                 1.925605943636125,
                                 1.930826790987627,
                                 1.9360617934922943,
                                 1.9413109895286405,
                                 1.9465744175792332,
                                 1.9518521162309783,
                                 1.9571441241754002,
                                 1.9624504802089273,
                                 1.9677712232331759,
                                 1.9731063922552343,
                                 1.978456026387951,
                                 1.9838201648502194,
                                 1.9891988469672663,
                                 1.9945921121709402};

constexpr auto max_exp(double, std::integral_constant<int, 3>) -> double {
  return 709.7827128933841;
}
constexpr auto max_exp(float, std::integral_constant<int, 3>) -> float {
  return 88.72284F;
}
constexpr auto max_exp(double, std::integral_constant<int, 2>) -> double {
  return 1024.0;
}
constexpr auto max_exp(float, std::integral_constant<int, 2>) -> float {
  return 128.0F;
}
constexpr auto max_exp(double, std::integral_constant<int, 10>) -> double {
  return 308.25471555991675;
}
constexpr auto max_exp(float, std::integral_constant<int, 10>) -> float {
  return 38.53184F;
}

constexpr auto subnormal_exp(double, std::integral_constant<int, 3>) -> double {
  return -708.3964185322641;
}
constexpr auto subnormal_exp(float, std::integral_constant<int, 3>) -> float {
  return -87.33655F;
}
constexpr auto subnormal_exp(double, std::integral_constant<int, 2>) -> double {
  return -1022.0;
}
constexpr auto subnormal_exp(float, std::integral_constant<int, 2>) -> float {
  return -126.00001F;
}
constexpr auto subnormal_exp(double,
                             std::integral_constant<int, 10>) -> double {
  return -307.6526555685887;
}
constexpr auto subnormal_exp(float, std::integral_constant<int, 10>) -> float {
  return -37.92978F;
}

constexpr auto LogBo256INV(std::integral_constant<int, 3>) -> double {
  return 369.329930467;
}
constexpr auto LogBo256INV(std::integral_constant<int, 2>) -> double {
  return 256;
}
constexpr auto LogBo256INV(std::integral_constant<int, 10>) -> double {
  return 850.4135922911647;
}

constexpr auto LogBo256U(std::integral_constant<int, 2>) -> double {
  return -0.00390625;
}
constexpr auto LogBo256U(std::integral_constant<int, 3>) -> double {
  return -0.002707606173999011;
}
constexpr auto LogBo256U(std::integral_constant<int, 10>) -> double {
  return -0.0011758984204561784;
}
constexpr auto LogBo256L(std::integral_constant<int, 2>) -> double {
  return 0.0;
}
constexpr auto LogBo256L(std::integral_constant<int, 3>) -> double {
  return -6.327543041662719e-14;
}
constexpr auto LogBo256L(std::integral_constant<int, 10>) -> double {
  return -1.0624811566412999e-13;
}

template <std::floating_point T>
static inline constexpr double magic_round_const = 6.755399441055744e15;
template <> inline constexpr float magic_round_const<float> = 1.048576e7F;

constexpr auto trunclo(double x) -> double {
  return std::bit_cast<double>(std::bit_cast<uint64_t>(x) & 0xfffffffff8000000);
}

constexpr auto fmadd(double x, double y, double z) -> double {
#if __cpp_lib_constexpr_cmath < 202202L // no cmath constexpr support
  if consteval { // TODO drop when c++23 constexpr fma support is available
#if defined(__linux__) && defined(__x86_64__)
    __float128 a = x, b = y, c = z;
    return double(a * b + c);
#else
    // This is not a perfect implementation!!!
    double hx = trunclo(x), hy = trunclo(y), lx = x - hx, ly = y - hy;
    double hxy = x * y;
    double lxy = (((hx * hy - hxy) + lx * hy + hx * ly) + lx * ly);
    double s = hxy + z;
    return s + (((hxy - s) + z) + lxy);
#endif
  } else {
#endif // end no cmath constexpr support
    return std::fma(x, y, z);
#if __cpp_lib_constexpr_cmath < 202202L // close the `if consteval` block
  }
#endif
}

[[gnu::always_inline]] constexpr auto
expm1b_kernel(std::integral_constant<int, 2>, double x) -> double {
  return x * fmadd(fmadd(fmadd(0.009618130135925114, x, 0.055504115022757844),
                         x, 0.2402265069590989),
                   x, 0.6931471805599393);
}
[[gnu::always_inline]] constexpr auto
expm1b_kernel(std::integral_constant<int, 3>, double x) -> double {
  return x * fmadd(fmadd(fmadd(0.04166666762124105, x, 0.1666666704849642), x,
                         0.49999999999999983),
                   x, 0.9999999999999998);
}
[[gnu::always_inline]] constexpr auto
expm1b_kernel(std::integral_constant<int, 10>, double x) -> double {
  return x * fmadd(fmadd(fmadd(fmadd(0.5393833837413015, x, 1.1712561359457612),
                               x, 2.0346785922926713),
                         x, 2.6509490552382577),
                   x, 2.302585092994046);
}

template <int B> constexpr auto exp_impl(double x) -> double {
  constexpr std::integral_constant<int, B> base{};
  // #if __FAST_MATH__
  if (x >= max_exp(x, base)) return std::numeric_limits<double>::max();
  // #else
  // if (x >= max_exp(x, base)) return std::numeric_limits<double>::infinity();
  // #endif
  if (x <= subnormal_exp(x, base)) return 0.0;
  double float_n = fmadd(x, LogBo256INV(base), magic_round_const<double>);
  auto N = std::bit_cast<uint64_t>(float_n);
  float_n -= magic_round_const<double>;
  double r = fmadd(float_n, LogBo256U(base), x);
  r = fmadd(float_n, LogBo256L(base), r);
  double jU = J_TABLE[N & 255];
  double small = fmadd(jU, expm1b_kernel(base, r), jU);
  auto twopk = int64_t(N >> 8) << 52;
  return std::bit_cast<double>(twopk + std::bit_cast<int64_t>(small));
}

template <ptrdiff_t W>
[[gnu::always_inline]] constexpr auto
expm1b_kernel(std::integral_constant<int, 2>,
              simd::Vec<W, double> x) -> simd::Vec<W, double> {
  return x * (((0.009618130135925114 * x + 0.055504115022757844) * x +
               0.2402265069590989) *
              x * 0.6931471805599393);
}
template <ptrdiff_t W>
[[gnu::always_inline]] constexpr auto
expm1b_kernel(std::integral_constant<int, 3>,
              simd::Vec<W, double> x) -> simd::Vec<W, double> {
  return x * (((0.04166666762124105 * x + 0.1666666704849642) * x +
               0.49999999999999983) *
                x +
              0.9999999999999998);
}
template <ptrdiff_t W>
[[gnu::always_inline]] constexpr auto
expm1b_kernel(std::integral_constant<int, 10>,
              simd::Vec<W, double> x) -> simd::Vec<W, double> {
  return x * ((((0.5393833837413015 * x + 1.1712561359457612) * x +
                2.0346785922926713) *
                 x +
               2.6509490552382577) *
                x +
              2.302585092994046);
}
template <int B, ptrdiff_t W>
constexpr auto exp_impl_core(simd::Vec<W, double> x) -> simd::Vec<W, double> {
  constexpr std::integral_constant<int, B> base{};
  // #if __FAST_MATH__
  using V = simd::Vec<W, double>;
  auto maxmask = simd::cmp::ge<W, double>(
    x, simd::vbroadcast<W, double>(max_exp(0.0, base)));
  V maxval = simd::vbroadcast<W, double>(std::numeric_limits<double>::max());
  auto minmask = simd::cmp::le<W, double>(
    x, simd::vbroadcast<W, double>(subnormal_exp(0.0, base)));
  V alt = simd::select<double>(maxmask, maxval, V{});
  auto altmask = maxmask | minmask;
  // #else
  // if (x >= max_exp(x, base)) return std::numeric_limits<double>::infinity();
  // #endif
  V float_n = x * simd::vbroadcast<W, double>(LogBo256INV(base)) +
              simd::vbroadcast<W, double>(magic_round_const<double>);
  auto N = std::bit_cast<simd::Vec<W, int64_t>>(float_n);
  float_n -= simd::vbroadcast<W, double>(magic_round_const<double>);

  V r = float_n * simd::vbroadcast<W, double>(LogBo256U(base)) + x;
  r = float_n * simd::vbroadcast<W, double>(LogBo256L(base)) + r;
  V jU = simd::gather(J_TABLE, simd::mask::None<W>{},
                      N & simd::vbroadcast<W, int64_t>(255));
  V small = jU * expm1b_kernel<W>(base, r) + jU;
  simd::Vec<W, int64_t> twopk = std::bit_cast<simd::Vec<W, int64_t>>(
    (std::bit_cast<simd::Vec<W, int64_t>>(N) >> 8) << 52);
  V z = std::bit_cast<V>(twopk + std::bit_cast<simd::Vec<W, int64_t>>(small));
  return simd::select<double>(altmask, alt, z);
}
template <int B, ptrdiff_t R, ptrdiff_t U, ptrdiff_t W>
constexpr auto
exp_impl(simd::Unroll<R, U, W, double> x) -> simd::Unroll<R, U, W, double> {
  if constexpr (R * U == 1) {
    return {exp_impl_core<B, W>(x.vec_)};
  } else {
    simd::Unroll<R, U, W, double> ret;
    for (ptrdiff_t i = 0; i < R * U; ++i)
      ret.data_[i] = exp_impl_core<B, W>(x.data_[i]);
    return ret;
  }
}

#ifdef USE_MODULE
export namespace math {
#else
namespace math {
#endif
constexpr auto exp(double x) -> double { return exp_impl<3>(x); }
constexpr auto exp2(double x) -> double { return exp_impl<2>(x); }
constexpr auto exp10(double x) -> double { return exp_impl<10>(x); }

constexpr auto exp2(int64_t x) -> double {
  // if (x > 1023) return std::numeric_limits<double>::infinity();
  if (x > 1023) return std::numeric_limits<double>::max();
  if (x <= -1023) return std::bit_cast<double>(uint64_t(1) << ((x + 1074)));
  return std::bit_cast<double>((x + 1023) << 52);
}
constexpr auto exp2(unsigned x) -> double {
  // if (x > 1023) return std::numeric_limits<double>::infinity();
  if (x > 1023) return std::numeric_limits<double>::max();
  return bit::exp2unchecked(x);
}
constexpr auto log(double x) -> double { return std::log(x); }
constexpr auto log2(double x) -> double { return std::log2(x); }
constexpr auto log1p(double x) -> double { return std::log1p(x); }
constexpr auto sigmoid(double x) -> double { return 1.0 / (1.0 + exp(-x)); }
constexpr auto softplus(double x) -> double { return log1p(exp(x)); }
constexpr auto logit(double x) -> double { return log(x / (1.0 - x)); }

template <int l = 8> constexpr auto smax(auto x, auto y) {
  auto d = x > y ? (y - x) : (x - y);
  auto o = x > y ? decltype(d)(x) : decltype(d)(y);
  return o + (softplus(l * d) / l);
}

template <int l = 8> constexpr auto smin(auto x, auto y) {
  return smax<-l>(x, y);
}
template <ptrdiff_t W>
constexpr auto exp(simd::Vec<W, double> x) -> simd::Vec<W, double> {
  return exp_impl_core<3, W>(x);
}
template <ptrdiff_t W>
constexpr auto exp2(simd::Vec<W, double> x) -> simd::Vec<W, double> {
  return exp_impl_core<2, W>(x);
}
template <ptrdiff_t W>
constexpr auto exp10(simd::Vec<W, double> x) -> simd::Vec<W, double> {
  return exp_impl_core<10, W>(x);
}
template <ptrdiff_t R, ptrdiff_t U, ptrdiff_t W>
constexpr auto
exp(simd::Unroll<R, U, W, double> x) -> simd::Unroll<R, U, W, double> {
  return exp_impl<3>(x);
}
template <ptrdiff_t R, ptrdiff_t U, ptrdiff_t W>
constexpr auto
exp2(simd::Unroll<R, U, W, double> x) -> simd::Unroll<R, U, W, double> {
  return exp_impl<2>(x);
}
template <ptrdiff_t R, ptrdiff_t U, ptrdiff_t W>
constexpr auto
exp10(simd::Unroll<R, U, W, double> x) -> simd::Unroll<R, U, W, double> {
  return exp_impl<10>(x);
}

template <ptrdiff_t W>
constexpr auto sigmoid(simd::Vec<W, double> x) -> simd::Vec<W, double> {
  return 1.0 / (1.0 + exp<W>(-x));
}
} // namespace math
