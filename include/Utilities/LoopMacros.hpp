#pragma once
#ifndef LoopMacros_hpp_INCLUDED
#define LoopMacros_hpp_INCLUDED

#if !defined(__clang__) && defined(__GNUC__)
#define POLYMATHVECTORIZE _Pragma("GCC ivdep")
#define POLYMATHIVDEP _Pragma("GCC ivdep")
#define POLYMATHNOUNROLL _Pragma("GCC unroll 0")
#if __GNUC__ >= 14
#define POLYMATHNOVECTORIZE _Pragma("GCC novector")
#else
#define POLYMATHNOVECTORIZE
#endif
#define POLYMATHFAST
// #define POLYMATHVECTORIZE _Pragma("GCC unroll 2") _Pragma("GCC ivdep")
#elif defined (__clang__)
#define POLYMATHVECTORIZE _Pragma("omp simd")\
 _Pragma("clang loop vectorize(enable) interleave_count(2) vectorize_predicate(enable)")
#define POLYMATHIVDEP _Pragma("clang loop vectorize(disable)")
#define POLYMATHNOUNROLL _Pragma("nounroll")
#define POLYMATHNOVECTORIZE _Pragma("clang loop vectorize(disable)")
#define POLYMATHFAST _Pragma("clang fp reassociate(on) contract(fast)")
#else
#define POLYMATHVECTORIZE
#define POLYMATHIVDEP
#define POLYMATHNOUNROLL
#define POLYMATHNOVECTORIZE
#define POLYMATHFAST
#endif

#endif // LoopMacros_hpp_INCLUDED
