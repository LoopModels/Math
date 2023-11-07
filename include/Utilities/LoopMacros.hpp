#pragma once
#ifndef LoopMacros_hpp_INCLUDED
#define LoopMacros_hpp_INCLUDED

#if !defined(__clang__) && defined(__GNUC__)
#define POLYMATHVECTORIZE _Pragma("GCC ivdep")
#define POLYMATHIVDEP _Pragma("GCC ivdep")
#define POLYMATHNOUNROLL _Pragma("GCC unroll 0")
#define POLYMATHNOVECTORIZE _Pragma("GCC novector")
// #define POLYMATHVECTORIZE _Pragma("GCC unroll 2") _Pragma("GCC ivdep")
#elif defined (__clang__)
#define POLYMATHVECTORIZE _Pragma("omp simd")\
 _Pragma("clang loop vectorize(enable) interleave_count(2) vectorize_predicate(enable)")
#define POLYMATHIVDEP
#define POLYMATHNOUNROLL _Pragma("nounroll")
#define POLYMATHNOVECTORIZE _Pragma("clang loop vectorize(disable)")
#else
#define POLYMATHVECTORIZE
#define POLYMATHIVDEP
#define POLYMATHNOUNROLL
#define POLYMATHNOVECTORIZE
#endif

#endif // LoopMacros_hpp_INCLUDED
