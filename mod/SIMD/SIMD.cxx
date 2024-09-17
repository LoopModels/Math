#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifndef USE_MODULE
#include "SIMD/Indexing.cxx"
#include "SIMD/Intrin.cxx"
#include "SIMD/Masks.cxx"
#include "SIMD/Unroll.cxx"
#include "SIMD/UnrollIndex.cxx"
#include "SIMD/Vec.cxx"
#else
export module SIMD;

export import :Index;
export import :Intrin;
export import :Mask;
export import :Unroll;
export import :UnrollIndex;
export import :Vec;
#endif