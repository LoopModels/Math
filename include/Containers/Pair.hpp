#pragma once
#ifndef Pair_hpp_INCLUDED
#define Pair_hpp_INCLUDED

template <class F, class S> struct Pair {
  [[no_unique_address]] F first;
  [[no_unique_address]] S second;
};

#endif // Pair_hpp_INCLUDED
