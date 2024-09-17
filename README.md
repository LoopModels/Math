# Math

Supported compilers: Clang 17, GCC 13.2.
This project uses C++23, and thus requires the latest versions of the compilers.
Once Clang 18 and GCC 14 are released, we will update to those as the minimum supported compilers (for deducing this, and perhaps `std::print`).

This is the math library used by LoopModels.
It primarily serves to present:
1. a convenient API for array math.
2. a focus on integer operations.
3. a rational simplex solver for linear programs, focused on use cases from Polyhedral analysis.

It uses expression templates for fusing operations, and has slicing via `_` for an entire slice, or `_(i,j)` to select the close-open interval from `i` to `j`. `end` and `last` are keywords for the end of an axis, and the last element. E.g., `A[i,last]` would select the element from the `i`th row and last column, while `A[i,_(j,end)]` would select the `i`th row, from `j` to the end. `A[i,_(j,last)]` would drop the last element, and be equivalent to `A[i,_(j,end-1)]` (which is also of course legal and works as intended). Slicing creates views.

Copying, e.g. from an expression template to an array, is done via `destination << source`. 
`operator=` is reserved for copying the actual objects, rather than referenced memory, as this seems to be the behavior C++ views.
Thus if `A` and `B` are views, `A = B` will make `A` the same view as `B`, while `A << B` will copy memory from `B` to `A`.

Vectors are interpreted as row vectors by default. `v.t()` or `transpose(v)` may be used to transpose.
`A[_,i]` is a column-vector, `A[i,_]` a row-vector.

This repository was created using the [ModernCppStarter](https://github.com/TheLartians/ModernCppStarter).
