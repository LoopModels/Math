#pragma once

#if __has_cpp_attribute(gsl::Owner)
#define MATH_GSL_OWNER [[gsl::Owner]]
#else
#define MATH_GSL_OWNER
#endif
#if __has_cpp_attribute(gsl::Pointer)
#define MATH_GSL_POINTER [[gsl::Pointer]]
#else
#define MATH_GSL_POINTER
#endif
