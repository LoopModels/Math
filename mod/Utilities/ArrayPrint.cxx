#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#ifndef USE_MODULE
#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <type_traits>

#include "Alloc/Mallocator.cxx"
#include "Containers/Flat.cxx"
#include "Math/Rational.cxx"
#else
export module ArrayPrint;

import Allocator;
import Flat;
import Rational;
import STL;
#endif

#ifdef USE_MODULE
export namespace utils {
#else
namespace utils {
#endif
namespace detail {
template <std::integral T> consteval auto maxPow10() -> size_t {
  if constexpr (sizeof(T) == 1) return 3;
  else if constexpr (sizeof(T) == 2) return 5;
  else if constexpr (sizeof(T) == 4) return 10;
  else if constexpr (std::signed_integral<T>) return 19;
  else return 20;
}

template <std::unsigned_integral T> constexpr auto countDigits(T x) {
  if constexpr (!std::same_as<T, bool>) {
    std::array<T, maxPow10<T>() + 1> powers;
    powers[0] = 0;
    powers[1] = 10;
    for (ptrdiff_t i = 2; i < std::ssize(powers); i++)
      powers[i] = powers[i - 1] * 10;
    std::array<T, sizeof(T) * 8 + 1> bits;
    if constexpr (sizeof(T) == 8) {
      bits = {1,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4,
              4,  5,  5,  5,  6,  6,  6,  7,  7,  7,  7,  8,  8,
              8,  9,  9,  9,  10, 10, 10, 10, 11, 11, 11, 12, 12,
              12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16,
              16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 19, 20};
    } else if constexpr (sizeof(T) == 4) {
      bits = {1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4,  5,  5, 5,
              6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10};
    } else if constexpr (sizeof(T) == 2) {
      bits = {1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5};
    } else if constexpr (sizeof(T) == 1) {
      bits = {1, 1, 1, 1, 2, 2, 2, 3, 3};
    }
    T digits;
    if constexpr (std::same_as<T, char>)
      digits =
        bits[8 * sizeof(unsigned char) - std::countl_zero((unsigned char)x)];
    else digits = bits[8 * sizeof(T) - std::countl_zero(x)];
    return std::make_signed_t<T>(digits - (x < powers[digits - 1]));
  } else return 1;
}
template <std::signed_integral T> constexpr auto countDigits(T x) -> T {
  using U = std::make_unsigned_t<T>;
  if (x == std::numeric_limits<T>::min()) return T(sizeof(T) == 8 ? 20 : 11);
  return countDigits<U>(U(std::abs(x))) + T{x < 0};
}

template <typename T> inline auto countDigits(T *x) -> int {
  return countDigits(std::bit_cast<size_t>(x));
}
constexpr auto countDigits(math::Rational x) -> ptrdiff_t {
  ptrdiff_t num = countDigits(x.numerator);
  return (x.denominator == 1) ? num : num + countDigits(x.denominator) + 2;
}
/// \brief Returns the maximum number of digits per column of a matrix.
constexpr auto getMaxDigits(const math::Rational *A, ptrdiff_t M, ptrdiff_t N,
                            ptrdiff_t X) -> containers::Flat<ptrdiff_t> {
  containers::Flat<ptrdiff_t> max_digits(N, 0z);
  // this is slow, because we count the digits of every element
  // we could optimize this by reducing the number of calls to countDigits
  for (ptrdiff_t i = 0; i < M; i++) {
    for (ptrdiff_t j = 0; j < N; j++) {
      ptrdiff_t c = countDigits(A[i * X + j]);
      max_digits[j] = std::max(max_digits[j], c);
    }
  }
  return max_digits;
}

/// Returns the number of digits of the largest number in the matrix.
template <std::integral T>
constexpr auto getMaxDigits(const T *A, ptrdiff_t M, ptrdiff_t N, ptrdiff_t X)
  -> containers::Flat<std::remove_const_t<T>> {
  containers::Flat<std::remove_const_t<T>> max_digits(N, T{});
  // first, we find the digits with the maximum value per column
  for (ptrdiff_t i = 0; i < M; i++) {
    for (ptrdiff_t j = 0; j < N; j++) {
      // negative numbers need one more digit
      // first, we find the maximum value per column,
      // dividing positive numbers by -10
      T Aij = A[i * X + j];
      if constexpr (std::signed_integral<T>)
        max_digits[j] = std::min(max_digits[j], Aij > 0 ? Aij / -10 : Aij);
      else max_digits[j] = std::max(max_digits[j], Aij);
    }
  }
  // then, we count the digits of the maximum value per column
  for (ptrdiff_t j = 0; j < max_digits.size(); j++)
    max_digits[j] = countDigits(max_digits[j]);
  return max_digits;
}

} // namespace detail

template <typename T>
concept Printable = std::same_as<T, double> || requires(std::ostream &os, T x) {
  { os << x } -> std::same_as<std::ostream &>;
  { detail::countDigits(x) } -> std::integral;
};
static_assert(Printable<math::Rational>);

static_assert(Printable<int64_t>);
inline void print_obj(std::ostream &os, Printable auto x) { os << x; };

inline auto printVector(std::ostream &os, auto B, auto E) -> std::ostream & {
  os << "[ ";
  if (B != E) {
    print_obj(os, *B);
    for (; ++B != E;) print_obj(os << ", ", *B);
  }
  os << " ]";
  return os;
}

template <typename T>
inline auto printMatrix(std::ostream &os, const T *A, ptrdiff_t M, ptrdiff_t N,
                        ptrdiff_t X) -> std::ostream & {
  // std::ostream &printMatrix(std::ostream &os, T const &A) {
  if ((!M) || (!N)) return os << "[ ]";
  // first, we determine the number of digits needed per column
  auto max_digits{detail::getMaxDigits(A, M, N, X)};
  using U = decltype(detail::countDigits(std::declval<T>()));
  for (ptrdiff_t i = 0; i < M; i++) {
    if (i) os << "  ";
    else os << "\n[ ";
    for (ptrdiff_t j = 0; j < N; j++) {
      auto Aij = A[i * X + j];
      for (U k = 0; k < U(max_digits[j]) - detail::countDigits(Aij); k++)
        os << " ";
      os << Aij;
      if (j != ptrdiff_t(N) - 1) os << " ";
      else if (i != ptrdiff_t(M) - 1) os << "\n";
    }
  }
  return os << " ]";
}
// We mirror `A` with a matrix of integers indicating sizes, and a vectors of
// chars. We fill the matrix with the number of digits of each element, and
// the vector with the characters of each element. We could use a vector of
// vectors of chars to avoid needing to copy memory on reallocation, but this
// would yield more complicated management. We should also generally be able
// to avoid allocations. We can use a Vector with a lot of initial capacity,
// and then resize based on a conservative estimate of the number of chars per
// elements.
inline auto printMatrix(std::ostream &os, const double *A, ptrdiff_t M,
                        ptrdiff_t N, ptrdiff_t X) -> std::ostream & {
  // std::ostream &printMatrix(std::ostream &os, T const &A) {
  if ((!M) || (!N)) return os << "[ ]";
  // first, we determine the number of digits needed per column
  // we can't have more than 255 digits
  containers::Flat<uint8_t> num_digits{M * N};
  char smem[512];
  char *p0 = smem, *ptr = p0, *p_end = p0 + 512;
  for (ptrdiff_t m = 0; m < M; m++) {
    for (ptrdiff_t n = 0; n < N; n++) {
      auto Aij = A[m * X + n];
      while (true) {
        auto [p, ec] = std::to_chars(ptr, p_end, Aij);
        if (ec == std::errc()) [[likely]] {
          num_digits[m * N + n] = std::distance(ptr, p);
          ptr = p;
          break;
        }
        // we need more space
        ptrdiff_t elem_so_far = m * ptrdiff_t(N) + n;
        ptrdiff_t char_so_far = std::distance(p0, ptr);
        // cld
        ptrdiff_t char_per_elem = (char_so_far + elem_so_far - 1) / elem_so_far;
        ptrdiff_t new_capacity =
          (1 + char_per_elem) * M * N; // +1 for good measure
        char *pnew = alloc::Mallocator<char>{}.allocate(new_capacity);
        std::memcpy(pnew, p0, char_so_far);
        if (smem != p0)
          alloc::Mallocator<char>{}.deallocate(p0, std::distance(p0, p_end));
        p0 = pnew;
        ptr = pnew + char_so_far;
        p_end = pnew + new_capacity;
      }
    }
  }
  containers::Flat<uint8_t> max_digits{num_digits.begin(),
                                       num_digits.begin() + N};
  for (ptrdiff_t m = 0; ++m < M;)
    for (ptrdiff_t n = 0; n < N; n++)
      max_digits[n] = std::max(max_digits[n], num_digits[m * N + n]);
  // we will allocate 512 bytes at a time
  ptr = p0;
  for (ptrdiff_t i = 0; i < M; i++) {
    if (i) os << "  ";
    else os << "\n[ ";
    for (ptrdiff_t j = 0; j < N; j++) {
      ptrdiff_t nD = num_digits[i * N + j];
      for (ptrdiff_t k = 0; k < max_digits[j] - nD; k++) os << " ";
      for (ptrdiff_t n = 0; n < nD; ++n) os << ptr[n];
      if (j != ptrdiff_t(N) - 1) os << " ";
      else if (i != ptrdiff_t(M) - 1) os << "\n";
      ptr += nD;
    }
  }
  if (smem != p0)
    alloc::Mallocator<char>{}.deallocate(p0, std::distance(p0, p_end));
  return os << " ]";
}
} // namespace utils
