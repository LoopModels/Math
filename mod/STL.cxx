module;

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <print>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>

#ifdef MATHTESTSTLDEFINITIONS
#include <random>
#endif

export module STL;

export using std::int16_t;
export using std::int32_t;
export using std::int64_t;
export using std::int8_t;
export using std::ptrdiff_t;
export using std::size_t;
export using std::uint16_t;
export using std::uint32_t;
export using std::uint64_t;
export using std::uint8_t;
export using std::uint_fast16_t;
export using i64 = signed long long int;
export using u64 = unsigned long long int;

namespace std {

export using ::std::abort;
export using ::std::abs;
export using ::std::align;
export using ::std::align_val_t;
export using ::std::allocator;
export using ::std::allocator_traits;
export using ::std::all_of;
export using ::std::array;
export using ::std::assignable_from;
export using ::std::assume_aligned;
export using ::std::begin;
export using ::std::bidirectional_iterator;
export using ::std::bit_and;
export using ::std::bit_cast;
export using ::std::bit_ceil;
export using ::std::bit_not;
export using ::std::bit_or;
export using ::std::bit_xor;
export using ::std::ceil;
export using ::std::clamp;
export using ::std::common_type;
export using ::std::common_type_t;
export using ::std::common_with;
export using ::std::conditional_t;
export using ::std::construct_at;
export using ::std::constructible_from;
export using ::std::convertible_to;
export using ::std::copy;
export using ::std::copyable;
export using ::std::copy_backward;
export using ::std::copy_constructible;
export using ::std::copy_n;
export using ::std::count_if;
export using ::std::countl_one;
export using ::std::countl_zero;
export using ::std::countr_one;
export using ::std::countr_zero;
export using ::std::cout;
export using ::std::declval;
export using ::std::destroy_at;
export using ::std::destroy_n;
export using ::std::distance;
export using ::std::divides;
export using ::std::end;
export using ::std::equal_to;
export using ::std::errc;
export using ::std::exchange;
export using ::std::false_type;
export using ::std::fill;
export using ::std::fill_n;
export using ::std::find_if;
export using ::std::floating_point;
export using ::std::floor;
export using ::std::fma;
export using ::std::forward;
export using ::std::forward_iterator;
export using ::std::get;
export using ::std::greater;
export using ::std::greater_equal;
export using ::std::has_single_bit;
export using ::std::index_sequence;
export using ::std::indirectly_readable;
export using ::std::initializer_list;
export using ::std::input_iterator;
export using ::std::input_or_output_iterator;
export using ::std::integral;
export using ::std::integral_constant;
export using ::std::invocable;
export using ::std::is_assignable_v;
export using ::std::is_constructible_v;
export using ::std::is_const_v;
export using ::std::is_convertible_v;
export using ::std::is_copy_assignable_v;
export using ::std::is_empty_v;
export using ::std::is_integral_v;
export using ::std::is_invocable_v;
export using ::std::is_same_v;
export using ::std::is_signed_v;
export using ::std::is_trivially_copyable_v;
export using ::std::is_trivially_copy_constructible_v;
export using ::std::is_trivially_default_constructible_v;
export using ::std::is_trivially_destructible_v;
export using ::std::is_trivially_move_assignable_v;
export using ::std::is_trivially_move_constructible_v;
export using ::std::iter_rvalue_reference_t;
export using ::std::launder;
export using ::std::less;
export using ::std::less_equal;
export using ::std::log;
export using ::std::log1p;
export using ::std::log2;
export using ::std::logical_not;
export using ::std::make_index_sequence;
export using ::std::make_pair;
export using ::std::make_signed_t;
export using ::std::make_unsigned_t;
export using ::std::max;
export using ::std::max_align_t;
export using ::std::memcpy;
export using ::std::memset;
export using ::std::min;
export using ::std::minus;
export using ::std::modulus;
export using ::std::movable;
export using ::std::move;
export using ::std::move_constructible;
export using ::std::multiplies;
export using ::std::negate;
export using ::std::not_equal_to;
export using ::std::nullopt_t;
export using ::std::numeric_limits;
export using ::std::optional;
export using ::std::ostream;
export using ::std::output_iterator;
export using ::std::partial_ordering;
export using ::std::plus;
export using ::std::popcount;
export using ::std::print;
export using ::std::println;
export using ::std::random_access_iterator;
export using ::std::remove_const_t;
export using ::std::remove_cvref_t;
export using ::std::remove_reference_t;
export using ::std::reverse_iterator;
export using ::std::round;
export using ::std::same_as;
export using ::std::signbit;
export using ::std::signed_integral;
export using ::std::size;
export using ::std::ssize;
export using ::std::strong_ordering;
export using ::std::swap;
export using ::std::to_chars;
export using ::std::totally_ordered;
export using ::std::true_type;
export using ::std::trunc;
export using ::std::tuple_element;
export using ::std::tuple_element_t;
export using ::std::tuple_size;
export using ::std::tuple_size_v;
export using ::std::type_identity;
export using ::std::type_identity_t;
export using ::std::uninitialized_copy;
export using ::std::uninitialized_copy_n;
export using ::std::uninitialized_default_construct;
export using ::std::uninitialized_default_construct_n;
export using ::std::uninitialized_fill_n;
export using ::std::uninitialized_move_n;
export using ::std::unique_ptr;
export using ::std::unsigned_integral;
export using ::std::weakly_incrementable;

#ifdef MATHTESTSTLDEFINITIONS
export using ::std::exp2;
export using ::std::exponential_distribution;
export using ::std::gcd;
export using ::std::mt19937;
export using ::std::random_device;
export using ::std::uniform_real_distribution;
export { namespace views = ::std::ranges::views; }
#endif

namespace ranges {
export using ::std::ranges::all_of;
export using ::std::ranges::any_of;
export using ::std::ranges::enable_borrowed_range;
export using ::std::ranges::forward_range;
export using ::std::ranges::range;
export using ::std::ranges::swap;
export using ::std::ranges::view;
export using ::std::ranges::view_interface;
#ifdef MATHTESTSTLDEFINITIONS
export using ::std::ranges::filter_view;
export using ::std::ranges::owning_view;
export using ::std::ranges::reverse_view;
export using ::std::ranges::transform;
namespace views {
export using ::std::ranges::views::filter;
export using ::std::ranges::views::reverse;
export using ::std::ranges::views::zip;
} // namespace views
#endif
} // namespace ranges

} // namespace std
