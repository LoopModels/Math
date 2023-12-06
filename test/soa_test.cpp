#include <gtest/gtest.h>
#include "Math/SOA.hpp"
#include "Containers/Tuple.hpp"
// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(SOATest, BasicAssertions) {
  poly::containers::Tuple x{3,2.1,5.0f};
  // poly::math::ManagedSOA soa{std::type_identity<decltype(x)>{}, 5};
  poly::math::ManagedSOA soa(std::type_identity<decltype(x)>{}, ptrdiff_t(5));
}
