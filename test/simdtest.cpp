#include <gtest/gtest.h>
#ifndef USE_MODULE
#include "Math/AxisTypes.cxx"
#include "Math/ElementarySIMD.cxx"
#include "Math/ExpressionTemplates.cxx"
#include "Math/ManagedArray.cxx"
#include <array>
#include <cstddef>
#else
import AxisTypes;
import Elementary;
import ExprTemplates;
import ManagedArray;
import STL;
#endif

TEST(ElementarySIMD, BasicAssertions) {

  math::Vector<double> x{
    std::array{1.564299025169599, 4.328641127555183, -10.43843599926044,
               -1.650625233314754, -0.5851694806951444, 0.07422197149516746,
               -5.231238164802142, 7.298495240920298, 6.983762398719033,
               4.148462859390057, -1.5877584351154996, 0.21815818734467177,
               6.717052006977858, 1.1366111295246064, -3.2809440656442757}},
    y{std::array{4.779323575889951, 75.8411580559438, 2.9284974204673125e-5,
                 0.19192987014972496, 0.5570114511240655, 1.077045852442665,
                 0.005346900857986191, 1478.074107909007, 1078.9702561142797,
                 63.336568222798185, 0.20438323693119723, 1.2437838029152637,
                 826.3777611913293, 3.116190086521906, 0.037592750025674}},
    z{math::length(15)};

  z << math::elementwise(x.view(), [](auto a) { return math::exp(a); });
  for (ptrdiff_t i = 0; i < 15; ++i) EXPECT_DOUBLE_EQ(y[i], z[i]);
}
