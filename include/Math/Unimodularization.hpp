#pragma once
#include "Math/Array.hpp"
#include "Math/NormalForm.hpp"
#include <cstdint>

namespace poly::math {
// if `A` can be unimodularized, returns the inverse of the unimodularized `A`
[[nodiscard]] inline auto unimodularize(IntMatrix<> A)
  -> std::optional<SquareMatrix<int64_t>> {
  std::optional<std::pair<IntMatrix<>, SquareMatrix<int64_t>>> ohnf =
    NormalForm::hermite(std::move(A));
  if (!ohnf.has_value()) return {};
  auto &[H, U] = *ohnf;
  for (ptrdiff_t m = 0; m < H.numCol(); ++m)
    if (H[m, m] != 1) return {};
  return std::move(U);
}
} // namespace poly::math
