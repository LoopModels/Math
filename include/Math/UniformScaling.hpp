#pragma once
#include "Math/Matrix.hpp"
namespace poly::math {

template <class T> struct UniformScaling {
  using value_type = T;
  T value;
  constexpr UniformScaling(T x) : value(x) {}
  constexpr auto operator()(Row r, Col c) const -> T {
    return r == c ? value : T{};
  }
  static constexpr auto numRow() -> Row { return 0; }
  static constexpr auto numCol() -> Col { return 0; }
  static constexpr auto size() -> CartesianIndex<Row, Col> { return {0, 0}; }
  static constexpr auto dim() -> DenseDims { return {0, 0}; }
  [[nodiscard]] constexpr auto view() const -> auto { return *this; };
  template <class U> constexpr auto operator*(const U &x) const {
    if constexpr (std::is_same_v<std::remove_cvref_t<T>, std::true_type>)
      return UniformScaling<U>{x};
    else return UniformScaling<U>{value * x};
  }
  constexpr auto isEqual(const AbstractMatrix auto &A) const -> bool {
    auto R = ptrdiff_t(A.numRow());
    if (R != A.numCol()) return false;
    for (ptrdiff_t r = 0; r < R; ++r)
      for (ptrdiff_t c = 0; c < R; ++c)
        if (A(r, c) != ((r == c) * value)) return false;
    return true;
  }
  constexpr auto operator==(const AbstractMatrix auto &A) const -> bool {
    return isEqual(A);
  }
  friend inline auto operator<<(std::ostream &os, UniformScaling S)
    -> std::ostream & {
    return os << "UniformScaling(" << S.value << ")";
  }
};
template <class T>
constexpr auto operator==(const AbstractMatrix auto &A,
                          const UniformScaling<T> &B) -> bool {
  return B.isEqual(A);
}
template <class T, class U>
constexpr auto operator*(const U &x, UniformScaling<T> d) {
  if constexpr (std::is_same_v<std::remove_cvref_t<T>, std::true_type>)
    return UniformScaling<U>{x};
  else return UniformScaling<U>{d.value * x};
}

static constexpr inline UniformScaling<std::true_type> I{
  std::true_type{}}; // identity

template <class T> UniformScaling(T) -> UniformScaling<T>;
static_assert(AbstractMatrix<UniformScaling<int64_t>>);

} // namespace poly::math
