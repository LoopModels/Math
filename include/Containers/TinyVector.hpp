#pragma once

#include "Containers/Storage.hpp"
#include "Utilities/Invariant.hpp"
#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <limits>

namespace poly::containers {
using utils::invariant;

template <class T, size_t N, typename L = ptrdiff_t> class TinyVector {
  static_assert(N > 0);
  static_assert(std::numeric_limits<L>::max() >= N);
  Storage<T, N> data;
  L len{};

public:
  constexpr TinyVector() = default;
  constexpr TinyVector(const std::initializer_list<T> &list) {
    invariant(list.size() <= N);
    len = L(list.size());
    std::copy(list.begin(), list.end(), data.data());
  }
  constexpr TinyVector(T t) : len{1} { data.data()[0] = std::move(t); }

  constexpr auto operator=(const std::initializer_list<T> &list)
    -> TinyVector & {
    invariant(list.size() <= ptrdiff_t(N));
    len = ptrdiff_t(list.size());
    std::copy(list.begin(), list.end(), data.data());
    return *this;
  }
  constexpr auto operator[](ptrdiff_t i) -> T & {
    invariant(i < len);
    return data.data()[i];
  }
  constexpr auto operator[](ptrdiff_t i) const -> const T & {
    invariant(i < len);
    return data.data()[i];
  }
  constexpr auto back() -> T & {
    invariant(len > 0);
    return data.data()[len - 1];
  }
  constexpr auto back() const -> const T & {
    invariant(len > 0);
    return data.data()[len - 1];
  }
  constexpr auto front() -> T & {
    invariant(len > 0);
    return data.data()[0];
  }
  constexpr auto front() const -> const T & {
    invariant(len > 0);
    return data.data()[0];
  }
  constexpr void push_back(const T &t) {
    invariant(len < ptrdiff_t(N));
    data.data()[len++] = t;
  }
  constexpr void push_back(T &&t) {
    invariant(len < ptrdiff_t(N));
    data.data()[len++] = std::move(t);
  }
  template <class... Args> constexpr void emplace_back(Args &&...args) {
    invariant(len < ptrdiff_t(N));
    data.data()[len++] = T(std::forward<Args>(args)...);
  }
  constexpr void pop_back() {
    invariant(len > 0);
    --len;
  }
  [[nodiscard]] constexpr auto size() const -> ptrdiff_t {
    ptrdiff_t l = ptrdiff_t(len);
    invariant(l >= 0);
    invariant(l <= ptrdiff_t(N));
    return l;
  }
  [[nodiscard]] constexpr auto empty() const -> bool { return len == 0; }
  constexpr void clear() { len = 0; }

  constexpr auto begin() -> T * { return data.data(); }
  constexpr auto begin() const -> const T * { return data.data(); }
  constexpr auto end() -> T * { return data.data() + size(); }
  constexpr auto end() const -> const T * { return data.data() + size(); }
};
} // namespace poly::containers
