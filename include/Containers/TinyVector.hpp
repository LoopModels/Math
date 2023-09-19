#pragma once

#include "Containers/Storage.hpp"
#include "Utilities/Invariant.hpp"
#include <cstddef>

namespace poly::containers {
using utils::invariant;
template <class T, size_t N> class TinyVector {
  static_assert(N > 0);
  Storage<T, N> data;
  ptrdiff_t len{};

public:
  constexpr TinyVector() = default;
  constexpr TinyVector(const std::initializer_list<T> &list) {
    invariant(list.size() <= N);
    len = list.size();
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
    invariant(len <= ptrdiff_t(N));
    return len;
  }
  [[nodiscard]] constexpr auto empty() const -> bool { return len == 0; }
  constexpr void clear() { len = 0; }

  constexpr auto begin() -> T * { return data.data(); }
  constexpr auto begin() const -> const T * { return data.data(); }
  constexpr auto end() -> T * { return data.data() + size(); }
  constexpr auto end() const -> const T * { return data.data() + size(); }
};
} // namespace poly::containers
