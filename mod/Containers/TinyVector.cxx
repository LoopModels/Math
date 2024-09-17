#ifdef USE_MODULE
module;
#else
#pragma once
#endif

#include "Owner.hxx"
#ifndef USE_MODULE
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <ostream>
#include <type_traits>

#include "Containers/Storage.cxx"
#include "Math/AxisTypes.cxx"
#include "Utilities/Invariant.cxx"
#else
export module TinyVector;

import AxisTypes;
import Invariant;
import MatDim;
import Storage;
import STL;
#endif

#ifdef USE_MODULE
export namespace containers {
#else
namespace containers {
#endif
using utils::invariant;

template <class T, size_t N, std::signed_integral L = ptrdiff_t>
class MATH_GSL_OWNER TinyVector {
  static_assert(N > 0);
  static_assert(std::numeric_limits<ptrdiff_t>::max() >= N);
  using Length = math::Length<-1, L>;
  Storage<T, N> data_;
  Length len_{};

public:
  using value_type = T;
  constexpr TinyVector() {}; // NOLINT (modernize-use-equals-default)
  constexpr TinyVector(const std::initializer_list<T> &list)
    : len_{math::length(L(list.size()))} {
    invariant(list.size() <= N);
    if constexpr (Storage<T, N>::trivial)
      std::copy_n(list.begin(), size(), data_.data());
    else std::uninitialized_move_n(list.begin(), size(), data_.data());
  }
  constexpr TinyVector(T t) : len_{math::length(L(1))} {
    std::construct_at(data_.data(), std::move(t));
  }

  constexpr auto
  operator=(const std::initializer_list<T> &list) -> TinyVector & {
    invariant(list.size() <= ptrdiff_t(N));
    ptrdiff_t slen = std::ssize(list), old_len = size();
    len_ = math::length(L(slen));
    auto I = list.begin();
    if constexpr (Storage<T, N>::trivial) {
      // implicit lifetime type
      std::copy_n(I, slen, data_.data());
    } else {
      ptrdiff_t J = std::min(slen, old_len);
      // old values exist
      std::move(I, I + J, data_.data());
      if (old_len < len_) {
        std::uninitialized_move_n(I + old_len, slen - old_len,
                                  data_.data() + old_len);
      } else if constexpr (!std::is_trivially_destructible_v<T>)
        if (old_len > len_) std::destroy_n(data_.data() + slen, old_len - slen);
    }
    return *this;
  }
  constexpr auto operator[](ptrdiff_t i) -> T & {
    invariant(i < len_);
    return data_.data()[i];
  }
  constexpr auto operator[](ptrdiff_t i) const -> const T & {
    invariant(i < len_);
    return data_.data()[i];
  }
  constexpr auto back() -> T & {
    invariant(len_ > 0);
    return data_.data()[size() - 1z];
  }
  constexpr auto back() const -> const T & {
    invariant(len_ > 0);
    return data_.data()[size() - 1z];
  }
  constexpr auto front() -> T & {
    invariant(len_ > 0);
    return data_.data()[0z];
  }
  constexpr auto front() const -> const T & {
    invariant(len_ > 0);
    return data_.data()[0z];
  }
  constexpr void push_back(const T &t) {
    invariant(len_ < ptrdiff_t(N));
    std::construct_at(data_.data() + ptrdiff_t(L(len_++)), t);
  }
  constexpr void push_back(T &&t) {
    invariant(len_ < ptrdiff_t(N));
    std::construct_at(data_.data() + ptrdiff_t(L(len_++)), std::move(t));
  }
  template <class... Args> constexpr auto emplace_back(Args &&...args) -> T & {
    invariant(len_ < ptrdiff_t(N));
    return *std::construct_at(data_.data() + ptrdiff_t(L(len_++)),
                              std::forward<Args>(args)...);
  }
  constexpr void pop_back() {
    invariant(len_ > 0);
    --len_;
    if constexpr (!std::is_trivially_destructible_v<T>)
      std::destroy_at(data_.data() + size());
  }
  [[nodiscard]] constexpr auto pop_back_val() -> T {
    invariant(len_ > 0);
    return std::move(data_.data()[ptrdiff_t(L(--len_))]);
  }
  [[nodiscard]] constexpr auto size() const -> ptrdiff_t {
    auto l = ptrdiff_t(L(len_));
    utils::assume(l <= ptrdiff_t(N));
    return l;
  }
  [[nodiscard]] constexpr auto empty() const -> bool { return len_ == 0z; }
  constexpr void clear() {
    if constexpr (!std::is_trivially_destructible_v<T>)
      std::destroy_n(data_.data(), size());
    len_ = Length{};
  }

  constexpr auto data() -> T * { return data_.data(); }
  constexpr auto data() const -> const T * { return data_.data(); }
  constexpr auto begin() -> T * { return data_.data(); }
  constexpr auto begin() const -> const T * { return data_.data(); }
  constexpr auto end() -> T * { return data_.data() + size(); }
  constexpr auto end() const -> const T * { return data_.data() + size(); }
  constexpr void resize(L new_size) {
    // initialize new data
    for (ptrdiff_t i = size(); i < new_size; ++i)
      std::construct_at(data_.data() + i);
    len_ = math::length(new_size);
  }
  constexpr void reserve(L space) {
    invariant(space >= 0);
    invariant(size_t(space) <= N);
  }
  constexpr ~TinyVector()
  requires(std::is_trivially_destructible_v<T>)
  = default;
  constexpr ~TinyVector()
  requires(!std::is_trivially_destructible_v<T>)
  {
    std::destroy_n(data_.data(), size());
  }

private:
  friend auto operator<<(std::ostream &os,
                         const TinyVector &x) -> std::ostream & {
    os << "[";
    if constexpr (std::same_as<T, int8_t> || std::same_as<T, uint8_t>) {
      if (!x.empty()) os << int(x[0]);
      for (L i = 1; i < x.size(); ++i) os << ", " << int(x[i]);
    } else {
      if (!x.empty()) os << x[0];
      for (L i = 1; i < x.size(); ++i) os << ", " << x[i];
    }
    return os << "]";
  }
};
} // namespace containers
