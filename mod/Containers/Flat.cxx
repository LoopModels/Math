#ifdef USE_MODULE
module;
#else
#pragma once
#endif
#include "Owner.hxx"
#ifndef USE_MODULE
#include <cstddef>
#include <memory>
#include <type_traits>

#include "Alloc/Mallocator.cxx"
#include "Utilities/Invariant.cxx"
#else
export module Flat;

import Allocator;
import Invariant;
import STL;
#endif

#ifdef USE_MODULE
export namespace containers {
#else
namespace containers {
#endif
template <typename T> struct MATH_GSL_OWNER Flat {
  static_assert(std::is_same_v<T, std::remove_cvref_t<T>>);
  explicit constexpr Flat(ptrdiff_t len)
    : ptr_{alloc::Mallocator<T>{}.allocate(len)}, len_{len} {
    std::uninitialized_default_construct_n(ptr_, len_);
  };
  explicit constexpr Flat(ptrdiff_t len, T x)
    : ptr_{alloc::Mallocator<T>{}.allocate(len)}, len_{len} {
    std::uninitialized_fill_n(ptr_, len_, x);
  };

  constexpr Flat(const Flat &other)
    : ptr_{alloc::Mallocator<T>{}.allocate(other.size())}, len_{other.size()} {
    std::uninitialized_copy_n(other.data(), len_, data());
  };
  constexpr Flat(auto B, auto E)
    : ptr_{alloc::Mallocator<T>{}.allocate(std::distance(B, E))},
      len_{std::distance(B, E)} {
    std::uninitialized_copy(B, E, data());
  };
  constexpr auto operator=(const Flat &other) -> Flat & {
    if (this == &other) return *this;
    if (len_ != other.size()) {
      maybeDeallocate();
      ptr_ = alloc::Mallocator<T>{}.allocate(len_);
      len_ = other.size();
      if constexpr (!std::is_trivially_default_constructible_v<T>) {
        std::uninitialized_copy_n(other.data(), len_, ptr_);
        return *this;
      }
    }
    std::copy_n(other.data(), len_, data());
    return *this;
  };
  constexpr Flat(Flat &&other) : ptr_{other.data()}, len_{other.size()} {
    other.ptr_ = nullptr;
    other.len_ = 0;
  };
  constexpr auto operator=(Flat &&other) -> Flat & {
    if (this == &other) return *this;
    ptr_ = other.ptr_;
    len_ = other.size();
    other.ptr_ = nullptr;
    other.len_ = 0;
    return *this;
  }

  constexpr ~Flat() { maybeDeallocate(); }
  constexpr auto data() -> T * { return ptr_; }
  constexpr auto data() const -> const T * { return ptr_; }
  [[nodiscard]] constexpr auto size() const -> ptrdiff_t {
    utils::invariant(len_ >= 0);
    return len_;
  }
  constexpr auto operator[](ptrdiff_t i) -> T & {
    utils::invariant((i >= 0) && i < size());
    return ptr_[i];
  }
  constexpr auto operator[](ptrdiff_t i) const -> const T & {
    utils::invariant((i >= 0) && i < size());
    return ptr_[i];
  }
  constexpr auto begin() -> T * { return ptr_; }
  constexpr auto end() -> T * { return ptr_ + len_; }
  constexpr auto begin() const -> const T * { return ptr_; }
  constexpr auto end() const -> const T * { return ptr_ + len_; }

private:
  constexpr void maybeDeallocate() {
    if (!ptr_) return;
    if constexpr (!std::is_trivially_destructible_v<T>)
      std::destroy_n(ptr_, len_);
    alloc::Mallocator<T>{}.deallocate(ptr_, len_);
  }

  T *ptr_{nullptr};
  ptrdiff_t len_{0};
};
} // namespace containers
