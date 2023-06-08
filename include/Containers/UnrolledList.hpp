#pragma once
#include "Utilities/Invariant.hpp"
#include <memory>
namespace poly::utils {
template <size_t SlabSize = 16384, bool BumpUp = false,
          size_t MinAlignment = alignof(std::max_align_t)>
class BumpAlloc; // forward declaration
} // namespace poly::utils

namespace poly::containers {
using utils::invariant;
template <typename T> class UList {
  T data[6]; // NOLINT(modernize-avoid-c-arrays)
  size_t count{0};
  UList<T> *next{nullptr};

public:
  constexpr UList() = default;
  constexpr UList(T t) : count(1) { data[0] = t; }
  constexpr UList(T t, UList *n) : count(1), next(n) { data[0] = t; }
  constexpr UList(const UList &other) = default;
  constexpr void forEach(const auto &f) {
    invariant(count <= std::size(data));
    for (auto *L = this; L != nullptr; L = L->next)
      for (size_t i = 0, N = L->count; i < N; i++) f(L->data[i]);
  }
  constexpr void forEachStack(const auto &f) {
    invariant(count <= std::size(data));
    // the motivation of this implementation is that we use this to
    // deallocate the list, which may contain pointers that themselves
    // allocated this.
    UList<T> C{*this};
    while (true) {
      for (size_t i = 0, N = C.count; i < N; i++) f(C.data[i]);
      if (C.next == nullptr) return;
      C = *C.next;
    }
  }
  constexpr void forEachNoRecurse(const auto &f) {
    invariant(count <= std::size(data));
    for (size_t i = 0; i < count; i++) f(data[i]);
  }
  constexpr auto reduce(auto init, const auto &f) const {
    invariant(count <= std::size(data));
    decltype(f(init, std::declval<T>())) acc = init;
    for (auto *L = this; L != nullptr; L = L->next)
      for (size_t i = 0, N = L->count; i < N; i++) acc = f(acc, L->data[i]);
    return acc;
  }
  constexpr auto transform_reduce(auto init, const auto &f) {
    invariant(count <= std::size(data));
    decltype(f(init, std::declval<T>())) acc = init;
    for (auto *L = this; L != nullptr; L = L->next)
      for (size_t i = 0, N = L->count; i < N; i++) acc = f(acc, L->data[i]);
    return acc;
  }
  constexpr void pushHasCapacity(T t) {
    invariant(count < std::size(data));
    data[count++] = t;
  }
  /// unordered push
  template <class A>
  [[nodiscard]] constexpr auto push(A &alloc, T t) -> UList * {
    invariant(count <= std::size(data));
    if (!isFull()) {
      data[count++] = t;
      return this;
    }
    UList<T> *other = alloc.allocate(1);
    std::construct_at(other, t, this);
    return other;
  }
  /// ordered push
  template <class A> constexpr void push_ordered(A &alloc, T t) {
    invariant(count <= std::size(data));
    if (!isFull()) {
      data[count++] = t;
      return;
    }
    if (next == nullptr) {
      next = alloc.allocate(1);
      std::construct_at(next, t);
    } else next->push_ordered(alloc, t);
  }
  [[nodiscard]] constexpr auto push(T t) -> UList * {
    std::allocator<UList<T>> alloc;
    return push(alloc, t);
  }
  /// ordered push
  constexpr void push_ordered(T t) {
    std::allocator<UList<T>> alloc;
    push_ordered(alloc, t);
  }
  [[nodiscard]] constexpr auto push(utils::BumpAlloc<> &alloc, T t) -> UList *;
  constexpr void push_ordered(utils::BumpAlloc<> &alloc, T t);
  constexpr auto copy(utils::BumpAlloc<> &alloc) const -> UList *;
  /// erase
  /// behavior is undefined if `x` doesn't point to this node
  constexpr void erase(T *x) {
    invariant(count <= std::size(data));
    for (auto i = x, e = data + --count; i != e; ++i) *i = *(i + 1);
  }
  /// eraseUnordered
  /// behavior is undefined if `x` doesn't point to this node
  constexpr void eraseUnordered(T *x) {
    invariant(count <= std::size(data));
    *x = data[--count];
  }
  [[nodiscard]] constexpr auto isFull() const -> bool {
    return count == std::size(data);
  }
  constexpr auto getNext() const -> UList * { return next; }
  constexpr void clear() {
    invariant(count <= std::size(data));
    count = 0;
    next = nullptr;
  }
  constexpr auto front() -> T & {
    invariant(count > 0);
    return data[0];
  }
  constexpr auto only() -> T & {
    invariant(count == 1);
    return data[0];
  }
  constexpr auto front() const -> const T & {
    invariant(count > 0);
    return data[0];
  }
  constexpr auto only() const -> const T & {
    invariant(count == 1);
    return data[0];
  }
  [[nodiscard]] constexpr auto empty() const -> bool { return count == 0; }
  [[nodiscard]] constexpr auto operator==(const UList &other) const -> bool {
    if (count != other.count) return false;
    for (size_t i = 0; i < count; i++)
      if (data[i] != other.data[i]) return false;
    if (next == nullptr && other.getNext() == nullptr) return true;
    if (next == nullptr || other.getNext() == nullptr) return false;
    return *next == *other.getNext();
  }
  constexpr auto operator[](size_t i) -> T & {
    return (i < count) ? data[i] : next->operator[](i - count);
  }
  constexpr auto operator=(const UList &other) -> UList & = default;
  struct End {};
  struct Iterator {
    UList *list;
    size_t index;
    constexpr auto operator==(End) const -> bool { return list == nullptr; }
    constexpr auto operator++() -> Iterator & {
      invariant(list != nullptr);
      if (++index == list->count) {
        list = list->next;
        index = 0;
      }
      return *this;
    }
    constexpr auto operator*() -> T & {
      invariant(list != nullptr);
      return list->data[index];
    }
    constexpr auto operator->() -> T * { return &**this; }
  };
  constexpr auto begin() -> Iterator { return {this, 0}; }
  static constexpr auto end() -> End { return {}; }
  constexpr auto dbegin() -> T * { return data; }
  constexpr auto dend() -> T * { return data + count; }
};
} // namespace poly::containers
