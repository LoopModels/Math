#ifdef USE_MODULE
module;
#else
#pragma once
#endif
#ifndef USE_MODULE
#include <cstddef>
#include <memory>

#include "Alloc/Arena.cxx"
#include "Utilities/Invariant.cxx"
#else
export module UnrolledList;

import Arena;
import Invariant;
import STL;
#endif

#ifdef USE_MODULE
export namespace containers {
#else
namespace containers {
#endif
using utils::invariant;
template <typename T> class UList {
  T data[6]; // NOLINT(modernize-avoid-c-arrays)
  ptrdiff_t count{0};
  UList<T> *next{nullptr};

public:
  constexpr UList() = default;
  constexpr UList(T t) : count(1) { data[0] = t; }
  constexpr UList(T t, UList *n) : count(1), next(n) { data[0] = t; }
  constexpr UList(const UList &other) = default;
  [[nodiscard]] constexpr auto getHeadCount() const -> ptrdiff_t {
    return count;
  }
  constexpr void forEach(const auto &f) {
    invariant(count <= std::ssize(data));
    for (auto *L = this; L != nullptr; L = L->next)
      for (ptrdiff_t i = 0, N = L->count; i < N; i++) f(L->data[i]);
  }
  constexpr void forEachRev(const auto &f) {
    invariant(count <= std::ssize(data));
    for (auto *L = this; L != nullptr; L = L->next)
      for (ptrdiff_t i = L->count; i;) f(L->data[--i]);
  }
  constexpr void forEachStack(const auto &f) {
    invariant(count <= std::ssize(data));
    // the motivation of this implementation is that we use this to
    // deallocate the list, which may contain pointers that themselves
    // allocated this.
    UList<T> C{*this};
    while (true) {
      for (ptrdiff_t i = 0, N = C.count; i < N; i++) f(C.data[i]);
      if (C.next == nullptr) return;
      C = *C.next;
    }
  }
  constexpr void forEachNoRecurse(const auto &f) {
    invariant(count <= std::ssize(data));
    for (ptrdiff_t i = 0; i < count; i++) f(data[i]);
  }
  constexpr auto reduce(auto init, const auto &f) const {
    invariant(count <= std::ssize(data));
    decltype(f(init, std::declval<T>())) acc = init;
    for (auto *L = this; L != nullptr; L = L->next)
      for (ptrdiff_t i = 0, N = L->count; i < N; i++) acc = f(acc, L->data[i]);
    return acc;
  }
  constexpr auto transform_reduce(auto init, const auto &f) {
    invariant(count <= std::ssize(data));
    decltype(f(init, std::declval<T &>())) acc = init;
    for (auto *L = this; L != nullptr; L = L->next)
      for (ptrdiff_t i = 0, N = L->count; i < N; i++) acc = f(acc, L->data[i]);
    return acc;
  }
  constexpr void pushHasCapacity(T t) {
    invariant(count < std::ssize(data));
    data[count++] = t;
  }
  /// unordered push
  template <class A>
  [[nodiscard]] constexpr auto push(A &alloc, T t) -> UList * {
    invariant(count <= std::ssize(data));
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
    invariant(count <= std::ssize(data));
    if (!isFull()) {
      data[count++] = t;
      return;
    }
    if (next == nullptr) {
      next = alloc.allocate(1);
      std::construct_at(next, t);
    } else next->push_ordered(alloc, t);
  }
  [[nodiscard]] constexpr auto contains(T t) const -> bool {
    invariant(count <= std::ssize(data));
    for (const UList *L = this; L; L = L->getNext())
      for (size_t i = 0, N = L->getHeadCount(); i < N; ++i)
        if (data[i] == t) return true;
    return false;
  }
  /// pushUnique(allocator, t)
  /// pushes `t` if it is unique
  template <class A>
  [[nodiscard]] constexpr auto pushUnique(A &alloc, T t) -> UList * {
    if (contains(t)) return this;
    return push(alloc, t);
  }

  // too dangerous
  // [[nodiscard]] constexpr auto push(T t) -> UList * {
  //   std::allocator<UList<T>> alloc;
  //   return push(alloc, t);
  // }
  // /// ordered push
  // constexpr void push_ordered(T t) {
  //   std::allocator<UList<T>> alloc;
  //   push_ordered(alloc, t);
  // }
  [[nodiscard]] constexpr auto push(alloc::Arena<> *alloc, T t) -> UList * {
    invariant(count <= std::ssize(data));
    if (!isFull()) {
      data[count++] = t;
      return this;
    }
    return alloc->create<UList<T>>(t, this);
  };
  constexpr void push_ordered(alloc::Arena<> *alloc, T t) {
    invariant(count <= std::ssize(data));
    if (!isFull()) {
      data[count++] = t;
      return;
    }
    if (next == nullptr) next = alloc->create<UList<T>>(t);
    else next->push_ordered(alloc, t);
  }
  constexpr auto copy(alloc::Arena<> *alloc) const -> UList * {
    UList<T> *L = alloc->create<UList<T>>();
    L->count = count;
    std::copy(std::begin(data), std::end(data), std::begin(L->data));
    if (next) L->next = next->copy(alloc);
    return L;
  }
  /// erase
  /// behavior is undefined if `x` doesn't point to this node
  constexpr void erase(T *x) {
    invariant(count <= std::ssize(data));
    for (auto i = x, e = data + --count; i != e; ++i) *i = *(i + 1);
  }
  /// eraseUnordered
  /// behavior is undefined if `x` doesn't point to this node
  constexpr void eraseUnordered(T *x) {
    invariant(count <= std::ssize(data));
    *x = data[--count];
  }
  constexpr auto searchHead(T x) -> T * {
    for (auto *d = data, *e = d + count; d != e; ++d)
      if (*d == x) return d;
    return nullptr;
  }
  //
  constexpr void eraseUnordered(T x) {
    invariant(count || next != nullptr);
    if (!count) next->eraseUnordered(x);
    if (T *p = searchHead(x)) return eraseUnordered(p);
    // not in head -> search next until we find it;
    // move last here there.
    invariant(next != nullptr);
    next->swapWith(x, std::move(data[--count]));
  }
  // search for `x`, swap with `y`.
  void swapWith(T x, T y) {
    for (UList *L = this; L; L->getNext()) {
      if (T *p = searchHead(x)) {
        *p = y;
        return;
      }
    }
  }
  [[nodiscard]] constexpr auto isFull() const -> bool {
    return count == std::ssize(data);
  }
  [[nodiscard]] constexpr auto getNext() const -> UList * { return next; }
  constexpr void clear() {
    invariant(count <= std::ssize(data));
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
  [[nodiscard]] constexpr auto front() const -> const T & {
    invariant(count > 0);
    return data[0];
  }
  [[nodiscard]] constexpr auto only() const -> const T & {
    invariant(count == 1);
    return data[0];
  }
  constexpr void append(UList *L) {
    UList *N = this;
    while (N->next != nullptr) N = N->next;
    N->next = L;
  }
  [[nodiscard]] constexpr auto empty() const -> bool { return count == 0; }
  [[nodiscard]] constexpr auto operator==(const UList &other) const -> bool {
    if (count != other.count) return false;
    for (ptrdiff_t i = 0; i < count; i++)
      if (data[i] != other.data[i]) return false;
    if (next == nullptr && other.getNext() == nullptr) return true;
    if (next == nullptr || other.getNext() == nullptr) return false;
    return *next == *other.getNext();
  }
  constexpr auto operator[](ptrdiff_t i) -> T & {
    return (i < count) ? data[i] : next->operator[](i - count);
  }
  constexpr auto operator[](ptrdiff_t i) const -> const T & {
    return (i < count) ? data[i] : next->operator[](i - count);
  }
  constexpr auto operator=(const UList &other) -> UList & = default;
  struct End {};
  struct MutIterator {

    UList *list;
    ptrdiff_t index;
    constexpr auto operator==(End) const -> bool { return list == nullptr; }
    constexpr auto operator==(MutIterator other) const -> bool {
      return list == other.list && index == other.index;
    }
    constexpr auto operator++() -> MutIterator & {
      invariant(list != nullptr);
      if (++index == list->count) {
        list = list->next;
        index = 0;
      }
      return *this;
    }
    constexpr auto operator++(int) -> MutIterator {
      invariant(list != nullptr);
      auto ret = *this;
      ++*this;
      return ret;
    }
    constexpr auto operator*() -> T & {
      invariant(list != nullptr);
      return list->data[index];
    }
    constexpr auto operator->() -> T * { return &**this; }
  };
  struct Iterator {
    const UList *list;
    ptrdiff_t index;
    constexpr auto operator==(End) const -> bool { return list == nullptr; }
    constexpr auto operator==(Iterator other) const -> bool {
      return list == other.list && index == other.index;
    }
    constexpr auto operator++() -> Iterator & {
      invariant(list != nullptr);
      if (++index == list->count) {
        list = list->next;
        index = 0;
      }
      return *this;
    }
    constexpr auto operator++(int) -> Iterator {
      invariant(list != nullptr);
      auto ret = *this;
      ++*this;
      return ret;
    }
    constexpr auto operator*() -> T {
      invariant(list != nullptr);
      return list->data[index];
    }
    constexpr auto operator->() -> const T * { return &**this; }
  };
  [[nodiscard]] constexpr auto begin() const -> Iterator { return {this, 0}; }
  [[nodiscard]] constexpr auto begin() -> MutIterator { return {this, 0}; }
  static constexpr auto end() -> End { return {}; }
  constexpr auto dbegin() -> T * { return data; }
  constexpr auto dend() -> T * { return data + count; }
};

} // namespace containers
