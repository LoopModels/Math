#pragma once

namespace poly::utils {

/// The idea here is that some types may have a compression/decompression that
/// is potentially costly. To work around this, temporaries can live in
/// uncompressed form, while longer term storage can be compressed/decompressed.
/// E.g., we have have an array of a compressed type.
/// When loading, the uncompressed form is returned.
/// The particular motivating example is around using SIMD types with arrays
/// that are themselves not a perfect multiple of the SIMD-width.
/// Writing/reading from temporary locations thus requires masking. For
/// compilers to be able to optimize these loads and stores, they'd have need to
/// track provenance using these masks to see that they don't alias, and be able
/// to actually eliminate the temporaries. Compilers do not seem to do this.

template <typename T>
concept DefinesCompress = requires(T *t) {
  { T::decompress(t) };
  { T::compress(t, T::decompress(t)) };
};

template <typename T> struct Uncompressed {
  using uncompressed = T;
};
template <DefinesCompress T> struct Uncompressed<T> {
  using uncompressed = decltype(T::decompress(nullptr));
};
template <typename T>
using uncompressed_t = typename Uncompressed<T>::uncompressed;

template <typename T>
concept Compressible =
  !std::same_as<T, uncompressed_t<T>> && requires(T *t, uncompressed_t<T> u) {
    { T::decompress(t) } -> std::same_as<uncompressed_t<T>>;
    { T::compress(t, u) };
  };

template <Compressible T>
[[gnu::always_inline]] constexpr auto decompress(const T *t)
  -> uncompressed_t<T> {
  return T::decompress(t);
}
template <typename T>
[[gnu::always_inline]] constexpr auto decompress(const T *t) -> const T & {
  return *t;
}
template <typename T>
[[gnu::always_inline]] constexpr void compress(T *t,
                                               const uncompressed_t<T> &u) {
  if constexpr (Compressible<T>) T::compress(t, u);
  else *t = u;
}

} // namespace poly::utils
