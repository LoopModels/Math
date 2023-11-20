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
///
/// The canonical type is the decompressed form.
///

/// `T` is the canonical type, which may define `compress`
template <typename T>
concept Compressible =
  (!std::same_as<T, typename T::compressed_type>)&&requires(
    T t, typename T::compressed_type *p) {
    { t.compress(p) };
    { T::decompress(p) } -> std::same_as<T>;
    { t = *p }; // need generic code to work reasonably well with pointers `p`
    { T{*p} };  // and value_type `T`
  };

template <typename T> struct Uncompressed {
  using compressed = T;
};
template <Compressible T> struct Uncompressed<T> {
  using compressed = typename T::compressed_type;
};
template <typename T>
using compressed_t = typename Uncompressed<std::remove_cvref_t<T>>::compressed;

template <typename T>
concept Decompressible =
  Compressible<typename T::decompressed_type> &&
  std::same_as<T, compressed_t<typename T::decompressed_type>>;

template <typename T> struct Compressed {
  using uncompressed = T;
};
template <Decompressible T> struct Compressed<T> {
  using uncompressed = typename T::decompressed_type;
};
template <typename T>
using decompressed_t =
  typename Compressed<std::remove_cvref_t<T>>::uncompressed;

static_assert(std::same_as<decompressed_t<double>, double>);
static_assert(!Decompressible<double>);
static_assert(!Compressible<double>);

template <typename T> constexpr void compress(const T &x, compressed_t<T> *p) {
  if constexpr (Compressible<T>) x.compress(p);
  else *p = x;
}
template <typename T>
constexpr auto decompress(const compressed_t<T> *p) -> decltype(auto) {
  if constexpr (Compressible<T>) return T::decompress(p);
  else return *p;
}

} // namespace poly::utils
