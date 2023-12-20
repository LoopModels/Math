#include "Alloc/Arena.hpp"
#include "Math/Array.hpp"
#include "Math/Math.hpp"
#include "Math/MatrixDimensions.hpp"
#include "Math/SmallSparseMatrix.hpp"
#include "Math/StaticArrays.hpp"
#include "Utilities/MatrixStringParse.hpp"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <ostream>
#include <sstream>

using namespace poly::math;
using poly::utils::operator""_mat;
// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(SparseIndexingTest, BasicAssertions) {
  SmallSparseMatrix<int64_t> sparseA(Row<>{3}, Col<>{4});
  std::cout << "&Asparse = " << &sparseA << "\n";
  sparseA[0, 1] = 5;
  sparseA[1, 3] = 3;
  sparseA[2, 0] = -1;
  sparseA[2, 1] = 4;
  sparseA[2, 2] = -2;
  IntMatrix<> A = sparseA;
  {
    IntMatrix<> A2(DenseDims<>{{3}, {4}});
    MutPtrMatrix<int64_t> MA2 = A2;
    MA2 << sparseA;
    EXPECT_EQ(A, A2);
  }
  for (ptrdiff_t i = 0; i < 3; ++i) {
    for (ptrdiff_t j = 0; j < 4; ++j) {
      int64_t a = A[i, j], b = sparseA[i, j];
      EXPECT_EQ(a, b);
    }
  }
  // EXPECT_EQ(A(i, j), Asparse(i, j));
  ManagedArray B(std::type_identity<int64_t>{}, DenseDims<>{{4}, {5}});
  EXPECT_FALSE(B.isSquare());
  B[0, 0] = 3;
  B[0, 1] = -1;
  B[0, 2] = 0;
  B[0, 3] = -5;
  B[0, 4] = 1;
  B[1, 0] = -4;
  B[1, 1] = 5;
  B[1, 2] = -1;
  B[1, 3] = -1;
  B[1, 4] = -1;
  B[2, 0] = 1;
  B[2, 1] = 2;
  B[2, 2] = -5;
  B[2, 3] = 2;
  B[2, 4] = 3;
  B[3, 0] = -2;
  B[3, 1] = 1;
  B[3, 2] = 2;
  B[3, 3] = -3;
  B[3, 4] = 5;
  IntMatrix<> C{DenseDims<>{{3}, {5}}};
  C[0, 0] = -20;
  C[0, 1] = 25;
  C[0, 2] = -5;
  C[0, 3] = -5;
  C[0, 4] = -5;
  C[1, 0] = -6;
  C[1, 1] = 3;
  C[1, 2] = 6;
  C[1, 3] = -9;
  C[1, 4] = 15;
  C[2, 0] = -21;
  C[2, 1] = 17;
  C[2, 2] = 6;
  C[2, 3] = -3;
  C[2, 4] = -11;
  EXPECT_EQ(A.numRow(), (A * B).numRow());
  EXPECT_EQ(B.numCol(), (A * B).numCol());
  EXPECT_TRUE(C == A * B);
  IntMatrix<> C2{A * B};
  std::cout << "C=" << C << "\nC2=" << C2 << "\n";
  EXPECT_TRUE(C == C2);
  IntMatrix<> At{A.t()}, Bt{B.t()};
  // At << A.t();
  // Bt << B.t();
  C2 += At.t() * Bt.t();
  EXPECT_EQ(C * 2, C2);
  EXPECT_EQ(C, At.t() * B);
  EXPECT_EQ(C, A * Bt.t());
  EXPECT_EQ(C, At.t() * Bt.t());
  C2 -= A * Bt.t();
  EXPECT_EQ(C, C2);
  int64_t i = 0;
  IntMatrix<> D{C};
  std::cout << "C=" << C << "\n";
  static_assert(std::same_as<decltype(D[0, _]), MutArray<int64_t, ptrdiff_t>>);
  for (ptrdiff_t r : _(0, D.numRow())) D[r, _] += ptrdiff_t(r) + 1;
  for (auto r : C.eachRow()) {
    EXPECT_EQ(r.size(), ptrdiff_t(C.numCol()));
    r += (++i);
  }
  EXPECT_EQ(C, D);
  for (ptrdiff_t c : _(0, D.numCol()))
    D[_, c] += ptrdiff_t(c) + ptrdiff_t(D.numRow()) + 1;
  for (auto c : C.eachCol()) c += (++i);
  EXPECT_EQ(C, D);
}

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(ExpressionTemplateTest, BasicAssertions) {
  auto A{
    "[3 -5 1 10 -4 6 4 4; 4 6 3 -1 6 1 -4 0; -7 -2 0 0 -10 -2 3 7; 2 -7 -5 "
    "-5 -7 -5 1 -7; 2 -8 2 7 4 9 6 -3; -2 -8 -5 0 10 -4 5 -3]"_mat};

  auto A4{
    "[12 -20 4 40 -16 24 16 16; 16 24 12 -4 24 4 -16 0; -28 -8 0 0 -40 -8 "
    "12 28; 8 -28 -20 -20 -28 -20 4 -28; 8 -32 8 28 16 36 24 -12; -8 -32 "
    "-20 0 40 -16 20 -12]"_mat};
  // IntMatrix B{A*4};
  auto templateA4{A * 4};
  IntMatrix<> C{templateA4};
  IntMatrix<> B{A * 4};
  EXPECT_EQ(A4, B);
  EXPECT_EQ(A4, C);
  IntMatrix<> Z = A * 4 - A4;
  for (ptrdiff_t i = 0; i < Z.numRow(); ++i)
    for (ptrdiff_t j = 0; j < Z.numCol(); ++j) EXPECT_FALSE((Z[i, j]));
  auto D{
    "[-5 6 -1 -4 7 -9 6; -3 -5 -1 -2 -9 -4 -1; -4 7 -6 10 -2 2 9; -4 -7 -1 "
    "-7 5 9 -10; 5 -7 -5 -1 -3 -8 -8; 3 -6 4 10 9 0 -5; 0 -1 4 -4 -9 -3 "
    "-10; 2 1 4 5 -7 0 -8]"_mat};
  auto refAD{
    "[-38 -28 62 6 116 105 -138; -13 -22 -69 29 -10 -99 42; -1 54 91 45 "
    "-95 142 -36; -13 118 31 -91 78 8 151; 19 -74 15 26 153 31 -145; 86 "
    "-61 -18 -111 -22 -55 -135]"_mat};
  IntMatrix<> AD = A * D;
  EXPECT_EQ(AD, refAD);
  IntMatrix<> E{
    "[-4 7 9 -4 2 9 -8; 3 -5 6 0 -1 8 7; -7 9 -1 1 -5 2 10; -3 10 -10 -3 6 "
    "5 5; -6 7 -4 -7 10 5 3; 9 -8 7 9 2 2 6]"_mat};
  IntMatrix<> m7EpAD = A * D - 7 * E;
  auto refADm7E{
    "[-10 -77 -1 34 102 42 -82; -34 13 -111 29 -3 -155 -7; 48 -9 98 38 -60 "
    "128 -106; 8 48 101 -70 36 -27 116; 61 -123 43 75 83 -4 -166; 23 -5 "
    "-67 -174 -36 -69 -177]"_mat};
  EXPECT_EQ(m7EpAD, refADm7E);

  Vector<int64_t> a;
  a.push_back(-8);
  a.push_back(7);
  a.push_back(3);
  Vector<int64_t> b = a * 2;
  Vector<int64_t> c;
  c.push_back(-16);
  c.push_back(14);
  c.push_back(6);
  EXPECT_EQ(b, c);
  IntMatrix<> dA1x1(DenseDims<>{{1}, {1}}, 0);
  EXPECT_TRUE(dA1x1.isSquare());
  IntMatrix<> dA2x2(DenseDims<>{{2}, {2}}, 0);
  dA1x1.antiDiag() << 1;
  EXPECT_EQ((dA1x1[0, 0]), 1);
  dA2x2.antiDiag() << 1;
  EXPECT_EQ((dA2x2[0, 0]), 0);
  EXPECT_EQ((dA2x2[0, 1]), 1);
  EXPECT_EQ((dA2x2[1, 0]), 1);
  EXPECT_EQ((dA2x2[1, 1]), 0);
  for (ptrdiff_t i = 1; i < 20; ++i) {
    IntMatrix<> F(DenseDims<>{{i}, {i}});
    F << 0;
    F.antiDiag() << 1;
    for (ptrdiff_t j = 0; j < i; ++j)
      for (ptrdiff_t k = 0; k < i; ++k) EXPECT_EQ((F[j, k]), k + j == i - 1);
  }
}
// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(ExpressionTemplateTest2, BasicAssertions) {
  ManagedArray<double, DenseDims<>> W{{{3}, {3}}, 0}, X{{{3}, {3}}, 0},
    Y{{{3}, {3}}, 0}, Z{{{3}, {3}}, 0};
  W[0, 0] = 0.29483432115939806;
  W[0, 1] = 1.5777027461040212;
  W[0, 2] = 0.8171761007267028;
  W[1, 0] = 1.0463632179853855;
  W[1, 1] = 0.9503214631611095;
  W[1, 2] = -0.17890983978584624;
  W[2, 0] = 1.5853551451194254;
  W[2, 1] = -0.784875301203305;
  W[2, 2] = 1.7033024094365752;
  X[0, 0] = -1.1175097244313117;
  X[0, 1] = -0.21769215316295054;
  X[0, 2] = -0.7340630927749082;
  X[1, 0] = -0.5750426169922397;
  X[1, 1] = 0.27174064995044767;
  X[1, 2] = -1.0669896577273217;
  X[2, 0] = 0.9302424251181362;
  X[2, 1] = -1.3157431480603476;
  X[2, 2] = 1.546836705770486;
  Y[0, 0] = 1.1701212478097331;
  Y[0, 1] = 0.7747688878004019;
  Y[0, 2] = -0.926815554991563;
  Y[1, 0] = -1.4441713498640656;
  Y[1, 1] = -1.3615487160168993;
  Y[1, 2] = 0.7908183008408143;
  Y[2, 0] = -0.7626497248468547;
  Y[2, 1] = -0.21682371102755368;
  Y[2, 2] = -0.07604892144743511;
  Z[0, 0] = 3.3759933640164708e16;
  Z[0, 1] = 9.176788687153845e14;
  Z[0, 2] = -1.1081818546676994e15;
  Z[1, 0] = -1.7207794047001762e15;
  Z[1, 1] = 3.0768637505289172e16;
  Z[1, 2] = 9.277082601207064e14;
  Z[2, 0] = -8.956589651911538e14;
  Z[2, 1] = -2.7136623944168075e14;
  Z[2, 2] = 3.2308470074953084e16;
  ManagedArray<double, DenseDims<>> A{
    W * (W + 16380 * X + 40840800 * Y) +
    (33522128640 * W + 10559470521600 * X + 1187353796428800 * Y) +
    32382376266240000 * I};

  EXPECT_EQ(A, Z);
}

// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(ArrayPrint, BasicAssertions) {
  {
    std::stringstream os;
    // std::basic_stringstream os(s);
    auto A{
      "[3 -5 1 10 -4 6 4 4; 4 6 3 -1 6 1 -4 0; -7 -2 0 0 -10 -2 3 7; 2 -7 -5 "
      "-5 -7 -5 1 -7; 2 -8 2 7 4 9 6 -3; -2 -8 -5 0 10 -4 5 -3]"_mat};
    os << A;
    std::cout << "std::cout << A yields:" << A << '\n';
    std::cout << "PrintTo(A, &std::cout) yields:";
    testing::internal::PrintTo(A, &std::cout);
    std::cout << '\n';
    EXPECT_EQ(os.str(), testing::PrintToString(A));
  }
  {
    std::stringstream os;
    Vector<int64_t> v;
    for (ptrdiff_t i = 0; i < 10; ++i) v.push_back(i);
    os << v;
    EXPECT_EQ(os.str(), testing::PrintToString(v));
  }
}
// NOLINTNEXTLINE(modernize-use-trailing-return-type)
TEST(OffsetEnd, BasicAssertions) {
  auto A{"[3 3 3 3; 2 2 2 2; 1 1 1 1; 0 0 0 0]"_mat};
  auto B = IntMatrix<>{DenseDims<>{{4}, {4}}};
  for (ptrdiff_t i = 0; i < 4; ++i) B[last - i, _] << i;
  EXPECT_EQ(A, B);
}
TEST(SquareMatrixTest, BasicAssertions) {
  SquareMatrix<int64_t> A{SquareDims<>{{4}}};
  for (ptrdiff_t i = 0; i < 4; ++i)
    for (ptrdiff_t j = 0; j < 4; ++j) A[i, j] = 4 * i + j;
  DenseMatrix<int64_t> B{DenseDims<>{{4}, {2}}};
  B << A[_(end - 2, end), _].t();
  for (ptrdiff_t j = 0; j < 4; ++j)
    for (ptrdiff_t i = 0; i < 2; ++i) EXPECT_EQ((B[j, i]), 4 * (i + 2) + j);
}
TEST(VectorTest, BasicAssertions) {
  poly::alloc::OwningArena<> alloc;
  ResizeableView<int64_t, ptrdiff_t> x;
  for (size_t i = 0; i < 100; ++i) {
    if (x.getCapacity() <= x.size())
      x.reserve(&alloc, std::max<ptrdiff_t>(8, 2 * x.size()));
    x.emplace_back(i);
  }
  EXPECT_EQ(x.size(), 100);
  EXPECT_EQ(x.sum(), 100 * 99 / 2);
}
TEST(SVectorTest, BasicAssertions) {
  SVector<int64_t, 3> x{1, 2, 3};
  static_assert(poly::utils::Compressible<SVector<int64_t, 3>>);
  static_assert(std::tuple_size_v<decltype(x)> == 3);
  static_assert(std::same_as<std::tuple_element_t<2, decltype(x)>, int64_t>);
  SVector<int64_t, 3> y{10, 20, 30};
  SVector<int64_t, 3> z{11, 22, 33};
  SVector<int64_t, 3> w = x + y;
  EXPECT_EQ(w, z);
  EXPECT_TRUE(w == z);
  constexpr auto constCmp = [](auto const &a, auto const &b) {
    return std::make_pair(a == b, std::is_constant_evaluated());
  };
  Vector<int64_t> v{std::array{1, 2, 3}};
  EXPECT_TRUE(constCmp(v.size(), unsigned(3)).first);
  EXPECT_FALSE(constCmp(v.size(), unsigned(3)).second);
  EXPECT_TRUE(constCmp(x.size(), unsigned(3)).first);
  // EXPECT_TRUE(constCmp(v.size()).first);
  // EXPECT_FALSE(constCmp(v.size()).second);
  // EXPECT_TRUE(constCmp(x.size()).first);
  static_assert(constCmp(decltype(x)::size(), unsigned(3)).second);
  static_assert(constCmp(decltype(x)::size(), decltype(y)::size()).second);
  // EXPECT_TRUE(constCmp(x.size()).second);
  // EXPECT_TRUE(constCmp(x.size(), unsigned(3)).second);
  // EXPECT_TRUE(constCmp(x.size(), y.size()).second);
  auto [a, b, c] = w;
  EXPECT_EQ(a, 11);
  EXPECT_EQ(b, 22);
  EXPECT_EQ(c, 33);
}
