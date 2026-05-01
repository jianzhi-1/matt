#include "matt/tensor.hpp"
#include <gtest/gtest.h>

using namespace matt;

class TensorTest : public ::testing::Test {
  protected:
    Tensor zeros_2x3 = Tensor::zeros({2, 3});
    Tensor ones_2x2 = Tensor::ones({2, 2});
    Tensor arange_6 = Tensor::arange(0, 6, 1);
};

// ── Creation ──────────────────────────────────────────────────────────────────
TEST_F(TensorTest, ZerosShape) {
    EXPECT_EQ(zeros_2x3.shape(), (std::vector<size_t>{2, 3}));
    EXPECT_EQ(zeros_2x3.ndim(), 2);
    EXPECT_EQ(zeros_2x3.numel(), 6);
}

TEST_F(TensorTest, ZerosValues) {
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            EXPECT_FLOAT_EQ(zeros_2x3.at({i, j}), 0.0f);
        }
    }
}

TEST_F(TensorTest, OnesValues) {
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            EXPECT_FLOAT_EQ(ones_2x2.at({i, j}), 1.0f);
        }
    }
}

TEST_F(TensorTest, FromData) {
    auto t = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({1, 2}), 6.0f);
}

TEST_F(TensorTest, FromDataSizeMismatchThrows) {
    EXPECT_THROW(Tensor::from_data({1, 2, 3}, {2, 3}), std::runtime_error);
}

// ── Strides ───────────────────────────────────────────────────────────────────
TEST_F(TensorTest, RowMajorStrides) {
    auto t = arange_6.reshape({2, 3});
    EXPECT_EQ(t.strides()[0], 3);
    EXPECT_EQ(t.strides()[1], 1);
}

TEST_F(TensorTest, StrideElementAccess) {
    auto t = arange_6.reshape({2, 3});
    EXPECT_FLOAT_EQ(t.at({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(t.at({0, 2}), 2.0f);
    EXPECT_FLOAT_EQ(t.at({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(t.at({1, 2}), 5.0f);
}

// ── Reshape ───────────────────────────────────────────────────────────────────
TEST_F(TensorTest, ReshapeShape) {
    auto r = arange_6.reshape({3, 2});
    EXPECT_EQ(r.shape(), (std::vector<size_t>{3, 2}));
}

TEST_F(TensorTest, ReshapePreservesValues) {
    auto r = arange_6.reshape({2, 3});
    EXPECT_FLOAT_EQ(r.at({1, 2}), 5.0f);
}

TEST_F(TensorTest, ReshapeMismatchThrows) {
    EXPECT_THROW(arange_6.reshape({4, 2}), std::runtime_error);
}

// ── Transpose ─────────────────────────────────────────────────────────────────
TEST_F(TensorTest, TransposeShape) {
    auto t = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    auto tt = t.transpose(0, 1);
    EXPECT_EQ(tt.shape(), (std::vector<size_t>{3, 2}));
}

TEST_F(TensorTest, TransposeValues) {
    auto t = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    auto tt = t.transpose(0, 1);
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 2; j++) {
            EXPECT_FLOAT_EQ(tt.at({i, j}), t.at({j, i}));
        }
    }
}

TEST_F(TensorTest, TransposeIsView) {
    auto t = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    auto tt = t.transpose(0, 1);
    tt.at({0, 0}) = 99.0f;
    EXPECT_FLOAT_EQ(t.at({0, 0}), 99.0f);
}

TEST_F(TensorTest, TransposeInvalidDimThrows) {
    EXPECT_THROW(zeros_2x3.transpose(0, 5), std::out_of_range);
}

// ── Contiguous ────────────────────────────────────────────────────────────────
TEST_F(TensorTest, FreshTensorIsContiguous) {
    EXPECT_TRUE(zeros_2x3.is_contiguous());
}

TEST_F(TensorTest, TransposeIsNotContiguous) {
    auto tt = zeros_2x3.transpose(0, 1);
    EXPECT_FALSE(tt.is_contiguous());
}

TEST_F(TensorTest, MakeContiguousCopiesCorrectly) {
    auto t = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3}).transpose(0, 1);
    auto tc = t.contiguous();
    EXPECT_TRUE(tc.is_contiguous());
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 2; j++) {
            EXPECT_FLOAT_EQ(tc.at({i, j}), t.at({i, j}));
        }
    }
}
