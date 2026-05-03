#include "matt/ops.hpp"
#include "matt/shape_utils.hpp"
#include "matt/tensor.hpp"
#include <gtest/gtest.h>

using namespace matt;

class TensorTest : public ::testing::Test {
  protected:
    Tensor zeros_2x3 = Tensor::zeros({2, 3});
    Tensor ones_2x2 = Tensor::ones({2, 2});
    Tensor arange_6 = Tensor::arange(0, 6, 1);
};

class SliceTest : public ::testing::Test {
  protected:
    // [[1, 2, 3],
    //  [4, 5, 6],
    //  [7, 8, 9]]
    Tensor t3x3 = Tensor::from_data({1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3});
};

class BroadcastTest : public ::testing::Test {
  protected:
    // scalar-like
    Tensor scalar = Tensor::from_data({3.0f}, {1});

    // 1D
    Tensor vec3 = Tensor::from_data({1, 2, 3}, {3});

    // 2D
    Tensor mat2x3 = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor mat3x1 = Tensor::from_data({1, 2, 3}, {3, 1});
    Tensor mat1x3 = Tensor::from_data({1, 2, 3}, {1, 3});
    Tensor mat2x1 = Tensor::from_data({10, 20}, {2, 1});
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

// ── Row slice ─────────────────────────────────────────────────────────────────

TEST_F(SliceTest, RowSliceShape) {
    auto row = t3x3.slice(0, 1); // second row
    EXPECT_EQ(row.shape(), (std::vector<size_t>{3}));
}

TEST_F(SliceTest, RowSliceValues) {
    auto row = t3x3.slice(0, 1); // [4, 5, 6]
    EXPECT_FLOAT_EQ(row.at({0}), 4.0f);
    EXPECT_FLOAT_EQ(row.at({1}), 5.0f);
    EXPECT_FLOAT_EQ(row.at({2}), 6.0f);
}

TEST_F(SliceTest, RowSliceIsView) {
    auto row = t3x3.slice(0, 1);
    row.at({0}) = 99.0f;
    EXPECT_FLOAT_EQ(t3x3.at({1, 0}), 99.0f); // same storage
}

TEST_F(SliceTest, RowSliceOffset) {
    auto row = t3x3.slice(0, 1);
    EXPECT_EQ(row.offset(), 3); // second row starts at index 3
}

TEST_F(SliceTest, FirstRowSlice) {
    auto row = t3x3.slice(0, 0);
    EXPECT_FLOAT_EQ(row.at({0}), 1.0f);
    EXPECT_FLOAT_EQ(row.at({2}), 3.0f);
    EXPECT_EQ(row.offset(), 0);
}

TEST_F(SliceTest, LastRowSlice) {
    auto row = t3x3.slice(0, 2); // [7, 8, 9]
    EXPECT_FLOAT_EQ(row.at({0}), 7.0f);
    EXPECT_EQ(row.offset(), 6);
}

TEST_F(SliceTest, RowSliceOutOfBoundsThrows) {
    EXPECT_THROW(t3x3.slice(0, 3), std::out_of_range);
}

// ── Col slice ─────────────────────────────────────────────────────────────────

TEST_F(SliceTest, ColSliceShape) {
    auto col = t3x3.slice(1, 1); // second column
    EXPECT_EQ(col.shape(), (std::vector<size_t>{3}));
}

TEST_F(SliceTest, ColSliceValues) {
    auto col = t3x3.slice(1, 1); // [2, 5, 8]
    EXPECT_FLOAT_EQ(col.at({0}), 2.0f);
    EXPECT_FLOAT_EQ(col.at({1}), 5.0f);
    EXPECT_FLOAT_EQ(col.at({2}), 8.0f);
}

TEST_F(SliceTest, ColSliceStride) {
    auto col = t3x3.slice(1, 0); // first column, stride should be 3
    EXPECT_EQ(col.strides()[0], 3);
}

TEST_F(SliceTest, ColSliceIsView) {
    auto col = t3x3.slice(1, 2); // third column
    col.at({0}) = 99.0f;
    EXPECT_FLOAT_EQ(t3x3.at({0, 2}), 99.0f); // same storage
}

TEST_F(SliceTest, ColSliceOffset) {
    auto col = t3x3.slice(1, 2); // third column starts at index 2
    EXPECT_EQ(col.offset(), 2);
}

// ── Chained slices ────────────────────────────────────────────────────────────

TEST_F(SliceTest, ChainedSliceRowThenCol) {
    // t[1][2] == 6
    auto row = t3x3.slice(0, 1); // second row [4,5,6]
    EXPECT_FLOAT_EQ(row.at({2}), 6.0f);
}

TEST_F(SliceTest, ChainedSliceSharesStorage) {
    auto row = t3x3.slice(0, 1);
    row.at({2}) = 99.0f;
    EXPECT_FLOAT_EQ(t3x3.at({1, 2}), 99.0f);
}

// ── Slice then contiguous ─────────────────────────────────────────────────────

TEST_F(SliceTest, ColSliceIsNotContiguous) {
    auto col = t3x3.slice(1, 0);
    EXPECT_FALSE(col.is_contiguous());
}

TEST_F(SliceTest, ColSliceContiguousCopiesCorrectly) {
    auto col = t3x3.slice(1, 1); // [2, 5, 8]
    auto cont = col.contiguous();
    EXPECT_TRUE(cont.is_contiguous());
    EXPECT_FLOAT_EQ(cont.at({0}), 2.0f);
    EXPECT_FLOAT_EQ(cont.at({1}), 5.0f);
    EXPECT_FLOAT_EQ(cont.at({2}), 8.0f);
}

TEST_F(SliceTest, ContiguousDoesNotShareStorage) {
    auto col = t3x3.slice(1, 1);
    auto cont = col.contiguous();
    cont.at({0}) = 99.0f;
    EXPECT_FLOAT_EQ(t3x3.at({0, 1}), 2.0f); // original unchanged
}

// ── Shape inference ───────────────────────────────────────────────────────────

TEST_F(BroadcastTest, SameShapeNoOp) {
    auto a = Tensor::ones({2, 3});
    auto b = Tensor::ones({2, 3});
    auto shape = shape_utils::broadcast_shape(a.shape(), b.shape());
    EXPECT_EQ(shape, (std::vector<size_t>{2, 3}));
}

TEST_F(BroadcastTest, ScalarBroadcastShape) {
    auto shape = shape_utils::broadcast_shape({1}, {2, 3});
    EXPECT_EQ(shape, (std::vector<size_t>{2, 3}));
}

TEST_F(BroadcastTest, RowVectorBroadcastShape) {
    // [1,3] + [2,3] -> [2,3]
    auto shape = shape_utils::broadcast_shape({1, 3}, {2, 3});
    EXPECT_EQ(shape, (std::vector<size_t>{2, 3}));
}

TEST_F(BroadcastTest, ColVectorBroadcastShape) {
    // [2,1] + [2,3] -> [2,3]
    auto shape = shape_utils::broadcast_shape({2, 1}, {2, 3});
    EXPECT_EQ(shape, (std::vector<size_t>{2, 3}));
}

TEST_F(BroadcastTest, BothDimsBroadcastShape) {
    // [2,1] + [1,3] -> [2,3]
    auto shape = shape_utils::broadcast_shape({2, 1}, {1, 3});
    EXPECT_EQ(shape, (std::vector<size_t>{2, 3}));
}

TEST_F(BroadcastTest, IncompatibleShapesThrows) {
    EXPECT_THROW(shape_utils::broadcast_shape({2, 3}, {2, 4}), std::runtime_error);
}

TEST_F(BroadcastTest, DifferentRanksBroadcastShape) {
    // [3] + [2,3] -> [2,3]
    auto shape = shape_utils::broadcast_shape({3}, {2, 3});
    EXPECT_EQ(shape, (std::vector<size_t>{2, 3}));
}

// ── Broadcast add values ──────────────────────────────────────────────────────

TEST_F(BroadcastTest, AddRowVector) {
    // [[1,2,3],   +  [1,2,3]  =  [[2,4,6],
    //  [4,5,6]]                    [5,7,9]]
    auto out = ops::add(mat2x3, mat1x3);
    EXPECT_EQ(out.shape(), (std::vector<size_t>{2, 3}));
    EXPECT_FLOAT_EQ(out.at({0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(out.at({0, 1}), 4.0f);
    EXPECT_FLOAT_EQ(out.at({0, 2}), 6.0f);
    EXPECT_FLOAT_EQ(out.at({1, 0}), 5.0f);
    EXPECT_FLOAT_EQ(out.at({1, 1}), 7.0f);
    EXPECT_FLOAT_EQ(out.at({1, 2}), 9.0f);
}

TEST_F(BroadcastTest, AddColVector) {
    // [[1,2,3],   +  [[10],  =  [[11,12,13],
    //  [4,5,6]]       [20]]      [24,25,26]]
    auto out = ops::add(mat2x3, mat2x1);
    EXPECT_EQ(out.shape(), (std::vector<size_t>{2, 3}));
    EXPECT_FLOAT_EQ(out.at({0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(out.at({0, 2}), 13.0f);
    EXPECT_FLOAT_EQ(out.at({1, 0}), 24.0f);
    EXPECT_FLOAT_EQ(out.at({1, 2}), 26.0f);
}

TEST_F(BroadcastTest, AddBothBroadcast) {
    // [2,1] + [1,3] -> [2,3]
    // [[1],   +  [1,2,3]  =  [[2,3,4],
    //  [2]]                   [3,4,5]]
    auto a = Tensor::from_data({1, 2}, {2, 1});
    auto b = Tensor::from_data({1, 2, 3}, {1, 3});
    auto out = ops::add(a, b);
    EXPECT_EQ(out.shape(), (std::vector<size_t>{2, 3}));
    EXPECT_FLOAT_EQ(out.at({0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(out.at({0, 2}), 4.0f);
    EXPECT_FLOAT_EQ(out.at({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(out.at({1, 2}), 5.0f);
}

TEST_F(BroadcastTest, AddDoesNotMutateInputs) {
    auto a_copy = Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    ops::add(mat2x3, mat1x3);
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            EXPECT_FLOAT_EQ(mat2x3.at({i, j}), a_copy.at({i, j}));
        }
    }
}

// ── Broadcast mul values ──────────────────────────────────────────────────────

TEST_F(BroadcastTest, MulRowVector) {
    // [[1,2,3],   *  [1,2,3]  =  [[1,4,9],
    //  [4,5,6]]                   [4,10,18]]
    auto out = ops::mul(mat2x3, mat1x3);
    EXPECT_FLOAT_EQ(out.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(out.at({0, 1}), 4.0f);
    EXPECT_FLOAT_EQ(out.at({0, 2}), 9.0f);
    EXPECT_FLOAT_EQ(out.at({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(out.at({1, 1}), 10.0f);
    EXPECT_FLOAT_EQ(out.at({1, 2}), 18.0f);
}

// ── Broadcast is a view (stride=0 trick) ─────────────────────────────────────

TEST_F(BroadcastTest, BroadcastedTensorHasZeroStride) {
    // when you expand [1,3] to [2,3], the new dim should have stride 0
    auto expanded = mat1x3.expand({2, 3});
    EXPECT_EQ(expanded.strides()[0], 0);
    EXPECT_EQ(expanded.strides()[1], 1);
    EXPECT_EQ(expanded.shape(), (std::vector<size_t>{2, 3}));
}

TEST_F(BroadcastTest, ExpandIsView) {
    auto expanded = mat1x3.expand({2, 3});
    // same storage — modifying expanded modifies original
    expanded.at({0, 0}) = 99.0f;
    EXPECT_FLOAT_EQ(mat1x3.at({0, 0}), 99.0f);
}

TEST_F(BroadcastTest, ExpandIncompatibleThrows) {
    EXPECT_THROW(mat1x3.expand({2, 4}), std::runtime_error);
}
