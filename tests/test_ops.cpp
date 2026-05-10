#include "matt/ops.hpp"
#include "matt/tensor.hpp"
#include <gtest/gtest.h>

using namespace matt;

class OpsTest : public ::testing::Test {
  protected:
    Tensor a2x2 = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    Tensor b2x2 = Tensor::from_data({5, 6, 7, 8}, {2, 2});
    Tensor ones = Tensor::ones({2, 2});
};

// ── Add ───────────────────────────────────────────────────────────────────────
TEST_F(OpsTest, AddCorrectValues) {
    auto c = ops::add(a2x2, b2x2);
    EXPECT_FLOAT_EQ(c.at({0, 0}), 6.0f);
    EXPECT_FLOAT_EQ(c.at({0, 1}), 8.0f);
    EXPECT_FLOAT_EQ(c.at({1, 0}), 10.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 12.0f);
}

TEST_F(OpsTest, AddShapeMismatchThrows) {
    auto c = Tensor::zeros({3, 2});
    EXPECT_THROW(ops::add(a2x2, c), std::runtime_error);
}

TEST_F(OpsTest, AddDoesNotMutateInputs) {
    auto a_copy = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    ops::add(a2x2, b2x2);
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            EXPECT_FLOAT_EQ(a2x2.at({i, j}), a_copy.at({i, j}));
        }
    }
}

// ── Mul ───────────────────────────────────────────────────────────────────────
TEST_F(OpsTest, MulElementWise) {
    auto c = ops::mul(a2x2, b2x2);
    EXPECT_FLOAT_EQ(c.at({0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(c.at({0, 1}), 12.0f);
    EXPECT_FLOAT_EQ(c.at({1, 0}), 21.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 32.0f);
}

// ── Matmul ────────────────────────────────────────────────────────────────────
TEST_F(OpsTest, MatmulCorrectValues) {
    auto c = ops::matmul(a2x2, b2x2);
    EXPECT_FLOAT_EQ(c.at({0, 0}), 19.0f);
    EXPECT_FLOAT_EQ(c.at({0, 1}), 22.0f);
    EXPECT_FLOAT_EQ(c.at({1, 0}), 43.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 50.0f);
}

TEST_F(OpsTest, MatmulOutputShape) {
    auto a = Tensor::ones({3, 4});
    auto b = Tensor::ones({4, 5});
    auto c = ops::matmul(a, b);
    EXPECT_EQ(c.shape(), (std::vector<size_t>{3, 5}));
}

TEST_F(OpsTest, MatmulInnerDimMismatchThrows) {
    auto a = Tensor::ones({2, 3});
    auto b = Tensor::ones({4, 2});
    EXPECT_THROW(ops::matmul(a, b), std::runtime_error);
}

TEST_F(OpsTest, MatmulIdentity) {
    auto identity = Tensor::from_data({1, 0, 0, 1}, {2, 2});
    auto c = ops::matmul(a2x2, identity);
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            EXPECT_FLOAT_EQ(c.at({i, j}), a2x2.at({i, j}));
        }
    }
}

// ── ReLU ──────────────────────────────────────────────────────────────────────
TEST_F(OpsTest, ReluZerosOutNegatives) {
    auto t = Tensor::from_data({-2, -1, 0, 1, 2, 3}, {2, 3});
    auto r = ops::relu(t);
    EXPECT_FLOAT_EQ(r.at({0, 0}), 0.0f); // was -2
    EXPECT_FLOAT_EQ(r.at({0, 1}), 0.0f); // was -1
    EXPECT_FLOAT_EQ(r.at({0, 2}), 0.0f); // was  0
    EXPECT_FLOAT_EQ(r.at({1, 0}), 1.0f);
    EXPECT_FLOAT_EQ(r.at({1, 1}), 2.0f);
    EXPECT_FLOAT_EQ(r.at({1, 2}), 3.0f);
}

// ── Sum ───────────────────────────────────────────────────────────────────────
TEST_F(OpsTest, SumAllElements) {
    auto t = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    auto s = ops::sum(t);
    EXPECT_EQ(s.numel(), 1);
    EXPECT_FLOAT_EQ(s.data_ptr()[0], 10.0f);
}

TEST_F(OpsTest, SumOfZeros) {
    EXPECT_FLOAT_EQ(ops::sum(Tensor::zeros({5, 5})).data_ptr()[0], 0.0f);
}