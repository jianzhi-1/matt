#include "matt/grad_fn.hpp"
#include "matt/ops.hpp"
#include "matt/tensor.hpp"
#include "test_utils.hpp"
#include <gtest/gtest.h>

using namespace matt;
using namespace matt::test_utils;

// ── Graph construction ────────────────────────────────────────────────────────

TEST(AutogradTest, LeafTensorHasNoGradFn) {
    auto x = Tensor::ones({2, 2});
    x.set_requires_grad(true);
    EXPECT_EQ(x.data()->grad_fn, nullptr);
}

TEST(AutogradTest, OpOutputHasGradFn) {
    auto x = Tensor::ones({2, 2});
    auto y = Tensor::ones({2, 2});
    x.set_requires_grad(true);
    auto z = ops::add(x, y);
    EXPECT_NE(z.data()->grad_fn, nullptr);
    EXPECT_STREQ(z.data()->grad_fn->name(), "AddBackward");
}

TEST(AutogradTest, OpOutputRequiresGrad) {
    auto x = Tensor::ones({2, 2});
    x.set_requires_grad(true);
    auto y = ops::relu(x);
    EXPECT_TRUE(y.requires_grad());
}

TEST(AutogradTest, NoGradFnWhenInputsDoNotRequireGrad) {
    auto x = Tensor::ones({2, 2});
    auto y = Tensor::ones({2, 2});
    auto z = ops::add(x, y);
    EXPECT_EQ(z.data()->grad_fn, nullptr);
    EXPECT_FALSE(z.requires_grad());
}

TEST(AutogradTest, GradFnInputsAreCorrect) {
    auto x = Tensor::ones({2, 2});
    x.set_requires_grad(true);
    auto y = Tensor::ones({2, 2});
    auto z = ops::add(x, y);
    EXPECT_EQ(z.data()->grad_fn->inputs.size(), 2);
}

// ── Sum backward ──────────────────────────────────────────────────────────────

TEST(AutogradTest, SumBackwardOnes) {
    auto x = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    x.set_requires_grad(true);
    auto y = ops::sum(x);
    y.backward();
    ASSERT_NE(x.data()->grad, nullptr);
    expect_tensors_close(Tensor(x.data()->grad), Tensor::ones({2, 2}));
}

TEST(AutogradTest, SumBackwardShape) {
    auto x = Tensor::zeros({3, 4});
    x.set_requires_grad(true);
    auto y = ops::sum(x);
    y.backward();
    ASSERT_NE(x.data()->grad, nullptr);
    EXPECT_EQ(Tensor(x.data()->grad).shape(), (std::vector<size_t>{3, 4}));
}

// ── Add backward ──────────────────────────────────────────────────────────────

TEST(AutogradTest, AddBackwardBothInputs) {
    auto a = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    auto b = Tensor::from_data({5, 6, 7, 8}, {2, 2});
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    auto z = ops::sum(ops::add(a, b));
    z.backward();
    ASSERT_NE(a.data()->grad, nullptr);
    ASSERT_NE(b.data()->grad, nullptr);
    expect_tensors_close(Tensor(a.data()->grad), Tensor::ones({2, 2}));
    expect_tensors_close(Tensor(b.data()->grad), Tensor::ones({2, 2}));
}

TEST(AutogradTest, AddBackwardOnlyOneRequiresGrad) {
    auto a = Tensor::ones({2, 2});
    auto b = Tensor::ones({2, 2});
    a.set_requires_grad(true);
    auto z = ops::sum(ops::add(a, b));
    z.backward();
    ASSERT_NE(a.data()->grad, nullptr);
    EXPECT_EQ(b.data()->grad, nullptr);
}

// ── Mul backward ──────────────────────────────────────────────────────────────

TEST(AutogradTest, MulBackward) {
    auto a = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    auto b = Tensor::from_data({5, 6, 7, 8}, {2, 2});
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    auto z = ops::sum(ops::mul(a, b));
    z.backward();
    ASSERT_NE(a.data()->grad, nullptr);
    ASSERT_NE(b.data()->grad, nullptr);
    expect_tensors_close(Tensor(a.data()->grad), b);
    expect_tensors_close(Tensor(b.data()->grad), a);
}

// ── Matmul backward ───────────────────────────────────────────────────────────

TEST(AutogradTest, MatmulBackwardShape) {
    auto a = Tensor::ones({2, 3});
    auto b = Tensor::ones({3, 4});
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    auto out = ops::sum(ops::matmul(a, b));
    out.backward();
    ASSERT_NE(a.data()->grad, nullptr);
    ASSERT_NE(b.data()->grad, nullptr);
    EXPECT_EQ(Tensor(a.data()->grad).shape(), (std::vector<size_t>{2, 3}));
    EXPECT_EQ(Tensor(b.data()->grad).shape(), (std::vector<size_t>{3, 4}));
}

TEST(AutogradTest, MatmulBackwardValues) {
    auto a = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    auto b = Tensor::from_data({1, 0, 0, 1}, {2, 2});
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    auto out = ops::sum(ops::matmul(a, b));
    out.backward();
    ASSERT_NE(a.data()->grad, nullptr);
    expect_tensors_close(Tensor(a.data()->grad), Tensor::ones({2, 2}));
}

// ── Relu backward ─────────────────────────────────────────────────────────────

TEST(AutogradTest, ReluBackwardPositive) {
    auto x = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    x.set_requires_grad(true);
    auto y = ops::sum(ops::relu(x));
    y.backward();
    ASSERT_NE(x.data()->grad, nullptr);
    expect_tensors_close(Tensor(x.data()->grad), Tensor::ones({2, 2}));
}

TEST(AutogradTest, ReluBackwardNegative) {
    auto x = Tensor::from_data({-1, -2, -3, -4}, {2, 2});
    x.set_requires_grad(true);
    auto y = ops::sum(ops::relu(x));
    y.backward();
    ASSERT_NE(x.data()->grad, nullptr);
    expect_tensors_close(Tensor(x.data()->grad), Tensor::zeros({2, 2}));
}

TEST(AutogradTest, ReluBackwardMixed) {
    auto x = Tensor::from_data({-1, 2, -3, 4}, {2, 2});
    x.set_requires_grad(true);
    auto y = ops::sum(ops::relu(x));
    y.backward();
    ASSERT_NE(x.data()->grad, nullptr);
    expect_tensors_close(Tensor(x.data()->grad), Tensor::from_data({0, 1, 0, 1}, {2, 2}));
}

// ── Chained ops ───────────────────────────────────────────────────────────────

TEST(AutogradTest, ChainedAddMul) {
    auto a = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    auto b = Tensor::from_data({1, 1, 1, 1}, {2, 2});
    auto c = Tensor::from_data({2, 2, 2, 2}, {2, 2});
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    c.set_requires_grad(true);
    auto z = ops::sum(ops::mul(ops::add(a, b), c));
    z.backward();
    ASSERT_NE(a.data()->grad, nullptr);
    ASSERT_NE(b.data()->grad, nullptr);
    ASSERT_NE(c.data()->grad, nullptr);
    expect_tensors_close(Tensor(a.data()->grad), c);
    expect_tensors_close(Tensor(b.data()->grad), c);
    expect_tensors_close(Tensor(c.data()->grad), ops::add(a, b));
}

TEST(AutogradTest, ChainedReluAdd) {
    auto a = Tensor::from_data({-1, 2, -3, 4}, {2, 2});
    auto b = Tensor::ones({2, 2});
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    auto z = ops::sum(ops::add(ops::relu(a), b));
    z.backward();
    expect_tensors_close(Tensor(a.data()->grad), Tensor::from_data({0, 1, 0, 1}, {2, 2}));
    expect_tensors_close(Tensor(b.data()->grad), Tensor::ones({2, 2}));
}

// ── Gradient accumulation ─────────────────────────────────────────────────────

TEST(AutogradTest, GradAccumulatesWhenUsedTwice) {
    auto x = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    x.set_requires_grad(true);
    auto z = ops::sum(ops::add(x, x));
    z.backward();
    ASSERT_NE(x.data()->grad, nullptr);
    expect_tensors_close(Tensor(x.data()->grad), Tensor::from_data({2, 2, 2, 2}, {2, 2}));
}