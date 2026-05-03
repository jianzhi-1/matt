#include "matt/grad_fn.hpp"
#include "matt/ops.hpp"
#include "matt/tensor.hpp"
#include "test_utils.hpp"
#include <gtest/gtest.h>

using namespace matt;

// ── Graph construction ────────────────────────────────────────────────────────

TEST(AutogradTest, LeafTensorHasNoGradFn) {
    auto x = Tensor::ones({2, 2});
    x.set_requires_grad(true);
    EXPECT_EQ(x.grad_fn, nullptr);
}

TEST(AutogradTest, OpOutputHasGradFn) {
    auto x = Tensor::ones({2, 2});
    auto y = Tensor::ones({2, 2});
    x.set_requires_grad(true);
    auto z = ops::add(x, y);
    EXPECT_NE(z.grad_fn, nullptr);
    EXPECT_STREQ(z.grad_fn->name(), "AddBackward");
}

TEST(AutogradTest, NoGradFnWhenInputsDoNotRequireGrad) {
    auto x = Tensor::ones({2, 2});
    auto y = Tensor::ones({2, 2});
    auto z = ops::add(x, y);
    EXPECT_EQ(z.grad_fn, nullptr);
    EXPECT_FALSE(z.requires_grad());
}

TEST(AutogradTest, GradFnInputsAreCorrect) {
    auto x = Tensor::ones({2, 2});
    x.set_requires_grad(true);
    auto y = Tensor::ones({2, 2});
    auto z = ops::add(x, y);
    EXPECT_EQ(z.grad_fn->inputs.size(), 2);
}
