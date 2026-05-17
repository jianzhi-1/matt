#include "matt/nn/activations/relu.hpp"
#include "matt/nn/linear.hpp"
#include "matt/nn/loss.hpp"
#include "matt/nn/sequential.hpp"
#include "matt/ops.hpp"
#include "matt/optim/sgd.hpp"
#include "matt/tensor.hpp"
#include <gtest/gtest.h>

using namespace matt;
using namespace matt::nn;
using namespace matt::optim;

class LinearTest : public ::testing::Test {
  protected:
    weight_initializer::Zeros zeros;
    Linear linear{4, 8, zeros};
    Tensor x = Tensor::ones({2, 4});
};

// ── Construction ──────────────────────────────────────────────────────────────

TEST_F(LinearTest, ParameterCount) {
    // weight + bias = 2 parameters
    EXPECT_EQ(linear.parameters().size(), 2);
}

TEST_F(LinearTest, NoBiasParameterCount) {
    Linear no_bias{4, 8, zeros, false};
    EXPECT_EQ(no_bias.parameters().size(), 1);
}

TEST_F(LinearTest, WeightShape) {
    auto params = linear.named_parameters();
    // First registered parameter is weight
    EXPECT_EQ(params[0].first, "weight");
    EXPECT_EQ(params[0].second.shape(), (std::vector<size_t>{8, 4}));
}

TEST_F(LinearTest, BiasShape) {
    auto params = linear.named_parameters();
    EXPECT_EQ(params[1].first, "bias");
    EXPECT_EQ(params[1].second.shape(), (std::vector<size_t>{8}));
}

TEST_F(LinearTest, ParametersRequireGrad) {
    for (auto &p : linear.parameters()) {
        EXPECT_TRUE(p.requires_grad());
    }
}

// ── Forward ───────────────────────────────────────────────────────────────────

TEST_F(LinearTest, ForwardOutputShape) {
    Tensor out = linear.forward(x);
    EXPECT_EQ(out.shape(), (std::vector<size_t>{2, 8}));
}

TEST_F(LinearTest, ForwardZeroWeightZeroBiasOutputIsZero) {
    // Zeros initializer → output must be zero regardless of input
    Tensor out = linear.forward(x);
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 8; j++)
            EXPECT_FLOAT_EQ(out.at({i, j}), 0.0f);
}

TEST_F(LinearTest, ForwardNoBiasOutputShape) {
    Linear no_bias{4, 8, zeros, false};
    Tensor out = no_bias.forward(x);
    EXPECT_EQ(out.shape(), (std::vector<size_t>{2, 8}));
}

// ── Training mode ─────────────────────────────────────────────────────────────

TEST_F(LinearTest, DefaultIsTraining) {
    EXPECT_TRUE(linear.is_training());
}

TEST_F(LinearTest, EvalMode) {
    linear.train(false);
    EXPECT_FALSE(linear.is_training());
}

TEST_F(LinearTest, TrainModeRoundtrip) {
    linear.train(false);
    linear.train(true);
    EXPECT_TRUE(linear.is_training());
}

// ── Autograd wiring ───────────────────────────────────────────────────────────

TEST_F(LinearTest, ForwardOutputHasGradFn) {
    // Output of a linear layer applied to a non-grad input still has a grad_fn
    // because the parameters require grad.
    Tensor out = linear.forward(x);
    EXPECT_NE(out.data()->grad_fn, nullptr);
}

TEST_F(LinearTest, BackwardRuns) {
    Tensor out = linear.forward(x);
    Tensor loss = ops::sum(out);
    // Should not throw
    EXPECT_NO_THROW(loss.backward());
}

TEST_F(LinearTest, WeightGradShapeAfterBackward) {
    Tensor out = linear.forward(x);
    Tensor loss = ops::sum(out);
    loss.backward();
    // weight grad should have same shape as weight
    EXPECT_NE(linear.parameters()[0].data()->grad, nullptr);
    EXPECT_EQ(linear.parameters()[0].data()->grad->shape(), (std::vector<size_t>{8, 4}));
}

TEST_F(LinearTest, ZeroGradClearsGrad) {
    Tensor out = linear.forward(x);
    Tensor loss = ops::sum(out);
    loss.backward();
    linear.zero_grad();
    for (auto &p : linear.parameters()) {
        EXPECT_EQ(p.data()->grad, nullptr);
    }
}

// -- E2E --
TEST(E2ETest, MLPLossDecreasesOverSteps) {
    weight_initializer::Normal init({0, 1, 42});
    auto net =
        std::make_shared<Sequential>(std::make_shared<Linear>(2, 4, init), std::make_shared<ReLU>(),
                                     std::make_shared<Linear>(4, 1, init));

    SGD optimizer(net->parameters(), 0.01f);
    MSELoss criterion;

    Tensor x = Tensor::from_data({1.f, 2.f}, {1, 2});
    Tensor target = Tensor::from_data({0.5f}, {1, 1});

    float first_loss, last_loss;
    for (int i = 0; i < 10; i++) {
        Tensor pred = net->forward(x);
        Tensor loss = criterion.forward(pred, target);
        if (i == 0)
            first_loss = loss.at({0});
        last_loss = loss.at({0});
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    }
    EXPECT_LT(last_loss, first_loss);
}