#include "matt/nn/linear.hpp"
#include "matt/ops.hpp"
#include "matt/tensor.hpp"
#include <gtest/gtest.h>

using namespace matt;
using namespace matt::nn;

class LinearTest : public ::testing::Test {
  protected:
    Linear linear{4, 8};
    Tensor x = Tensor::ones({2, 4});
};

// ── Construction ──────────────────────────────────────────────────────────────

TEST_F(LinearTest, ParameterCount) {
    // weight + bias = 2 parameters
    EXPECT_EQ(linear.parameters().size(), 2);
}

TEST_F(LinearTest, NoBiasParameterCount) {
    Linear no_bias{4, 8, false};
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

// ── Forward ───────────────────────────────────────────────────────────────────

TEST_F(LinearTest, ForwardOutputShape) {
    Tensor out = linear.forward(x);
    EXPECT_EQ(out.shape(), (std::vector<size_t>{2, 8}));
}

TEST_F(LinearTest, ForwardZeroWeightZeroBias) {
    // weight and bias are zeros (default init), so output should be all zeros
    Tensor out = linear.forward(x);
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 8; j++)
            EXPECT_FLOAT_EQ(out.at({i, j}), 0.0f);
}

TEST_F(LinearTest, ForwardNoBiasOutputShape) {
    Linear no_bias{4, 8, false};
    Tensor out = no_bias.forward(x);
    EXPECT_EQ(out.shape(), (std::vector<size_t>{2, 8}));
}