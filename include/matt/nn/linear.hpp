#pragma once
#include "matt/nn/module.hpp"
#include "matt/nn/weight_initializer/weight_initializer.hpp"
#include "matt/tensor.hpp"
namespace matt {
namespace nn {

class Linear: public Module {
public:

    Linear(size_t input_dim, size_t output_dim, const weight_initializer::WeightInitializer& weight_initializer, bool use_bias=true);
    Tensor forward(const Tensor& x) override;
    std::string name() const override { return "Linear"; }

private:
    size_t input_dim_, output_dim_;
    bool use_bias_;

    Tensor weight_, bias_;
};
}
}