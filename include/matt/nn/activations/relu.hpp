#pragma once
#include "matt/nn/module.hpp"
#include "matt/tensor.hpp"
namespace matt {
namespace nn {
class ReLU : public Module {
public:
    Tensor forward(const Tensor& x) override;
    std::string name() const override { return "ReLU"; }
};
}
}
