#pragma once
#include "matt/nn/module.hpp"
#include "matt/ops.hpp"
#include "matt/tensor.hpp"

namespace matt {
namespace nn {

class MSELoss {
public:
    Tensor forward(const Tensor& pred, const Tensor& target) const {
        Tensor diff = ops::sub(pred, target);
        Tensor n = Tensor::from_data({1.0f / static_cast<float>(pred.numel())}, {1});
        return ops::mul(ops::sum(ops::mul(diff, diff)), n);
    }
};
}
}