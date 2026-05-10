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
        // TODO: divide by numel
        return ops::sum(ops::mul(diff, diff));
    }
};

}
}