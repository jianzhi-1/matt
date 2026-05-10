#include "matt/nn/activations/relu.hpp"
#include "matt/ops.hpp"
namespace matt {
namespace nn {
Tensor ReLU::forward(const Tensor &x) {
    return ops::relu(x);
};
} // namespace nn
} // namespace matt
