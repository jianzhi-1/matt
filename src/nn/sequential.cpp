#include "matt/nn/sequential.hpp"
namespace matt {
namespace nn {
Tensor Sequential::forward(const Tensor &x) {
    Tensor out = x;
    for (auto &layer : layers_)
        out = layer(out);
    return out;
}
} // namespace nn
} // namespace matt