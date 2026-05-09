#include "matt/nn/linear.hpp"
#include "matt/ops.hpp"
namespace matt {
namespace nn {
Linear::Linear(size_t input_dim, size_t output_dim, bool use_bias)
    : input_dim_(input_dim), output_dim_(output_dim), use_bias_(use_bias) {
    weight_ = Tensor::zeros({output_dim, input_dim});
    register_parameter("weight", weight_);

    if (use_bias) {
        bias_ = Tensor::zeros({output_dim});
        register_parameter("bias", bias_);
    }
}

Tensor Linear::forward(const Tensor &x) {
    auto out = ops::matmul(x, weight_.transpose(0, 1));
    if (use_bias_)
        out = ops::add(out, bias_);
    return out;
}

} // namespace nn
} // namespace matt