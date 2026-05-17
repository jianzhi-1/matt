#include "matt/nn/linear.hpp"
#include "matt/nn/weight_initializer/weight_initializer.hpp"
#include "matt/ops.hpp"
namespace matt {
namespace nn {
Linear::Linear(size_t input_dim, size_t output_dim,
               const weight_initializer::WeightInitializer &weight_initializer, bool use_bias)
    : input_dim_(input_dim), output_dim_(output_dim), use_bias_(use_bias) {
    weight_ = weight_initializer.initialize({output_dim, input_dim});
    weight_.set_requires_grad(true);
    register_parameter("weight", weight_);

    if (use_bias) {
        // TODO: weight_initializer can't be Kaiming in this case. Fix.
        bias_ = weight_initializer.initialize({output_dim});
        bias_.set_requires_grad(true);
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