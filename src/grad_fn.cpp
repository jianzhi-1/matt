#include "matt/grad_fn.hpp"
#include "matt/ops.hpp"
#include "matt/tensor.hpp"

namespace matt {

BinaryGradFn::BinaryGradFn(Tensor a, Tensor b) : a_(a.data()), b_(b.data()) {
    inputs = {a_, b_};
}

UnaryGradFn::UnaryGradFn(Tensor a) : a_(a.data()) {
    inputs = {a_};
}

std::vector<Tensor> AddBackward::backward(const Tensor &grad_out) const {
    return {grad_out, grad_out};
}

std::vector<Tensor> MulBackward::backward(const Tensor &grad_out) const {
    return {ops::MulOp::forward(grad_out, Tensor(b_)), ops::MulOp::forward(grad_out, Tensor(a_))};
}

std::vector<Tensor> MatmulBackward::backward(const Tensor &grad_out) const {
    return {ops::MatmulOp::forward(grad_out, Tensor(b_).transpose(0, 1)),
            ops::MatmulOp::forward(Tensor(a_).transpose(0, 1), grad_out)};
}

std::vector<Tensor> ReluBackward::backward(const Tensor &grad_out) const {
    return {ops::elementwise(Tensor(a_), grad_out,
                             [](float a, float g) { return (a > 0.f) ? g : 0.f; })};
}

std::vector<Tensor> SumBackward::backward(const Tensor &grad_out) const {
    if (grad_out.numel() != 1)
        throw std::runtime_error("SumBackward: gradient of unexpeced shape");
    float scale = grad_out.at({0});
    return {Tensor::fill(a_.get()->shape(), scale)};
}

} // namespace matt