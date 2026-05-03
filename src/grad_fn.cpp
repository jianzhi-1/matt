#include "matt/grad_fn.hpp"
#include "matt/ops.hpp"
#include "matt/tensor.hpp"

namespace matt {

BinaryGradFn::BinaryGradFn(Tensor a, Tensor b)
    : a_(std::make_shared<Tensor>(std::move(a))), b_(std::make_shared<Tensor>(std::move(b))) {
    inputs = {a_, b_};
}

UnaryGradFn::UnaryGradFn(Tensor a) : a_(std::make_shared<Tensor>(std::move(a))) {
    inputs = {a_};
}

std::vector<Tensor> AddBackward::backward(const Tensor &grad_out) const {
    return {grad_out, grad_out};
}

std::vector<Tensor> MulBackward::backward(const Tensor &grad_out) const {
    return {ops::MulOp::forward(grad_out, *b_), ops::MulOp::forward(grad_out, *a_)};
}

std::vector<Tensor> MatmulBackward::backward(const Tensor &grad_out) const {
    return {ops::MatmulOp::forward(grad_out, b_->transpose(0, 1)),
            ops::MatmulOp::forward(a_->transpose(0, 1), grad_out)};
}

} // namespace matt