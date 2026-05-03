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

} // namespace matt