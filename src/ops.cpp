#include "matt/ops.hpp"
#include "matt/shape_utils.hpp"
#include "matt/tensor.hpp"
#include <cmath>
#include <stdexcept>

namespace matt {
namespace ops {

Tensor elementwise(const Tensor &a, const Tensor &b, std::function<float(float, float)> op) {
    auto out_shape = shape_utils::broadcast_shape(a.shape(), b.shape());
    auto a_broadcasted = a.broadcast_to(out_shape);
    auto b_broadcasted = b.broadcast_to(out_shape);

    auto out = Tensor::zeros(out_shape);
    size_t n = out.numel();

    std::vector<size_t> idx(out_shape.size(), 0);
    for (size_t i = 0; i < n; i++) {
        out.at(idx) = op(a_broadcasted.at(idx), b_broadcasted.at(idx));
        for (int d = (int)out_shape.size() - 1; d >= 0; d--) {
            if (++idx[d] < out_shape[d])
                break;
            idx[d] = 0;
        }
    }
    return out;
}

Tensor elementwise_unary(const Tensor &a, std::function<float(float)> op) {
    auto out_shape = a.shape();
    auto out = Tensor::zeros(out_shape);
    size_t n = out.numel();

    std::vector<size_t> idx(out_shape.size(), 0);
    for (size_t i = 0; i < n; i++) {
        out.at(idx) = op(a.at(idx));
        for (int d = (int)out_shape.size() - 1; d >= 0; d--) {
            if (++idx[d] < out_shape[d])
                break;
            idx[d] = 0;
        }
    }
    return out;
}

Tensor accumulate(const Tensor &a, std::function<float(float, float)> op) {
    auto a_shape = a.shape();
    auto out = Tensor::zeros({1});
    size_t n = a.numel();

    std::vector<size_t> idx(a_shape.size(), 0);
    for (size_t i = 0; i < n; i++) {
        out.at({0}) = op(a.at(idx), out.at({0}));
        for (int d = (int)a_shape.size() - 1; d >= 0; d--) {
            if (++idx[d] < a_shape[d])
                break;
            idx[d] = 0;
        }
    }
    return out;
}

template <typename Op> Tensor apply_binary(const Tensor &a, const Tensor &b) {
    if (a.device() != b.device())
        throw std::runtime_error("ops::apply_binary: tensors must be on the same device.");
    Tensor out = Op::forward(a, b);
    if (a.requires_grad() || b.requires_grad()) {
        out.data()->grad_fn = Op::make_grad_fn(a, b);
        out.set_requires_grad(true);
    }
    return out;
}

template <typename Op> Tensor apply_unary(const Tensor &a) {
    Tensor out = Op::forward(a);
    if (a.requires_grad()) {
        out.data()->grad_fn = Op::make_grad_fn(a);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor add(const Tensor &a, const Tensor &b) {
    return apply_binary<AddOp>(a, b);
}

Tensor sub(const Tensor &a, const Tensor &b) {
    return apply_binary<SubOp>(a, b);
}

Tensor mul(const Tensor &a, const Tensor &b) {
    return apply_binary<MulOp>(a, b);
}

Tensor matmul(const Tensor &a, const Tensor &b) {
    return apply_binary<MatmulOp>(a, b);
}

Tensor relu(const Tensor &a) {
    return apply_unary<ReluOp>(a);
}

Tensor sum(const Tensor &a) {
    return apply_unary<SumOp>(a);
}

Tensor AddOp::forward(const Tensor &a, const Tensor &b) {
    auto out_shape = shape_utils::broadcast_shape(a.shape(), b.shape());
    auto a_broadcasted = a.broadcast_to(out_shape).contiguous();
    auto b_broadcasted = b.broadcast_to(out_shape).contiguous();
    auto out = Tensor::zeros(out_shape, a.device());
    get_backend(a.device())
        ->elementwise_binary(a_broadcasted.data_ptr(), b_broadcasted.data_ptr(), out.data_ptr(),
                             out.numel(), BinaryOpType::Add);
    return out;
}

Tensor SubOp::forward(const Tensor &a, const Tensor &b) {
    auto out_shape = shape_utils::broadcast_shape(a.shape(), b.shape());
    auto a_broadcasted = a.broadcast_to(out_shape).contiguous();
    auto b_broadcasted = b.broadcast_to(out_shape).contiguous();
    auto out = Tensor::zeros(out_shape, a.device());
    get_backend(a.device())
        ->elementwise_binary(a_broadcasted.data_ptr(), b_broadcasted.data_ptr(), out.data_ptr(),
                             out.numel(), BinaryOpType::Sub);
    return out;
}

Tensor MulOp::forward(const Tensor &a, const Tensor &b) {
    auto out_shape = shape_utils::broadcast_shape(a.shape(), b.shape());
    auto a_broadcasted = a.broadcast_to(out_shape).contiguous();
    auto b_broadcasted = b.broadcast_to(out_shape).contiguous();
    auto out = Tensor::zeros(out_shape, a.device());
    get_backend(a.device())
        ->elementwise_binary(a_broadcasted.data_ptr(), b_broadcasted.data_ptr(), out.data_ptr(),
                             out.numel(), BinaryOpType::Mul);
    return out;
}

Tensor MatmulOp::forward(const Tensor &a, const Tensor &b) {
    if (a.ndim() != 2 || b.ndim() != 2)
        throw std::runtime_error("matmul: only 2D matrices allowed");
    if (a.shape()[1] != b.shape()[0])
        throw std::runtime_error("matmul: input matrices do not share common dimension");

    size_t M = a.shape()[0];
    size_t N = b.shape()[1];
    size_t K = a.shape()[1];

    auto a_contiguous = a.contiguous();
    auto b_contiguous = b.contiguous();
    auto out = Tensor::zeros({M, N}, a.device());
    get_backend(a.device())
        ->matmul(a_contiguous.data_ptr(), b_contiguous.data_ptr(), out.data_ptr(), M, K, N);
    return out;
}

Tensor ReluOp::forward(const Tensor &a) {
    auto a_contiguous = a.contiguous();
    auto out = Tensor::zeros(a.shape(), a.device());
    get_backend(a.device())
        ->elementwise_unary(a_contiguous.data_ptr(), out.data_ptr(), out.numel(),
                            UnaryOpType::Relu);
    return out;
}

Tensor SumOp::forward(const Tensor &a) {
    auto a_contiguous = a.contiguous();
    auto out = Tensor::zeros({1}, a.device());
    get_backend(a.device())
        ->reduce(a_contiguous.data_ptr(), out.data_ptr(), a_contiguous.numel(), ReduceOpType::Sum);
    return out;
}

} // namespace ops
} // namespace matt