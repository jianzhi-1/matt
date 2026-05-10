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
    return ops::elementwise(a, b, [](float x, float y) { return x + y; });
}

Tensor SubOp::forward(const Tensor &a, const Tensor &b) {
    return ops::elementwise(a, b, [](float x, float y) { return x - y; });
}

Tensor MulOp::forward(const Tensor &a, const Tensor &b) {
    return ops::elementwise(a, b, [](float x, float y) { return x * y; });
}

Tensor MatmulOp::forward(const Tensor &a, const Tensor &b) {
    if (a.ndim() != 2 || b.ndim() != 2)
        throw std::runtime_error("matmul: only 2D matrices allowed");
    if (a.shape()[1] != b.shape()[0])
        throw std::runtime_error("matmul: input matrices do not share common dimension");
    size_t M = a.shape()[0];
    size_t N = b.shape()[1];
    size_t K = a.shape()[1];

    auto out = Tensor::zeros({M, N});
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float acc = 0.;
            for (size_t k = 0; k < K; k++) {
                acc += a.at({i, k}) * b.at({k, j});
            }
            out.at({i, j}) = acc;
        }
    }
    return out;
}

Tensor ReluOp::forward(const Tensor &a) {
    return ops::elementwise_unary(a, [](float x) { return ((x >= 0.) ? x : 0.); });
}

Tensor SumOp::forward(const Tensor &a) {
    return ops::accumulate(a, [](float x, float y) { return x + y; });
}

} // namespace ops
} // namespace matt