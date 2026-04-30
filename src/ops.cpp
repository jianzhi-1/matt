#include "matt/ops.hpp"
#include <cmath>
#include <stdexcept>

namespace matt {
namespace ops {

Tensor add(const Tensor &a, const Tensor &b) {
    if (a.shape() != b.shape())
        throw std::runtime_error("add: shape mismatch");
    auto out = Tensor::zeros(a.shape());
    size_t n = a.numel();
    for (size_t i = 0; i < n; i++) {
        out.data_ptr()[i] = a.data_ptr()[i] + b.data_ptr()[i];
    }
    return out;
}

Tensor mul(const Tensor &a, const Tensor &b) {
    if (a.shape() != b.shape())
        throw std::runtime_error("multiply: shape mismatch");
    auto out = Tensor::zeros(a.shape());
    size_t n = a.numel();
    for (size_t i = 0; i < n; i++) {
        out.data_ptr()[i] = a.data_ptr()[i] * b.data_ptr()[i];
    }
    return out;
}

Tensor matmul(const Tensor &a, const Tensor &b) {
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

} // namespace ops
} // namespace matt