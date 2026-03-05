#pragma once
#include "tensor.hpp"
namespace matt {
    namespace ops {
        Tensor add(const Tensor& a, const Tensor& b);
        Tensor mul(const Tensor& a, const Tensor& b);
        Tensor matmul(const Tensor& a, const Tensor& b);
        Tensor relu(const Tensor& a);
        Tensor sum(const Tensor& a);
        Tensor softmax(const Tensor& a, int dim);
    }
}