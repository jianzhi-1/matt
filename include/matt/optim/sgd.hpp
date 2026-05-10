#pragma once
#include "matt/tensor.hpp"
#include "matt/optim/optimizer.hpp"
#include <vector>
namespace matt {
namespace optim {
class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor> params, float lr): Optimizer(std::move(params), lr){}
    void step() override {
        for (auto& p: params_){
            if (!p.data()->grad) continue;
            float* p_data = p.data_ptr();
            Tensor grad(p.data()->grad);
            const float* g_data = grad.data_ptr();
            for (size_t i = 0; i < p.numel(); i++){
                p_data[i] -= lr_ * g_data[i];
            }
        }
    }
};
}
}