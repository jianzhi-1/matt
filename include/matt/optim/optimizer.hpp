#pragma once
#include "matt/tensor.hpp"
#include <vector>
namespace matt {
namespace optim {
class Optimizer {
public:
    Optimizer(std::vector<Tensor> params, float lr): params_(std::move(params)), lr_(lr){}
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    void zero_grad(){
        for (auto& p: params_) p.data()->grad=nullptr;
    }
    float lr() const {return lr_;}
    void set_lr(float lr){lr_ = lr;}

protected:
    std::vector<Tensor> params_;
    float lr_;
};
}
}