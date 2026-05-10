#pragma once
#include "matt/nn/module.hpp"
#include <functional>
namespace matt{
namespace nn {
class Sequential: public Module {
public:
    Sequential() = default;
    template<typename... Modules>
    explicit Sequential(Modules... modules){
        (add(std::move(modules)), ...);
    }

    template<typename T>
    Sequential& add(std::shared_ptr<T> m){
        std::string layer_name = std::to_string(layers_.size());
        register_module(layer_name, m);
        layers_.push_back([m](const Tensor& x) -> Tensor {
            return m->forward(x);
        });
        return *this;
    }

    Tensor forward(const Tensor& x) override;
    std::string name() const override { return "Sequential"; }

private:
    std::vector<std::function<Tensor(const Tensor&)>> layers_;
};
}
}