#pragma once
#include "matt/tensor.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <functional>
#include <iostream>
namespace matt {
namespace nn {

class Module {
public:
    virtual ~Module() = default;

    virtual Tensor forward(const Tensor& x) = 0;

    std::vector<Tensor> parameters() const;
    std::vector<std::pair<std::string, Tensor>> named_parameters() const;

    void zero_grad();
    // void to(Device device); // TODO: GPU support
    // TODO: add buffers.

    void train(bool mode);
    bool is_training() const { return training_; }

    virtual void save(std::ostream& os) const;
    virtual void load(std::istream& is);
    virtual std::string name() const { return "Module"; };

protected:

    void register_parameter(const std::string& name, const Tensor& t);
    
    template<typename T>
    std::shared_ptr<T> register_module(const std::string& name, std::shared_ptr<T> m);

private:
    std::vector<std::string> parameter_names_;
    std::unordered_map<std::string, Tensor> parameters_;

    std::vector<std::string> submodule_names_;
    std::unordered_map<std::string, std::shared_ptr<Module>> submodules_;

    bool training_ = true;
};

template<typename T>
std::shared_ptr<T> Module::register_module(
    const std::string& name,
    std::shared_ptr<T> m
){
    if (submodules_.count(name)) throw std::runtime_error("nn:register_module: already registered submodule with name");
    submodule_names_.push_back(name);
    return submodules_[name] = m;
}

}
}

