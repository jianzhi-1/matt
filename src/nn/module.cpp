#include "matt/nn/module.hpp"
#include <stdexcept>
namespace matt {
namespace nn {

void Module::register_parameter(const std::string &name, const Tensor &t) {
    if (parameters_.count(name))
        throw std::runtime_error("nn:register_parameter: already registered parameter with name");
    parameter_names_.push_back(name);
    parameters_[name] = t;
}

std::vector<std::pair<std::string, Tensor>> Module::named_parameters() const {
    std::vector<std::pair<std::string, Tensor>> named_params;
    for (const auto &pname : parameter_names_) {
        named_params.push_back({pname, parameters_.at(pname)});
    }
    for (const auto &submodule_name : submodule_names_) {
        for (auto &[name, t] : submodules_.at(submodule_name)->named_parameters()) {
            named_params.push_back({submodule_name + "." + name, t});
        }
    }
    return named_params;
}

std::vector<Tensor> Module::parameters() const {
    std::vector<Tensor> params;
    for (auto &[name, t] : named_parameters()) {
        params.push_back(t);
    }
    return params;
}

void Module::zero_grad() {
    for (const auto &pname : parameter_names_) {
        parameters_.at(pname).grad = nullptr;
    }
    for (const auto &submodule_name : submodule_names_) {
        submodules_.at(submodule_name)->zero_grad();
    }
}

void Module::train(bool mode) {
    training_ = mode;
    for (const auto &submodule_name : submodule_names_) {
        submodules_.at(submodule_name)->train(mode);
    }
}

void Module::save(std::ostream &os) const {}

void Module::load(std::istream &is) {}

} // namespace nn
} // namespace matt