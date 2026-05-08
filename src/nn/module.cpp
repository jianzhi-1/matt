#include "module.hpp"
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
    // TODO: recurse into submodules to get all parameters
    // for now, just return all parameters in self.
    std::vector<std::pair<std::string, Tensor>> named_params;
    named_params.reserve(parameters_.size());
    for (const auto &param_name : parameter_names_) {
        named_params.push_back(std::make_pair(param_name, parameters_.at(param_name)));
    }
    return named_params;
}

std::vector<Tensor> Module::parameters() const {
    // TODO: recurse into submodules to get all parameters
    // for now, just return all parameters in self.
    std::vector<Tensor> params;
    params.reserve(parameters_.size());
    for (const auto &[key, value] : parameters_) {
        params.push_back(value);
    }
    return params;
}

void Module::zero_grad() {
    // TODO
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