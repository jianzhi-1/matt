#include "matt/nn/activations/relu.hpp"
#include "matt/nn/linear.hpp"
#include "matt/nn/loss.hpp"
#include "matt/nn/sequential.hpp"
#include "matt/nn/weight_initializer/weight_initializer.hpp"
#include "matt/ops.hpp"
#include "matt/optim/sgd.hpp"
#include "matt/tensor.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace matt;
using namespace matt::nn;
using namespace matt::optim;
using namespace matt::nn::weight_initializer;

PYBIND11_MODULE(matt, m) {
    m.doc() = "matt library";

    // Device
    py::class_<Device>(m, "Device")
        .def_static("cpu", &Device::cpu)
        .def_static("cuda", &Device::cuda, py::arg("index") = 0)
        .def("is_cpu", &Device::is_cpu)
        .def("is_cuda", &Device::is_cuda)
        .def("__repr__", &Device::str);

    // Tensor
    py::class_<Tensor>(m, "Tensor")
        .def_static("zeros", &Tensor::zeros, py::arg("shape"), py::arg("device") = Device::cpu())
        .def_static("ones", &Tensor::ones, py::arg("shape"), py::arg("device") = Device::cpu())
        .def_static("from_data", &Tensor::from_data, py::arg("data"), py::arg("shape"),
                    py::arg("device") = Device::cpu())
        .def("shape", &Tensor::shape)
        .def("numel", &Tensor::numel)
        .def("ndim", &Tensor::ndim)
        .def("requires_grad", &Tensor::requires_grad)
        .def("set_requires_grad", &Tensor::set_requires_grad)
        .def("backward", &Tensor::backward)
        .def("transpose", &Tensor::transpose)
        .def("at", [](const Tensor &t, std::vector<size_t> indices) { return t.at(indices); });

    // Ops
    py::module_ ops = m.def_submodule("ops", "operations");
    ops.def("add", &ops::add);
    ops.def("sub", &ops::sub);
    ops.def("mul", &ops::mul);
    ops.def("matmul", &ops::matmul);
    ops.def("relu", &ops::relu);
    ops.def("sum", &ops::sum);

    // NN
    py::module_ nn = m.def_submodule("nn", "neural network");

    py::class_<Module, std::shared_ptr<Module>>(nn, "Module")
        .def("parameters", &Module::parameters)
        .def("named_parameters", &Module::named_parameters)
        .def("zero_grad", &Module::zero_grad)
        .def("train", &Module::train, py::arg("mode") = true)
        .def("is_training", &Module::is_training);

    py::class_<Linear, Module, std::shared_ptr<Linear>>(nn, "Linear")
        .def(py::init<size_t, size_t, const WeightInitializer &, bool>(), py::arg("input_dim"),
             py::arg("output_dim"), py::arg("weight_initializer"), py::arg("use_bias") = true)
        .def("forward", &Linear::forward);

    py::class_<ReLU, Module, std::shared_ptr<ReLU>>(nn, "ReLU")
        .def(py::init<>())
        .def("forward", &ReLU::forward);

    py::class_<Sequential, Module, std::shared_ptr<Sequential>>(nn, "Sequential")
        .def(py::init<>())
        .def("add", &Sequential::add_module)
        .def("forward", &Sequential::forward);

    py::class_<MSELoss, std::shared_ptr<MSELoss>>(nn, "MSELoss")
        .def(py::init<>())
        .def("forward", &MSELoss::forward);

    // Weight initializers
    py::module_ wi = nn.def_submodule("weight_initializer", "weight initializers");

    py::class_<WeightInitializer>(wi, "WeightInitializer");

    py::class_<Zeros, WeightInitializer>(wi, "Zeros")
        .def(py::init<>())
        .def("initialize", &Zeros::initialize);

    py::class_<Uniform, WeightInitializer>(wi, "Uniform")
        .def(py::init<float, float, uint32_t>(), py::arg("low"), py::arg("high"), py::arg("seed"))
        .def("initialize", &Uniform::initialize);

    py::class_<Normal, WeightInitializer>(wi, "Normal")
        .def(py::init<float, float, uint32_t>(), py::arg("mu"), py::arg("sigma"), py::arg("seed"))
        .def("initialize", &Normal::initialize);

    py::class_<KaimingUniform, WeightInitializer>(wi, "KaimingUniform")
        .def(py::init<uint32_t>(), py::arg("seed"))
        .def("initialize", &KaimingUniform::initialize);

    // Optim
    py::module_ optim = m.def_submodule("optim", "optimizers");

    py::class_<Optimizer>(optim, "Optimizer")
        .def("zero_grad", &Optimizer::zero_grad)
        .def("lr", &Optimizer::lr)
        .def("set_lr", &Optimizer::set_lr);

    py::class_<SGD, Optimizer>(optim, "SGD")
        .def(py::init([](py::list params, float lr) {
                 std::vector<Tensor> cpp_params;
                 for (auto &p : params) {
                     cpp_params.push_back(p.cast<Tensor>());
                 }
                 return std::make_unique<SGD>(cpp_params, lr);
             }),
             py::arg("params"), py::arg("lr"))
        .def("step", &SGD::step);
}