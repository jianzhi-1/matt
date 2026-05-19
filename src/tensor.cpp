#include "matt/tensor.hpp"
#include "matt/grad_fn.hpp"
#include "matt/ops.hpp"
#include "matt/shape_utils.hpp"
#include "matt/storage.hpp"
#include <algorithm>
#include <cassert>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace matt {

Tensor::Tensor(std::shared_ptr<Storage> storage, std::vector<size_t> shape,
               std::vector<size_t> strides, size_t offset, bool requires_grad)
    : data_(std::make_shared<TensorData>()) {
    data_->storage_ = std::move(storage);
    data_->shape_ = std::move(shape);
    data_->strides_ = std::move(strides);
    data_->offset_ = offset;
    data_->requires_grad_ = requires_grad;
}

Tensor Tensor::fill(const std::vector<size_t> &shape, float val, Device device) {
    size_t cumulative_size = shape_utils::numel_of(shape);
    auto storage = std::make_shared<Storage>(cumulative_size, device);
    get_backend(device)->fill(storage->data(), val, cumulative_size);
    return Tensor(storage, shape, get_default_strides(shape));
}

Tensor Tensor::zeros(std::vector<size_t> shape, Device device) {
    return Tensor::fill(shape, 0.f);
}

Tensor Tensor::ones(std::vector<size_t> shape, Device device) {
    return Tensor::fill(shape, 1.f);
}

Tensor Tensor::from_data(const std::vector<float> &data, std::vector<size_t> shape, Device device) {
    size_t cumulative_size = shape_utils::numel_of(shape);
    if (cumulative_size != data.size()) {
        throw std::runtime_error(
            "Data size does not match shape."); // is there a more standard way of raising
                                                // assertions, also with data.shape and
                                                // cumulative_size
    }
    auto storage = std::make_shared<Storage>(cumulative_size, device);
    std::copy(data.begin(), data.end(), storage->data()); // why not memcpy?
    return Tensor(storage, shape, get_default_strides(shape));
}

Tensor Tensor::arange(float start, float stop, float step, Device device) {
    std::vector<float> data;
    for (float v = start; v < stop; v += step)
        data.push_back(v);
    size_t n = data.size();
    auto storage = std::make_shared<Storage>(n, device);
    std::copy(data.begin(), data.end(), storage->data());
    return Tensor(storage, {n}, {1});
}

std::vector<size_t> Tensor::get_default_strides(const std::vector<size_t> &shape) {
    std::vector<size_t> strides(shape.size());
    size_t cumulative_stride = 1;
    for (int i = (int)shape.size() - 1; i >= 0; i--) {
        strides[i] = cumulative_stride;
        cumulative_stride *= shape[i];
    }
    return strides;
}

float Tensor::at(std::vector<size_t> indices) const {
    size_t idx = flat_index(indices);
    if (device().is_cpu()) {
        return data_->storage_->data()[idx];
    }
#ifdef MATT_CUDA
    float val;
    cudaMemcpy(&val, data_->storage_->data() + idx, sizeof(float), cudaMemcpyDeviceToHost);
    return val;
#endif
    throw std::runtime_error("Tensor::at: unsupported device");
}

float &Tensor::at(std::vector<size_t> indices) {
    if (!device().is_cpu())
        throw std::runtime_error("Tensor::at: mutable at only supported on CPU");
    return data_->storage_->data()[flat_index(indices)];
}

bool Tensor::is_contiguous() const {
    if (data_->offset_ != 0)
        return false;
    auto expected_strides = get_default_strides(data_->shape_);
    return expected_strides == data_->strides_; // how does vector check equality here?
}

size_t Tensor::numel() const {
    size_t n = 1;
    for (auto d : data_->shape_) {
        n *= d;
    }
    return n;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous())
        return *this;
    auto contiguous_tensor = Tensor::zeros(data_->shape_, device());
    size_t n = numel();
    std::vector<size_t> idx(ndim(), 0);
    for (size_t i = 0; i < n; i++) {
        contiguous_tensor.at(idx) = at(idx);
        for (int d = ndim() - 1; d >= 0; d--) {
            if (++idx[d] < data_->shape_[d])
                break;
            idx[d] = 0;
        }
    }
    return contiguous_tensor;
}

size_t Tensor::flat_index(const std::vector<size_t> &indices) const {
    if (indices.size() != data_->shape_.size())
        throw std::runtime_error("Index rank mismatch.");
    for (int i = 0; i < data_->shape_.size(); i++) {
        if (indices[i] >= data_->shape_[i])
            throw std::runtime_error("Index out of bounds");
    }
    size_t _flat = data_->offset_;
    for (size_t i = 0; i < data_->shape_.size(); i++) {
        _flat += indices[i] * data_->strides_[i];
    }
    return _flat;
}

Tensor Tensor::reshape(const std::vector<size_t> &new_shape) const {
    // TODO: remove this constraint.
    if (!is_contiguous())
        throw std::runtime_error("reshape: tensor is not contiguous");
    if (shape_utils::numel_of(new_shape) != numel())
        throw std::runtime_error("reshape: element count mismatch");

    return Tensor(data_->storage_, new_shape, get_default_strides(new_shape), data_->offset_,
                  data_->requires_grad_);
}

Tensor Tensor::expand(const std::vector<size_t> &new_shape) const {
    if (new_shape.size() != ndim())
        throw std::runtime_error("expand: rank mismatch");
    auto new_strides = data_->strides_;
    for (size_t i = 0; i < ndim(); i++) {
        if (data_->shape_[i] == 1 && new_shape[i] != 1) {
            new_strides[i] = 0;
        } else if (data_->shape_[i] != new_shape[i])
            throw std::runtime_error("expand: incompatible shapes");
    }
    return Tensor(data_->storage_, new_shape, new_strides, data_->offset_, data_->requires_grad_);
}

Tensor Tensor::broadcast_to(const std::vector<size_t> &target_shape) const {
    auto final_shape = shape_utils::broadcast_shape(data_->shape_, target_shape);
    if (final_shape != target_shape)
        throw std::runtime_error("broadcast_to: invalid shape");
    // TODO: make the following logic less convoluted.
    std::vector<size_t> new_strides = data_->strides_;
    std::vector<size_t> tmp_shape = data_->shape_;
    std::reverse(tmp_shape.begin(), tmp_shape.end());
    std::reverse(new_strides.begin(), new_strides.end());
    while (tmp_shape.size() < target_shape.size()) {
        tmp_shape.push_back(1);
        new_strides.push_back(0);
    }
    std::reverse(tmp_shape.begin(), tmp_shape.end());
    std::reverse(new_strides.begin(), new_strides.end());
    for (int i = 0; i < tmp_shape.size(); i++) {
        if ((tmp_shape[i] == 1) && (target_shape[i] != 1)) {
            new_strides[i] = 0;
        }
    }
    return Tensor(data_->storage_, target_shape, new_strides, data_->offset_,
                  data_->requires_grad_);
}

// TODO: support range slicing
Tensor Tensor::slice(size_t dim, size_t index) const {
    // TODO: support 0D tensors (?)
    if (dim >= ndim())
        throw std::runtime_error("slice: invalid dimension");
    if (index >= data_->shape_[dim])
        throw std::out_of_range("slice: index out of bound");
    std::vector<size_t> new_shape;
    std::vector<size_t> new_strides;
    for (int i = 0; i < data_->shape_.size(); i++) {
        if (i == dim)
            continue;
        new_shape.push_back(data_->shape_[i]);
        new_strides.push_back(data_->strides_[i]);
    }
    size_t new_offset = data_->offset_ + index * data_->strides_[dim];
    return Tensor(data_->storage_, new_shape, new_strides, new_offset, data_->requires_grad_);
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
    if (dim0 >= ndim() || dim1 >= ndim())
        throw std::out_of_range("transpose: invalid dimensions");
    auto new_shape = data_->shape_;
    auto new_strides = data_->strides_;
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);
    Tensor out(data_->storage_, new_shape, new_strides, data_->offset_, data_->requires_grad_);
    // NOTE: treat transpose as an operation, so that gradients backpropagate correctly.
    if (data_->requires_grad_) {
        out.data()->grad_fn = std::make_shared<TransposeBackward>(*this, dim0, dim1);
        out.data()->grad_fn->inputs = {data_};
    }
    return out;
}

void Tensor::backward() {
    // TODO: improve verbosity of error messages
    if (numel() != 1)
        throw std::runtime_error("backward: for now, can only call on scalar values");
    data_->grad = Tensor::ones(data_->shape_).data();

    std::vector<TensorData *> topo;
    std::unordered_set<TensorData *> visited;

    std::function<void(TensorData *)> dfs;

    dfs = [&](TensorData *node) {
        visited.insert(node);
        if (!node->grad_fn)
            return;
        for (auto it : node->grad_fn->inputs) {
            if (visited.find(it.get()) != visited.end())
                continue;
            dfs(it.get());
        }
        topo.push_back(node);
    };
    dfs(data_.get());
    std::reverse(topo.begin(), topo.end());

    for (auto t : topo) {
        if (!t->grad_fn)
            continue;
        if (!t->grad)
            throw std::runtime_error("backward: gradient should have existed");
        auto input_grads = t->grad_fn->backward(Tensor(t->grad));
        auto &inputs = t->grad_fn->inputs;
        for (size_t i = 0; i < inputs.size(); i++) {
            if (!inputs[i]->requires_grad_)
                continue;
            if (!inputs[i]->grad) {
                inputs[i]->grad = input_grads[i].data();
            } else {
                auto accumulated_gradient = ops::add(Tensor(inputs[i]->grad), input_grads[i]);
                inputs[i]->grad = accumulated_gradient.data();
            }
        }
    }
}

bool allclose(const Tensor &a, const Tensor &b, float tol) {
    auto imm =
        ops::elementwise(a, b, [tol](float x, float y) { return abs(x - y) < tol ? 0.0 : 1.0; });
    auto out = ops::accumulate(imm, [](float x, float y) { return x + y; });
    // TODO: please use a better way.
    return out.at({0}) < 0.5;
}

Device Tensor::device() const {
    return data_->storage_->device();
}

} // namespace matt