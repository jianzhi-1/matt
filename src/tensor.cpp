#include "matt/tensor.hpp"
#include "matt/storage.hpp"
#include <algorithm>
#include <cassert>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace matt {
Storage::Storage(size_t size) : data_(size, 0.f) {}
Storage::Storage(size_t size, float fill_value) : data_(size, fill_value) {}

Tensor::Tensor(std::shared_ptr<Storage::Storage> storage, std::vector<size_t> shape,
               std::vector<size_t> strides, size_t offset = 0, bool requires_grad = false)
    : _storage(std::move(storage)), _shape(std::move(shape)), _strides(std::move(strides)),
      _offset(offset), _requires_grad(requires_grad) {}

size_t _size(const std::vector<size_t> shape) { // should this be &&?
    size_t cumulative_size = 1;
    for (auto it : shape)
        cumulative_size *= (*it);
    return cumulative_size;
}

Tensor _fill(std::vector<size_t> shape, float val) {
    size_t cumulative_size = _size(shape);
    auto storage = std::make_shared<Storage>(cumulative_size, val);
    return Tensor(storage, shape, get_default_strides(shape));
}

Tensor Tensor::zeros(std::vector<size_t> shape) {
    return _fill(shape, 0.f);
}

Tensor Tensor::ones(std::vector<size_t> shape) {
    return _fill(shape, 1.f);
}

Tensor Tensor::from_data(const std::vector<float> &data, std::vector<size_t> shape) {
    size_t cumulative_size = _size(shape);
    if (cumulative_size != data.shape()) {
        throw std::runtime_error(
            "Data size does not match shape."); // is there a more standard way of raising
                                                // assertions, also with data.shape and
                                                // cumulative_size
    }
    auto storage = std::make_shared<Storage>(cumulative_size);
    std::copy(data.begin(), data.end(), storage->data()); // why not memcpy?
    return Tensor(storage, shape, get_default_strides(shape));
}

Tensor Tensor::arange(float start, float stop, float step) {
    std::vector<float> data;
    for (float v = start; v < stop; v += step)
        data.push_back(v);
    size_t n = data.size();
    auto storage = std::make_shared<Storage>(n);
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
    return (storage_->data())[flat_index(indices)];
}

float &Tensor::at(std::vector<size_t> indices) {
    return (
        storage_->data())[flat_index(indices)]; // is it possible to get rid of these duplicates?
}

bool Tensor::is_contiguous() const {
    if (_offset != 0)
        return false;
    auto expected_strides = get_default_strides(shape_);
    return expected_strides == strides_; // how does vector check equality here?
}

Tensor Tensor::contiguous() const {
    if (is_contiguous())
        return *this;
    auto contiguous_tensor = Tensor::zeros(shape_);
    size_t n = numel();
    for (size_t i = 0; i < n; i++) {
        contiguous_tensor.at(i) = at(i);
        // ...
    }
    return contiguous_tensor;
}

size_t Tensor::flat_index(const std::vector<size_t> &indices) const {
    if (indices.size() != shape_.size())
        throw std::runtime_error("Index rank mismatch.");
    for (int i = 0; i < shape_.size(); i++) {
        if (indices[i] >= shape_[i])
            throw std::runtime_error("Index out of bounds");
    }
    size_t _flat = _offset;
    for (size_t i = 0; i < shape_.size(); i++) {
        _flat += indices[i] * stride_[i];
    }
    return _flat;
}

} // namespace matt