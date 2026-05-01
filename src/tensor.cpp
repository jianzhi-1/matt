#include "matt/tensor.hpp"
#include "matt/shape_utils.hpp"
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

Tensor::Tensor(std::shared_ptr<Storage> storage, std::vector<size_t> shape,
               std::vector<size_t> strides, size_t offset, bool requires_grad)
    : storage_(std::move(storage)), shape_(std::move(shape)), strides_(std::move(strides)),
      offset_(offset), requires_grad_(requires_grad) {}

Tensor Tensor::fill(const std::vector<size_t> &shape, float val) {
    size_t cumulative_size = shape_utils::numel_of(shape);
    auto storage = std::make_shared<Storage>(cumulative_size, val);
    return Tensor(storage, shape, get_default_strides(shape));
}

Tensor Tensor::zeros(std::vector<size_t> shape) {
    return Tensor::fill(shape, 0.f);
}

Tensor Tensor::ones(std::vector<size_t> shape) {
    return Tensor::fill(shape, 1.f);
}

Tensor Tensor::from_data(const std::vector<float> &data, std::vector<size_t> shape) {
    size_t cumulative_size = shape_utils::numel_of(shape);
    if (cumulative_size != data.size()) {
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
    if (offset_ != 0)
        return false;
    auto expected_strides = get_default_strides(shape_);
    return expected_strides == strides_; // how does vector check equality here?
}

size_t Tensor::numel() const {
    size_t n = 1;
    for (auto d : shape_) {
        n *= d;
    }
    return n;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous())
        return *this;
    auto contiguous_tensor = Tensor::zeros(shape_);
    size_t n = numel();
    std::vector<size_t> idx(ndim(), 0);
    for (size_t i = 0; i < n; i++) {
        contiguous_tensor.at(idx) = at(idx);
        for (int d = ndim() - 1; d >= 0; d--) {
            if (++idx[d] < shape_[d])
                break;
            idx[d] = 0;
        }
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
    size_t _flat = offset_;
    for (size_t i = 0; i < shape_.size(); i++) {
        _flat += indices[i] * strides_[i];
    }
    return _flat;
}

} // namespace matt