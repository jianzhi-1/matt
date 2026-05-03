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

Tensor Tensor::reshape(const std::vector<size_t> &new_shape) const {
    // TODO: remove this constraint.
    if (!is_contiguous())
        throw std::runtime_error("reshape: tensor is not contiguous");
    if (shape_utils::numel_of(new_shape) != numel())
        throw std::runtime_error("reshape: element count mismatch");

    return Tensor(storage_, new_shape, get_default_strides(new_shape), offset_, requires_grad_);
}

Tensor Tensor::expand(const std::vector<size_t> &new_shape) const {
    if (new_shape.size() != ndim())
        throw std::runtime_error("expand: rank mismatch");
    auto new_strides = strides_;
    for (size_t i = 0; i < ndim(); i++) {
        if (shape_[i] == 1 && new_shape[i] != 1) {
            new_strides[i] = 0;
        } else if (shape_[i] != new_shape[i])
            throw std::runtime_error("expand: incompatible shapes");
    }
    return Tensor(storage_, new_shape, new_strides, offset_, requires_grad_);
}

Tensor Tensor::broadcast_to(const std::vector<size_t> &target_shape) const {
    auto final_shape = shape_utils::broadcast_shape(shape_, target_shape);
    if (final_shape != target_shape)
        throw std::runtime_error("broadcast_to: invalid shape");
    // TODO: make the following logic less convoluted.
    std::vector<size_t> new_strides = strides_;
    std::vector<size_t> tmp_shape = shape_;
    reverse(tmp_shape.begin(), tmp_shape.end());
    reverse(new_strides.begin(), new_strides.end());
    while (tmp_shape.size() < target_shape.size()) {
        tmp_shape.push_back(1);
        new_strides.push_back(0);
    }
    reverse(tmp_shape.begin(), tmp_shape.end());
    reverse(new_strides.begin(), new_strides.end());
    for (int i = 0; i < tmp_shape.size(); i++) {
        if ((tmp_shape[i] == 1) && (target_shape[i] != 1)) {
            new_strides[i] = 0;
        }
    }
    return Tensor(storage_, target_shape, new_strides, offset_, requires_grad_);
}

// TODO: support range slicing
Tensor Tensor::slice(size_t dim, size_t index) const {
    // TODO: support 0D tensors (?)
    if (dim >= ndim())
        throw std::runtime_error("slice: invalid dimension");
    if (index >= shape_[dim])
        throw std::out_of_range("slice: index out of bound");
    std::vector<size_t> new_shape;
    std::vector<size_t> new_strides;
    for (int i = 0; i < shape_.size(); i++) {
        if (i == dim)
            continue;
        new_shape.push_back(shape_[i]);
        new_strides.push_back(strides_[i]);
    }
    size_t new_offset = offset_ + index * strides_[dim];
    return Tensor(storage_, new_shape, new_strides, new_offset, requires_grad_);
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
    if (dim0 >= ndim() || dim1 >= ndim())
        throw std::out_of_range("transpose: invalid dimensions");
    auto new_shape = shape_;
    auto new_strides = strides_;
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);
    return Tensor(storage_, new_shape, new_strides, offset_, requires_grad_);
}

} // namespace matt