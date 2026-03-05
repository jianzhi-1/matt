#pragma once
#include <vector>
#include <memory>
#include <cassert>
#include <numeric>
#include <iostream>


class Tensor {
    
}

class Tensor {
public:
    using Shape = std::vector<size_t>;

private:
    std::shared_ptr<float[]> data_;
    Shape shape_;
    Shape strides_;
    size_t offset_;

public:
    Tensor(const Shape& shape)
        : shape_(shape), offset_(0)
    {
        compute_strides();
        size_t total = numel();
        data_ = std::shared_ptr<float[]>(new float[total]());
    }

    // ---- Core Info ----
    const Shape& shape() const { return shape_; }
    const Shape& strides() const { return strides_; }
    size_t offset() const { return offset_; }

    size_t dim() const { return shape_.size(); }

    size_t numel() const {
        return std::accumulate(
            shape_.begin(), shape_.end(),
            1ULL, std::multiplies<size_t>()
        );
    }

    // ---- Indexing ----
    float& operator()(const Shape& indices) {
        assert(indices.size() == dim());
        size_t flat_index = offset_;
        for (size_t i = 0; i < dim(); ++i) {
            assert(indices[i] < shape_[i]);
            flat_index += indices[i] * strides_[i];
        }
        return data_.get()[flat_index];
    }

    const float& operator()(const Shape& indices) const {
        return const_cast<Tensor&>(*this)(indices);
    }

    // ---- Fill ----
    void fill(float value) {
        for (size_t i = 0; i < numel(); ++i)
            data_.get()[i] = value;
    }

    // ---- Debug ----
    void print_info() const {
        std::cout << "Shape: ";
        for (auto s : shape_) std::cout << s << " ";
        std::cout << "\nStrides: ";
        for (auto s : strides_) std::cout << s << " ";
        std::cout << "\nOffset: " << offset_ << "\n";
    }

private:
    void compute_strides() {
        strides_.resize(dim());
        if (dim() == 0) return;

        strides_.back() = 1;
        for (int i = dim() - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }
};