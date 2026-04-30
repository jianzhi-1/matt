#pragma once
#include "storage.hpp"
#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>
#include <numeric>
#include <iostream>
namespace matt {
    class Tensor {
        public:
        Tensor() = default;

        Tensor(
            std::shared_ptr<Storage> storage,
            std::vector<size_t> shape,
            std::vector<size_t> strides,
            size_t offset = 0,
            bool requires_grad = false
        );

        static Tensor zeros(std::vector<size_t> shape);
        static Tensor ones(std::vector<size_t> shape);
        static Tensor from_data(const std::vector<float>& data, std::vector<size_t> shape);
        static Tensor arange(float start, float stop, float step = 1.f);

        static Tensor fill(const std::vector<size_t>& shape, float val);

        const std::vector<size_t>& shape() const { return shape_; }
        const std::vector<size_t>& strides() const { return strides_; }
        size_t ndim() const { return shape_.size(); }
        size_t numel() const;
        size_t offset() const { return offset_; }
        bool requires_grad() const { return requires_grad_; }

        float* data_ptr() { return storage_->data() + offset_; }
        const float* data_ptr() const { return storage_->data() + offset_; }

        float at(std::vector<size_t> indices) const;
        float& at(std::vector<size_t> indices);

        Tensor reshape(std::vector<size_t> new_shape) const;
        Tensor transpose(size_t dim0, size_t dim1) const;
        Tensor contiguous() const;
        bool is_contiguous() const;

        void set_requires_grad(bool val) { requires_grad_ = val; }

        void print(std::ostream& os = std::cout) const;
        std::string shape_str() const;

        private:
        std::shared_ptr<Storage> storage_;
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        size_t offset_ = 0;
        bool requires_grad_ = false;

        size_t flat_index(const std::vector<size_t>& indices) const;
        static std::vector<size_t> get_default_strides(const std::vector<size_t>& shape);
    };
}