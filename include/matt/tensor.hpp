#pragma once
#include "storage.hpp"
#include "grad_fn.hpp"
#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>
#include <numeric>
#include <iostream>
namespace matt {
    class GradFn;

    class TensorData {
        friend class Tensor;
    public:
        // grad_fn stores the derivative f' of the function f that produced this Tensor as a result.
        std::shared_ptr<GradFn> grad_fn;
        // grad stores the numerical derivative; used by optimizers during their .step().
        std::shared_ptr<TensorData> grad;
        const std::vector<size_t>& shape() const { return shape_; }
    private:
        std::shared_ptr<Storage> storage_;
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        size_t offset_ = 0;
        bool requires_grad_ = false;
    };

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
        explicit Tensor(std::shared_ptr<TensorData> data): data_(std::move(data)){};

        static Tensor zeros(std::vector<size_t> shape);
        static Tensor ones(std::vector<size_t> shape);
        static Tensor from_data(const std::vector<float>& data, std::vector<size_t> shape);
        static Tensor arange(float start, float stop, float step = 1.f);

        static Tensor fill(const std::vector<size_t>& shape, float val);

        const std::vector<size_t>& shape() const { return data_->shape_; }
        const std::vector<size_t>& strides() const { return data_->strides_; }
        size_t ndim() const { return data_->shape_.size(); }
        size_t numel() const;
        size_t offset() const { return data_->offset_; }
        bool requires_grad() const { return data_->requires_grad_; }

        float* data_ptr() { return data_->storage_->data() + data_->offset_; }
        const float* data_ptr() const { return data_->storage_->data() + data_->offset_; }

        float at(std::vector<size_t> indices) const;
        float& at(std::vector<size_t> indices);

        Tensor reshape(const std::vector<size_t>& new_shape) const;
        Tensor expand(const std::vector<size_t>& new_shape) const;
        Tensor broadcast_to(const std::vector<size_t>& target_shape) const;

        Tensor slice(size_t dim, size_t index) const;
        Tensor transpose(size_t dim0, size_t dim1) const;
        Tensor contiguous() const;
        bool is_contiguous() const;

        void set_requires_grad(bool val) { data_ -> requires_grad_ = val; }

        void print(std::ostream& os = std::cout) const;
        std::string shape_str() const;
        void backward();
        std::shared_ptr<TensorData> data() const { return data_; }

    private:
        std::shared_ptr<TensorData> data_;
        size_t flat_index(const std::vector<size_t>& indices) const;
        static std::vector<size_t> get_default_strides(const std::vector<size_t>& shape);
    };

    bool allclose(const Tensor& a, const Tensor& b, float tol=1e-5f);
}