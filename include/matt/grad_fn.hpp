#pragma once
#include "tensor.hpp"

namespace matt {

class Tensor;
class TensorData;

class GradFn {

public:
    virtual ~GradFn() = default;
    virtual std::vector<Tensor> backward(const Tensor& grad_out) const = 0;
    virtual const char* name() const = 0;

    std::vector<std::shared_ptr<TensorData>> inputs;

};

class BinaryGradFn : public GradFn {
public:
    BinaryGradFn(Tensor a, Tensor b);
protected:
    std::shared_ptr<TensorData> a_, b_;
};

class UnaryGradFn : public GradFn {
public:
    UnaryGradFn(Tensor a);
protected:
    std::shared_ptr<TensorData> a_;
};

class AddBackward: public BinaryGradFn {
public:
    using BinaryGradFn::BinaryGradFn;
    std::vector<Tensor> backward(const Tensor& grad_out) const override;
    const char* name() const override { return "AddBackward"; }
};

class SubBackward: public BinaryGradFn {
public:
    using BinaryGradFn::BinaryGradFn;
    std::vector<Tensor> backward(const Tensor& grad_out) const override;
    const char* name() const override { return "SubBackward"; }
};

class MulBackward : public BinaryGradFn {
public:
    using BinaryGradFn::BinaryGradFn;
    std::vector<Tensor> backward(const Tensor& grad_out) const override;
    const char* name() const override { return "MulBackward"; }
};

class MatmulBackward : public BinaryGradFn {
public:
    using BinaryGradFn::BinaryGradFn;
    std::vector<Tensor> backward(const Tensor& grad_out) const override;
    const char* name() const override { return "MatmulBackward"; }
};

class ReluBackward : public UnaryGradFn {
public:
    using UnaryGradFn::UnaryGradFn;
    std::vector<Tensor> backward(const Tensor& grad_out) const override;
    const char* name() const override { return "ReluBackward"; }
};

class SumBackward : public UnaryGradFn {
public:
    using UnaryGradFn::UnaryGradFn;
    std::vector<Tensor> backward(const Tensor& grad_out) const override;
    const char* name() const override { return "SumBackward"; }
};

class TransposeBackward : public UnaryGradFn {
public:
    TransposeBackward(Tensor a, size_t dim0, size_t dim1);
    std::vector<Tensor> backward(const Tensor& grad_out) const override;
    const char* name() const override { return "TransposeBackward"; }
private:
    size_t dim0_, dim1_;
};

}