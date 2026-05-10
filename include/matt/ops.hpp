#pragma once
#include "tensor.hpp"
#include "grad_fn.hpp"
namespace matt {
namespace ops {

class Op {
public:
};

template<typename GradFnType>
class BinaryOp : public Op {
public:
    static std::shared_ptr<GradFn> make_grad_fn(const Tensor& a, const Tensor& b){
        return std::make_shared<GradFnType>(a, b);
    }
};

template<typename GradFnType>
class UnaryOp : public Op {
public:
    static std::shared_ptr<GradFn> make_grad_fn(const Tensor& a){
        return std::make_shared<GradFnType>(a);
    }
};

class AddOp: public BinaryOp<AddBackward>{
public:
    static Tensor forward(const Tensor& a, const Tensor& b);
};

class SubOp: public BinaryOp<AddBackward>{
public:
    static Tensor forward(const Tensor& a, const Tensor& b);
};

class MulOp: public BinaryOp<MulBackward>{
public:
    static Tensor forward(const Tensor& a, const Tensor& b);
};

class MatmulOp: public BinaryOp<MatmulBackward>{
public:
    static Tensor forward(const Tensor& a, const Tensor& b);
};

class ReluOp: public UnaryOp<ReluBackward>{
public:
    static Tensor forward(const Tensor& a);
};

class SumOp: public UnaryOp<SumBackward>{
public:
    static Tensor forward(const Tensor& a);
};


Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor relu(const Tensor& a);
Tensor sum(const Tensor& a);
Tensor softmax(const Tensor& a, int dim);

Tensor elementwise_unary(const Tensor &a, std::function<float(float)> op);
Tensor elementwise(const Tensor &a, const Tensor &b, std::function<float(float, float)> op);
Tensor accumulate(const Tensor &a, std::function<float(float, float)> op);


}
}