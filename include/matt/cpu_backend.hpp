#pragma once
#include "backend.hpp"
namespace matt {

class CPUBackend : public Backend {
public:
    static CPUBackend* get(){
        static CPUBackend instance;
        return &instance;
    }

    Device device() const override { return Device::cpu(); }

    float* allocate(size_t n) override;
    void deallocate(float* ptr) override;
    void fill(float* ptr, float val, size_t n) override;

    void elementwise_binary(
        const float* a, const float* b, float* out, size_t n, BinaryOpType op
    ) override;

    void elementwise_unary(
        const float* a, float* out, size_t n, UnaryOpType op
    ) override;

    void matmul(
        const float* a, const float* b, float* out, size_t M, size_t K, size_t N
    ) override;

    void reduce(
        const float* a, float* out, size_t n, ReduceOpType op
    ) override;


private:
    CPUBackend() = default;
};

}