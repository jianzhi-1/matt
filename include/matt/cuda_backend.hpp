#pragma once
#include "backend.hpp"
#include "device.hpp"
#include <unordered_map>

namespace matt {

class CUDABackend : public Backend {
public:
    static CUDABackend* get(int index = 0){
        static std::unordered_map<int, CUDABackend*> instances;
        if (!instances.count(index)) instances[index] = new CUDABackend(index);
        return instances[index];
    }

    Device device() const override { return Device::cuda(index_); }

    float* allocate(size_t n) override;
    void deallocate(float* ptr) override;
    void fill(float* ptr, float val, size_t n) override;

    // Operations
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
    explicit CUDABackend(int index): index_(index){};
    int index_;
};

}