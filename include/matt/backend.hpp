#pragma once
#include <cstddef>
#include "device.hpp"

namespace matt {

enum class BinaryOpType {
    Add, Sub, Mul
};

enum class UnaryOpType { Relu, Negate };
enum class ReduceOpType { Sum };

// Represents the memory and compute interfaces for a single device.
class Backend {
public:
    virtual ~Backend() = default;

    virtual Device device() const = 0;

    virtual float* allocate(size_t n) = 0;
    virtual void deallocate(float* ptr) = 0;
    virtual void fill(float* ptr, float val, size_t n) = 0;

    // Operations
    virtual void elementwise_binary(
        const float* a, const float* b, float* out, size_t n, BinaryOpType op
    ) = 0;

    virtual void elementwise_unary(
        const float* a, float* out, size_t n, UnaryOpType op
    ) = 0;

    virtual void matmul(
        const float* a, const float* b, float* out, size_t M, size_t K, size_t N
    ) = 0;

    virtual void reduce(
        const float* a, float* out, size_t n, ReduceOpType op
    ) = 0;
};

Backend* get_backend(Device device);


}

