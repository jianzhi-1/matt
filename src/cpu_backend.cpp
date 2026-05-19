#include "matt/cpu_backend.hpp"
#include "matt/shape_utils.hpp"
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>

namespace matt {

static const std::unordered_map<BinaryOpType, std::function<float(float, float)>> binary_ops = {
    {BinaryOpType::Add, [](float a, float b) { return a + b; }},
    {BinaryOpType::Sub, [](float a, float b) { return a - b; }},
    {BinaryOpType::Mul, [](float a, float b) { return a * b; }},
};

static const std::unordered_map<UnaryOpType, std::function<float(float)>> unary_ops = {
    {UnaryOpType::Negate, [](float a) { return -a; }},
    {UnaryOpType::Relu, [](float a) { return (a >= 0.0f) ? a : 0.0f; }}};

static const std::unordered_map<ReduceOpType, std::function<float(float, float)>> reduce_ops = {
    {ReduceOpType::Sum, [](float a, float b) { return a + b; }},
};

float *CPUBackend::allocate(size_t n) {
    return new float[n]();
}

void CPUBackend::deallocate(float *ptr) {
    delete[] ptr;
}

void CPUBackend::fill(float *ptr, float val, size_t n) {
    std::fill(ptr, ptr + n, val);
}

void CPUBackend::elementwise_binary(const float *a, const float *b, float *out, size_t n,
                                    BinaryOpType op) {
    auto it = binary_ops.find(op);
    if (it == binary_ops.end())
        throw std::runtime_error("CPUBackend: unknown binary op");
    auto &fn = it->second;
    for (size_t i = 0; i < n; i++)
        out[i] = fn(a[i], b[i]);
}

void CPUBackend::elementwise_unary(const float *a, float *out, size_t n, UnaryOpType op) {
    auto it = unary_ops.find(op);
    if (it == unary_ops.end())
        throw std::runtime_error("CPUBackend: unknown unary op");
    auto &fn = it->second;
    for (size_t i = 0; i < n; i++)
        out[i] = fn(a[i]);
}

void CPUBackend::matmul(const float *a, const float *b, float *out, size_t M, size_t K, size_t N) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; k++) {
                acc += a[i * K + k] * b[k * N + j];
            }
            out[i * N + j] = acc;
        }
    }
}

void CPUBackend::reduce(const float *a, float *out, size_t n, ReduceOpType op) {
    auto it = reduce_ops.find(op);
    if (it == reduce_ops.end())
        throw std::runtime_error("CPUBackend: unknown reduce op");
    auto &fn = it->second;
    float acc = 0.0f;
    for (size_t i = 0; i < n; i++) {
        acc = fn(acc, a[i]);
    }
    out[0] = acc;
}

} // namespace matt