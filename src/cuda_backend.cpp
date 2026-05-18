#include "matt/cuda_backend.hpp"
#include "matt/shape_utils.hpp"
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#ifdef MATT_CUDA
#include <cuda_runtime.h>
#endif

namespace matt {

float *CUDABackend::allocate(size_t n) {
#ifdef MATT_CUDA
    float* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, n * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return ptr;
#endif
    throw std::runtime_error("Matt built without CUDA support");
}

void CUDABackend::deallocate(float *ptr) {
#ifdef MATT_CUDA
    if (ptr == nullptr) throw std::runtime_error("CUDABackend::deallocate: deallocating a nullptr");
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return;
#endif
    throw std::runtime_error("Matt built without CUDA support");
}

#ifdef MATT_CUDA
__global__ void fill_kernel(float* ptr, float val, size_t n){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = val;
}
#endif

void CUDABackend::fill(float *ptr, float val, size_t n) {
    #ifdef MATT_CUDA
    if (val == 0.0f){
        cudaMemset(ptr, 0, n * sizeof(float));
        return;
    }
    constexpr int threads = 256;
    int blocks = (n + threads - 1)/threads;
    fill_kernel<<<blocks, threads>>>(ptr, val, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return;
    #endif
    throw std::runtime_error("Matt built without CUDA support");
}

void CUDABackend::elementwise_binary(const float *a, const float *b, float *out, size_t n,
                                    BinaryOpType op) {
    // TODO
}

void CUDABackend::elementwise_unary(const float *a, float *out, size_t n, UnaryOpType op) {
    // TODO
}

void CUDABackend::matmul(const float *a, const float *b, float *out, size_t M, size_t K, size_t N) {
    // TODO
}

void CUDABackend::reduce(const float *a, float *out, size_t n, ReduceOpType op) {
    // TODO
}

} // namespace matt