#include "matt/memory_transfer.hpp"
#include "matt/cpu_backend.hpp"
#include "matt/cuda_backend.hpp"
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>
namespace matt {
void memory_transfer(float* dst, Backend* dst_backend, const float* src, Backend* src_backend, size_t n){
    bool is_dst_cuda = dst_backend->device().is_cuda();
    bool is_src_cuda = src_backend->device().is_cuda();
    size_t bytes = n * sizeof(float);
    if (!is_dst_cuda && !is_src_cuda){
        std::memcpy(dst, src, bytes);
    } else if (is_dst_cuda && !is_src_cuda){
        cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    } else if (!is_dst_cuda && is_src_cuda){
        cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    } else {
        int dst_index = dst_backend->device().index();
        int src_index = src_backend->device().index();
        if (dst_index == src_index){
            cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpyPeer(dst, dst_idx, src, src_idx, bytes);
        }
    }
}
}