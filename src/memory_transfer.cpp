#include "matt/memory_transfer.hpp"
#include "matt/cpu_backend.hpp"
#include <cstring>

namespace matt {
#ifndef MATT_CUDA
void memory_transfer(float *dst, Backend *dst_backend, const float *src, Backend *src_backend,
                     size_t n) {
    if (!dst_backend->device().is_cpu() || !src_backend->device().is_cpu())
        throw std::runtime_error("memory_transfer: CUDA is not compiled");
    std::memcpy(dst, src, n * sizeof(float));
}
#endif
} // namespace matt