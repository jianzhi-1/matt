#include "matt/backend.hpp"
#include "matt/cpu_backend.hpp"
#ifdef MATT_CUDA
#include "matt/cuda_backend.hpp"
#endif
#include <stdexcept>

namespace matt {
Backend *get_backend(Device device) {
    if (device.is_cpu())
        return CPUBackend::get();
#ifdef MATT_CUDA
    if (device.is_cuda())
        return CUDABackend::get(device.index);
#endif
    throw std::runtime_error("Backend::get_backend: unknown device");
}
} // namespace matt