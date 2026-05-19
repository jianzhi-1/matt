#pragma once
#include "backend.hpp"

namespace matt {
// A function that encodes the D x D combinations of backend transfers, where D is the number of
// device types. 
void memory_transfer(float* dst, Backend* dst_backend, const float* src, Backend* src_backend, size_t n);
}