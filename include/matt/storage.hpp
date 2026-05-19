#pragma once
#include "backend.hpp"
#include "device.hpp"
#include <cstddef>
#include <memory>
#include <vector>

namespace matt {

// A handle to memory on a device.
class Storage {
public:
    explicit Storage(size_t size, Device device);
    ~Storage();

    float* data() { return data_; }
    const float* data() const { return data_; }

    size_t size() const { return size_; }
    Device device() const { return device_; }

private:
    float* data_ = nullptr;
    size_t size_ = 0;
    Device device_;
    Backend* backend_;
};

}
