#pragma once
#include <string>
#include <stdexcept>

namespace matt {

enum class DeviceType {
    CPU, CUDA
};

class Device {

public:
    Device(DeviceType device_type = DeviceType::CPU, int index = 0): device_type_(device_type), index_(index){};

    static Device cpu() { return {DeviceType::CPU, 0}; }
    static Device cuda(int index){ return {DeviceType::CUDA, index}; }

    bool is_cpu() const { return device_type_ == DeviceType::CPU; }
    bool is_cuda() const { return device_type_ == DeviceType::CUDA; }

    bool operator==(const Device& other) const {
        return device_type_ == other.device_type_ && index_ == other.index_;
    }

    bool operator!=(const Device& other) const {
        return !(*this == other);
    }

    std::string str() const {
        if (is_cpu()) return "cpu";
        return "cuda:" + std::to_string(index_);
    }

private:
    DeviceType device_type_;
    int index_;
};

}