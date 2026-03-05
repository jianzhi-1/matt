#pragma once
#include <cstddef>
#include <memory>
#include <vector>

namespace matt {

class Storage {
    public:
    explicit Storage(size_t size);
    Storage(size_t size, float fill_value);

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    size_t size() const {return data_.size(); }

    private:
    std::vector<float> data_;
};

}
