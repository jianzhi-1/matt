#pragma once
#include "matt/tensor.hpp"
#include <memory>

namespace matt {
namespace nn {
namespace weight_initializer {

class WeightInitializer {
public:
    virtual ~WeightInitializer() = default;
    virtual Tensor initialize(const std::vector<size_t>& shape) const = 0;
};

class Zeros: public WeightInitializer {
public:
    Tensor initialize(const std::vector<size_t>& shape) const override;
};

class Uniform : public WeightInitializer {
public:
    Uniform(float low, float high, uint32_t seed);
    Tensor initialize(const std::vector<size_t>& shape) const override;

private:
    float low_, high_;
    uint32_t seed_;
};

class KaimingUniform: public WeightInitializer {
public:
    KaimingUniform(uint32_t seed);
    Tensor initialize(const std::vector<size_t>& shape) const override;
private:
    uint32_t seed_;
};

class Normal : public WeightInitializer {
public:
    Normal(float mu, float sigma, uint32_t seed);
    Tensor initialize(const std::vector<size_t>& shape) const override;
private:
    float mu_, sigma_;
    uint32_t seed_;
};

}
}
}