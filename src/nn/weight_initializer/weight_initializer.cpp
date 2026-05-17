#include "matt/nn/weight_initializer/weight_initializer.hpp"
#include "matt/shape_utils.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

namespace matt {
namespace nn {
namespace weight_initializer {

Tensor Zeros::initialize(const std::vector<size_t>& shape) const {
    return Tensor::zeros(shape);
}

Uniform::Uniform(float low, float high, uint32_t seed) : low_(low), high_(high), seed_(seed) {}

Tensor Uniform::initialize(const std::vector<size_t>& shape) const {
    std::mt19937 rng(seed_);
    std::uniform_real_distribution<float> dist(low_, high_);
    std::vector<float> data(shape_utils::numel_of(shape));
    std::generate(data.begin(), data.end(), [&] { return dist(rng); });
    return Tensor::from_data(data, shape);
}

KaimingUniform::KaimingUniform(uint32_t seed) : seed_(seed) {}

Tensor KaimingUniform::initialize(const std::vector<size_t>& shape) const {
    if (shape.size() < 2)
        throw std::runtime_error("KaimingUniform: shape must be at least 2D");
    float fan_in = static_cast<float>(shape[1]);
    float bound = std::sqrt(1.0f / fan_in);
    std::mt19937 rng(seed_);
    std::uniform_real_distribution<float> dist(-bound, bound);
    std::vector<float> data(shape_utils::numel_of(shape));
    std::generate(data.begin(), data.end(), [&] { return dist(rng); });
    return Tensor::from_data(data, shape);
}

Normal::Normal(float mu, float sigma, uint32_t seed) : mu_(mu), sigma_(sigma), seed_(seed) {}

Tensor Normal::initialize(const std::vector<size_t>& shape) const {
    std::mt19937 rng(seed_);
    std::normal_distribution<float> dist(mu_, sigma_);
    std::vector<float> data(shape_utils::numel_of(shape));
    std::generate(data.begin(), data.end(), [&] { return dist(rng); });
    return Tensor::from_data(data, shape);
}

} // namespace weight_initializer
} // namespace nn
} // namespace matt
