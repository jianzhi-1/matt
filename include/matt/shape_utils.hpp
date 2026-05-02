#pragma once
#include <vector>
#include <stdexcept>
namespace matt {

namespace shape_utils {
    size_t numel_of(const std::vector<size_t>& shape);
    std::vector<size_t> broadcast_shape(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2);
}
}