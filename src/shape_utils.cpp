#include "matt/shape_utils.hpp"
namespace matt {
namespace shape_utils {
size_t numel_of(const std::vector<size_t> &shape) {
    size_t res = 1;
    for (auto it : shape) {
        res *= it;
    }
    return res;
}
} // namespace shape_utils
} // namespace matt