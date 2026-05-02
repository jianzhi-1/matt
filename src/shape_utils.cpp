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

std::vector<size_t> broadcast_shape(const std::vector<size_t> &shape1,
                                    const std::vector<size_t> &shape2) {
    // https://numpy.org/doc/stable/user/basics.broadcasting.html
    // https://docs.pytorch.org/docs/2.11/notes/broadcasting.html
    size_t s1_size = shape1.size();
    size_t s2_size = shape2.size();
    size_t mini = std::min(s1_size, s2_size);
    size_t maxi = std::max(s1_size, s2_size);
    std::vector<size_t> new_shape;
    for (int i = 0; i < maxi; i++) {
        if (i < mini) {
            size_t dim1 = shape1[s1_size - 1 - i];
            size_t dim2 = shape2[s2_size - 1 - i];
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                throw std::runtime_error("broadcast_shape: cannot be broadcasted");
            }
            new_shape.push_back(std::max(dim1, dim2));
        } else {
            if (s1_size > i) {
                new_shape.push_back(shape1[s1_size - 1 - i]);
            } else {
                new_shape.push_back(shape2[s2_size - 1 - i]);
            }
        }
    }
    std::reverse(new_shape.begin(), new_shape.end());
    return new_shape;
}

} // namespace shape_utils
} // namespace matt