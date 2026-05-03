#pragma once
#include <gtest/gtest.h>
#include "matt/tensor.hpp"

namespace matt {
namespace test_utils {

inline void expect_tensors_close(const Tensor& a, const Tensor& b, float tol = 1e-5f) {
    EXPECT_TRUE(matt::allclose(a, b, tol));
}

} // namespace test_utils
} // namespace matt