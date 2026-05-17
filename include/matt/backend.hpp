#pragma once
#include <cstddef>
#include "device.hpp"

namespace matt {

enum class BinaryOpType {
    Add, Sub
}

class Backend {
public:
    virtual float* allocate(size_t n) = 0;
    virtual void deallocate(float* ptr) = 0;
    virtual void fill(float* ptr, float val, size_t n) = 0;

    virtual void copy(

    ) = 0;

    virtual void elementwise_binary(

    ) = 0;

    virtual void elementwise_unary(

    ) = 0;

    virtual void matmul(

    ) = 0;

    virtual void reduce_sum(

    ) = 0;

    virtual ~Backend() = default;
};

Backend* get_backend(Device device){

}


}

