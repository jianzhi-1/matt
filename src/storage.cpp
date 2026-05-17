#include "matt/device.hpp"
#include "matt/storage.hpp"

namespace matt {

Storage::Storage(size_t size, Device device): size_(size), device_(device){
    backend_ = get_backend(device_);
    data_ = backend_->allocate(size_);
}

Storage::~Storage(){
    if (data_) backend_->deallocate(data_);
}
}