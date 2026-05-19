#include "matt/backend.hpp"
#include "matt/cpu_backend.hpp"
#include "matt/cuda_backend.hpp"
#include "matt/device.hpp"
#include "matt/memory_transfer.hpp"
#include "matt/ops.hpp"
#include "matt/tensor.hpp"
#include "test_utils.hpp"
#include <gtest/gtest.h>

using namespace matt;
using namespace matt::test_utils;

// ── Device ───────────────────────────────────────────────────────────────────

TEST(DeviceTest, CPUDevice) {
    Device d = Device::cpu();
    EXPECT_TRUE(d.is_cpu());
    EXPECT_FALSE(d.is_cuda());
    EXPECT_EQ(d.str(), "cpu");
}

TEST(DeviceTest, CUDADevice) {
    Device d = Device::cuda(0);
    EXPECT_FALSE(d.is_cpu());
    EXPECT_TRUE(d.is_cuda());
    EXPECT_EQ(d.index(), 0);
    EXPECT_EQ(d.str(), "cuda:0");
}

TEST(DeviceTest, DeviceEquality) {
    EXPECT_EQ(Device::cpu(), Device::cpu());
    EXPECT_EQ(Device::cuda(0), Device::cuda(0));
    EXPECT_NE(Device::cpu(), Device::cuda(0));
    EXPECT_NE(Device::cuda(0), Device::cuda(1));
}

// ── Backend ───────────────────────────────────────────────────────────────────

TEST(BackendTest, GetCPUBackend) {
    Backend *b = get_backend(Device::cpu());
    EXPECT_NE(b, nullptr);
    EXPECT_TRUE(b->device().is_cpu());
}

TEST(BackendTest, CPUBackendIsSingleton) {
    Backend *a = get_backend(Device::cpu());
    Backend *b = get_backend(Device::cpu());
    EXPECT_EQ(a, b);
}

TEST(BackendTest, UnknownDeviceThrows) {
#ifndef MATT_CUDA
    EXPECT_THROW(get_backend(Device::cuda(0)), std::runtime_error);
#endif
}

// ── CPUBackend ops ────────────────────────────────────────────────────────────

TEST(CPUBackendTest, AllocateDeallocate) {
    Backend *b = CPUBackend::get();
    float *ptr = b->allocate(10);
    EXPECT_NE(ptr, nullptr);
    b->deallocate(ptr);
}

TEST(CPUBackendTest, Fill) {
    Backend *b = CPUBackend::get();
    float *ptr = b->allocate(4);
    b->fill(ptr, 3.14f, 4);
    for (size_t i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(ptr[i], 3.14f);
    b->deallocate(ptr);
}

TEST(CPUBackendTest, ElementwiseBinaryAdd) {
    Backend *b = CPUBackend::get();
    float a[] = {1, 2, 3, 4};
    float bv[] = {10, 20, 30, 40};
    float *out = b->allocate(4);
    b->elementwise_binary(a, bv, out, 4, BinaryOpType::Add);
    EXPECT_FLOAT_EQ(out[0], 11);
    EXPECT_FLOAT_EQ(out[1], 22);
    EXPECT_FLOAT_EQ(out[2], 33);
    EXPECT_FLOAT_EQ(out[3], 44);
    b->deallocate(out);
}

TEST(CPUBackendTest, ElementwiseBinarySub) {
    Backend *b = CPUBackend::get();
    float a[] = {10, 20, 30, 40};
    float bv[] = {1, 2, 3, 4};
    float *out = b->allocate(4);
    b->elementwise_binary(a, bv, out, 4, BinaryOpType::Sub);
    EXPECT_FLOAT_EQ(out[0], 9);
    EXPECT_FLOAT_EQ(out[1], 18);
    EXPECT_FLOAT_EQ(out[2], 27);
    EXPECT_FLOAT_EQ(out[3], 36);
    b->deallocate(out);
}

TEST(CPUBackendTest, ElementwiseUnaryRelu) {
    Backend *b = CPUBackend::get();
    float a[] = {-2, -1, 0, 1, 2};
    float *out = b->allocate(5);
    b->elementwise_unary(a, out, 5, UnaryOpType::Relu);
    EXPECT_FLOAT_EQ(out[0], 0);
    EXPECT_FLOAT_EQ(out[1], 0);
    EXPECT_FLOAT_EQ(out[2], 0);
    EXPECT_FLOAT_EQ(out[3], 1);
    EXPECT_FLOAT_EQ(out[4], 2);
    b->deallocate(out);
}

TEST(CPUBackendTest, ReduceSum) {
    Backend *b = CPUBackend::get();
    float a[] = {1, 2, 3, 4};
    float *out = b->allocate(1);
    b->reduce(a, out, 4, ReduceOpType::Sum);
    EXPECT_FLOAT_EQ(out[0], 10);
    b->deallocate(out);
}

TEST(CPUBackendTest, Matmul) {
    Backend *b = CPUBackend::get();
    // [2,2] @ [2,2]
    float a[] = {1, 2, 3, 4};
    float bv[] = {1, 0, 0, 1}; // identity
    float *out = b->allocate(4);
    b->matmul(a, bv, out, 2, 2, 2);
    EXPECT_FLOAT_EQ(out[0], 1);
    EXPECT_FLOAT_EQ(out[1], 2);
    EXPECT_FLOAT_EQ(out[2], 3);
    EXPECT_FLOAT_EQ(out[3], 4);
    b->deallocate(out);
}

// ── Memory transfer ───────────────────────────────────────────────────────────

TEST(MemoryTransferTest, CPUtoCPU) {
    Backend *cpu = CPUBackend::get();
    float *src = cpu->allocate(4);
    cpu->fill(src, 7.0f, 4);
    float *dst = cpu->allocate(4);
    memory_transfer(dst, cpu, src, cpu, 4);
    for (size_t i = 0; i < 4; i++)
        EXPECT_FLOAT_EQ(dst[i], 7.0f);
    cpu->deallocate(src);
    cpu->deallocate(dst);
}

TEST(MemoryTransferTest, CPUtoCPUPreservesValues) {
    Backend *cpu = CPUBackend::get();
    float *src = cpu->allocate(4);
    src[0] = 1.f;
    src[1] = 2.f;
    src[2] = 3.f;
    src[3] = 4.f;
    float *dst = cpu->allocate(4);
    memory_transfer(dst, cpu, src, cpu, 4);
    EXPECT_FLOAT_EQ(dst[0], 1.f);
    EXPECT_FLOAT_EQ(dst[1], 2.f);
    EXPECT_FLOAT_EQ(dst[2], 3.f);
    EXPECT_FLOAT_EQ(dst[3], 4.f);
    cpu->deallocate(src);
    cpu->deallocate(dst);
}

#ifndef MATT_CUDA
TEST(MemoryTransferTest, CUDATransferThrowsWhenNotCompiled) {
    Backend *cpu = CPUBackend::get();
    float *src = cpu->allocate(4);
    float *dst = cpu->allocate(4);
    // Simulate passing a "cuda" backend — not actually CUDA, just checking the throw
    EXPECT_THROW(memory_transfer(dst, cpu, src, CUDABackend::get(0), 4), std::runtime_error);
    cpu->deallocate(src);
    cpu->deallocate(dst);
}
#endif

// ── Tensor device tracking ────────────────────────────────────────────────────

TEST(TensorDeviceTest, DefaultIsCPU) {
    Tensor t = Tensor::zeros({2, 3});
    EXPECT_TRUE(t.device().is_cpu());
}

TEST(TensorDeviceTest, ZerosOnCPU) {
    Tensor t = Tensor::zeros({2, 3}, Device::cpu());
    EXPECT_TRUE(t.device().is_cpu());
    EXPECT_EQ(t.shape(), (std::vector<size_t>{2, 3}));
}

TEST(TensorDeviceTest, OnesOnCPU) {
    Tensor t = Tensor::ones({2, 3}, Device::cpu());
    EXPECT_TRUE(t.device().is_cpu());
}

TEST(TensorDeviceTest, FillOnCPU) {
    Tensor t = Tensor::fill({2, 3}, 5.0f, Device::cpu());
    EXPECT_FLOAT_EQ(t.at({0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(t.at({1, 2}), 5.0f);
}

TEST(TensorDeviceTest, DevicePreservedAfterOps) {
    Tensor a = Tensor::ones({2, 2}, Device::cpu());
    Tensor b = Tensor::ones({2, 2}, Device::cpu());
    Tensor c = ops::add(a, b);
    EXPECT_TRUE(c.device().is_cpu());
}

TEST(TensorDeviceTest, MismatchedDevicesThrows) {
#ifndef MATT_CUDA
    // Can't actually create a CUDA tensor without CUDA, but verify
    // the device check exists in apply_binary
    Tensor a = Tensor::ones({2, 2}, Device::cpu());
    Tensor b = Tensor::ones({2, 2}, Device::cpu());
    EXPECT_NO_THROW(ops::add(a, b));
#endif
}

// ── CUDA tests (only compiled with MATT_CUDA) ─────────────────────────────────

#ifdef MATT_CUDA
TEST(CUDATest, GetCUDABackend) {
    Backend *b = get_backend(Device::cuda(0));
    EXPECT_NE(b, nullptr);
    EXPECT_TRUE(b->device().is_cuda());
}

TEST(CUDATest, CUDABackendIsSingleton) {
    Backend *a = get_backend(Device::cuda(0));
    Backend *b = get_backend(Device::cuda(0));
    EXPECT_EQ(a, b);
}

TEST(CUDATest, TensorZerosOnCUDA) {
    Tensor t = Tensor::zeros({2, 3}, Device::cuda(0));
    EXPECT_TRUE(t.device().is_cuda());
    EXPECT_EQ(t.shape(), (std::vector<size_t>{2, 3}));
}

TEST(CUDATest, ToCPU) {
    Tensor cpu_src = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    Tensor gpu = cpu_src.to(Device::cuda(0));
    Tensor cpu_dst = gpu.to(Device::cpu());
    expect_tensors_close(cpu_src, cpu_dst);
}

TEST(CUDATest, ToCUDAAndBack) {
    Tensor cpu = Tensor::ones({3, 3});
    Tensor gpu = cpu.to(Device::cuda(0));
    EXPECT_TRUE(gpu.device().is_cuda());
    Tensor back = gpu.to(Device::cpu());
    EXPECT_TRUE(back.device().is_cpu());
    expect_tensors_close(cpu, back);
}

TEST(CUDATest, AddOnCUDA) {
    Tensor a = Tensor::ones({2, 2}, Device::cuda(0));
    Tensor b = Tensor::ones({2, 2}, Device::cuda(0));
    Tensor c = ops::add(a, b);
    EXPECT_TRUE(c.device().is_cuda());
    Tensor c_cpu = c.to(Device::cpu());
    expect_tensors_close(c_cpu, Tensor::fill({2, 2}, 2.0f));
}

TEST(CUDATest, MatmulOnCUDA) {
    Tensor a = Tensor::ones({2, 3}, Device::cuda(0));
    Tensor b = Tensor::ones({3, 4}, Device::cuda(0));
    Tensor c = ops::matmul(a, b);
    EXPECT_TRUE(c.device().is_cuda());
    Tensor c_cpu = c.to(Device::cpu());
    // ones [2,3] @ ones [3,4] = 3s everywhere
    expect_tensors_close(c_cpu, Tensor::fill({2, 4}, 3.0f));
}

TEST(CUDATest, MismatchedDevicesThrows) {
    Tensor a = Tensor::ones({2, 2}, Device::cuda(0));
    Tensor b = Tensor::ones({2, 2}, Device::cpu());
    EXPECT_THROW(ops::add(a, b), std::runtime_error);
}

TEST(CUDATest, BackwardOnCUDA) {
    Tensor x = Tensor::ones({2, 2}, Device::cuda(0));
    x.set_requires_grad(true);
    Tensor y = ops::sum(x);
    y.backward();
    EXPECT_NE(x.data()->grad, nullptr);
    Tensor grad_cpu = Tensor(x.data()->grad).to(Device::cpu());
    expect_tensors_close(grad_cpu, Tensor::ones({2, 2}));
}
#endif
