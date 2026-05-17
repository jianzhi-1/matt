#include "matt/nn/weight_initializer/weight_initializer.hpp"
#include "matt/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
 
using namespace matt;
using namespace matt::nn::weight_initializer;
 
// ── Helpers ───────────────────────────────────────────────────────────────────
 
static std::vector<float> flatten(const Tensor &t) {
    size_t n = t.numel();
    std::vector<float> out(n);
    if (t.ndim() == 1) {
        for (size_t i = 0; i < n; i++)
            out[i] = t.at({i});
    } else if (t.ndim() == 2) {
        size_t rows = t.shape()[0], cols = t.shape()[1];
        for (size_t r = 0; r < rows; r++)
            for (size_t c = 0; c < cols; c++)
                out[r * cols + c] = t.at({r, c});
    }
    return out;
}
 
static float mean(const std::vector<float> &v) {
    return std::accumulate(v.begin(), v.end(), 0.f) / v.size();
}
 
static float stddev(const std::vector<float> &v) {
    float m = mean(v);
    float var = 0;
    for (float x : v)
        var += (x - m) * (x - m);
    return std::sqrt(var / v.size());
}
 
// ── Zeros ─────────────────────────────────────────────────────────────────────
 
TEST(ZerosInitializer, AllValuesAreZero) {
    auto data = flatten(Zeros().initialize({8, 16}));
    for (float v : data)
        EXPECT_FLOAT_EQ(v, 0.f);
}
 
TEST(ZerosInitializer, ShapePreserved) {
    std::vector<size_t> shape = {4, 6};
    EXPECT_EQ(Zeros().initialize(shape).shape(), shape);
}
 
TEST(ZerosInitializer, LargeShape) {
    auto data = flatten(Zeros().initialize({64, 128}));
    EXPECT_EQ(data.size(), 64u * 128u);
    EXPECT_TRUE(std::all_of(data.begin(), data.end(), [](float v) { return v == 0.f; }));
}
 
// ── Uniform ───────────────────────────────────────────────────────────────────
 
TEST(UniformInitializer, DeterministicSameSeed) {
    auto a = flatten(Uniform(-1, 1, 42).initialize({8, 16}));
    auto b = flatten(Uniform(-1, 1, 42).initialize({8, 16}));
    EXPECT_EQ(a, b);
}
 
TEST(UniformInitializer, DifferentSeedsDiffer) {
    auto a = flatten(Uniform(-1, 1, 42).initialize({8, 16}));
    auto b = flatten(Uniform(-1, 1, 99).initialize({8, 16}));
    EXPECT_NE(a, b);
}
 
TEST(UniformInitializer, ValuesWithinRange) {
    float lo = -3.f, hi = 2.f;
    auto data = flatten(Uniform(lo, hi, 7).initialize({64, 128}));
    for (float v : data) {
        EXPECT_GE(v, lo);
        EXPECT_LE(v, hi);
    }
}
 
TEST(UniformInitializer, MeanCloseToMidpoint) {
    // E[U(a,b)] = (a+b)/2
    float lo = -2.f, hi = 4.f;
    auto data = flatten(Uniform(lo, hi, 13).initialize({64, 128}));
    float expected = (lo + hi) / 2.f;
    EXPECT_NEAR(mean(data), expected, 0.1f);
}
 
TEST(UniformInitializer, StdCloseToTheoretical) {
    // Std[U(a,b)] = (b-a) / sqrt(12)
    float lo = -1.f, hi = 1.f;
    auto data = flatten(Uniform(lo, hi, 17).initialize({64, 128}));
    float expected_std = (hi - lo) / std::sqrt(12.f);
    EXPECT_NEAR(stddev(data), expected_std, 0.05f);
}
 
TEST(UniformInitializer, SymmetricRangeBalancedSign) {
    // U(-b, b): roughly half positive, half negative
    auto data = flatten(Uniform(-2, 2, 31).initialize({64, 128}));
    int pos = std::count_if(data.begin(), data.end(), [](float v) { return v > 0; });
    float frac = static_cast<float>(pos) / data.size();
    EXPECT_GT(frac, 0.45f);
    EXPECT_LT(frac, 0.55f);
}
 
TEST(UniformInitializer, ShapePreserved) {
    std::vector<size_t> shape = {5, 7};
    EXPECT_EQ(Uniform(-1, 1, 0).initialize(shape).shape(), shape);
}
 
// ── Normal ────────────────────────────────────────────────────────────────────
 
TEST(NormalInitializer, DeterministicSameSeed) {
    auto a = flatten(Normal(0, 1, 42).initialize({8, 16}));
    auto b = flatten(Normal(0, 1, 42).initialize({8, 16}));
    EXPECT_EQ(a, b);
}
 
TEST(NormalInitializer, DifferentSeedsDiffer) {
    auto a = flatten(Normal(0, 1, 42).initialize({8, 16}));
    auto b = flatten(Normal(0, 1, 99).initialize({8, 16}));
    EXPECT_NE(a, b);
}
 
TEST(NormalInitializer, MeanCloseToMu) {
    float mu = 2.5f, sigma = 1.f;
    auto data = flatten(Normal(mu, sigma, 7).initialize({64, 128}));
    EXPECT_NEAR(mean(data), mu, 0.1f);
}
 
TEST(NormalInitializer, StdCloseToSigma) {
    float mu = 0.f, sigma = 0.5f;
    auto data = flatten(Normal(mu, sigma, 11).initialize({64, 128}));
    EXPECT_NEAR(stddev(data), sigma, 0.05f);
}
 
TEST(NormalInitializer, LargerSigmaIncreasesSpread) {
    auto narrow = flatten(Normal(0, 0.1f, 3).initialize({64, 128}));
    auto wide   = flatten(Normal(0, 2.0f, 3).initialize({64, 128}));
    EXPECT_GT(stddev(wide), stddev(narrow) * 5.f);
}
 
TEST(NormalInitializer, NonzeroMuShiftsDistribution) {
    auto centered = flatten(Normal(0.f,  1.f, 5).initialize({8, 16}));
    auto shifted  = flatten(Normal(10.f, 1.f, 5).initialize({8, 16}));
    EXPECT_GT(mean(shifted), mean(centered) + 5.f);
}
 
TEST(NormalInitializer, ShapePreserved) {
    std::vector<size_t> shape = {3, 9};
    EXPECT_EQ(Normal(0, 1, 0).initialize(shape).shape(), shape);
}
 
// ── KaimingUniform ────────────────────────────────────────────────────────────
 
TEST(KaimingUniformInitializer, DeterministicSameSeed) {
    auto a = flatten(KaimingUniform(42).initialize({8, 16}));
    auto b = flatten(KaimingUniform(42).initialize({8, 16}));
    EXPECT_EQ(a, b);
}
 
TEST(KaimingUniformInitializer, DifferentSeedsDiffer) {
    auto a = flatten(KaimingUniform(42).initialize({8, 16}));
    auto b = flatten(KaimingUniform(99).initialize({8, 16}));
    EXPECT_NE(a, b);
}
 
TEST(KaimingUniformInitializer, ValuesWithinKaimingBound) {
    // bound = sqrt(1 / fan_in), fan_in = shape[1]
    std::vector<size_t> shape = {32, 64};
    float bound = std::sqrt(1.f / shape[1]);
    auto data = flatten(KaimingUniform(7).initialize(shape));
    for (float v : data) {
        EXPECT_GE(v, -bound - 1e-6f);
        EXPECT_LE(v,  bound + 1e-6f);
    }
}
 
TEST(KaimingUniformInitializer, BoundScalesWithFanIn) {
    // Larger fan_in → smaller bound → smaller magnitude weights
    auto small_fan = flatten(KaimingUniform(42).initialize({8, 4}));    // bound=0.5
    auto large_fan = flatten(KaimingUniform(42).initialize({8, 256}));  // bound=0.0625
    float max_small = *std::max_element(small_fan.begin(), small_fan.end(),
                                        [](float a, float b){ return std::abs(a) < std::abs(b); });
    float max_large = *std::max_element(large_fan.begin(), large_fan.end(),
                                        [](float a, float b){ return std::abs(a) < std::abs(b); });
    EXPECT_GT(std::abs(max_small), std::abs(max_large));
}
 
TEST(KaimingUniformInitializer, StdMatchesTheoreticalUniform) {
    // U(-b, b) has std = b / sqrt(3)
    std::vector<size_t> shape = {64, 128};  // fan_in = 128
    float bound = std::sqrt(1.f / shape[1]);
    float expected_std = bound / std::sqrt(3.f);
    auto data = flatten(KaimingUniform(13).initialize(shape));
    EXPECT_NEAR(stddev(data), expected_std, 0.01f);
}
 
TEST(KaimingUniformInitializer, MeanNearZero) {
    auto data = flatten(KaimingUniform(17).initialize({64, 128}));
    EXPECT_NEAR(mean(data), 0.f, 0.02f);
}
 
TEST(KaimingUniformInitializer, Rejects1DShape) {
    EXPECT_THROW(KaimingUniform(42).initialize({16}), std::runtime_error);
}
 
TEST(KaimingUniformInitializer, ShapePreserved) {
    std::vector<size_t> shape = {10, 20};
    EXPECT_EQ(KaimingUniform(0).initialize(shape).shape(), shape);
}
 
// ── Cross-initializer ─────────────────────────────────────────────────────────
 
TEST(CrossInitializer, SameSeedDifferentTypesDiffer) {
    auto u = flatten(Uniform(-1, 1, 42).initialize({8, 16}));
    auto n = flatten(Normal(0, 1,  42).initialize({8, 16}));
    EXPECT_NE(u, n);
}
 
TEST(CrossInitializer, ZerosDiffersFromRandom) {
    auto z = flatten(Zeros().initialize({8, 16}));
    auto u = flatten(Uniform(-1, 1, 1).initialize({8, 16}));
    EXPECT_NE(z, u);
}
 
TEST(CrossInitializer, AllInitializersHandleEdgeCaseShapes) {
    // [1, N] and [N, 1] are valid 2D shapes all initializers must handle
    for (auto shape : std::vector<std::vector<size_t>>{{1, 64}, {64, 1}, {1, 1}}) {
        EXPECT_EQ(Zeros().initialize(shape).shape(),          shape) << "Zeros";
        EXPECT_EQ(Uniform(-1, 1, 0).initialize(shape).shape(), shape) << "Uniform";
        EXPECT_EQ(Normal(0, 1, 0).initialize(shape).shape(),   shape) << "Normal";
        EXPECT_EQ(KaimingUniform(0).initialize(shape).shape(), shape) << "KaimingUniform";
    }
}