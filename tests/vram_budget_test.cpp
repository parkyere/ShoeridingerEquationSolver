// RED: pick the resident eigenstate-atlas precision that fits GPU VRAM.
//
// The n<=6 manifold keeps 91 complex-fp32 state buffers resident (256^3 ->
// 134,217,728 B each, ~12.2 GB). That overflows an 8 GB card: the driver pages
// the surplus into system RAM (WDDM shared memory) and the frame rate
// collapses. At startup the shell queries free VRAM and, when the fp32 atlas
// would not fit, drops the buffers to fp16 (half the footprint, ~6.1 GB).
//
// The DECISION is pure integer arithmetic (this function); the GL query that
// feeds it is the untested Humble-Object shell (app/src/main.cpp). Oracles:
//   - ample VRAM              -> Fp32, fits;
//   - fp32 overflows, fp16 ok -> Fp16, fits;
//   - even fp16 overflows     -> Fp16, but out_fits=false (caller warns);
//   - unmeasurable budget     -> Fp32 (never silently degrade fidelity).

#include <core/vram_budget.hpp>

#include <gtest/gtest.h>

#include <cstdint>

namespace {

using ses::choose_state_precision;
using ses::GpuPrecision;

// The real workload the shell will pass: the n<=6 manifold at 256^3.
constexpr int kNumStates = 91;
constexpr std::int64_t kBytesPerStateFp32 = 134217728;  // 256^3 * 2 floats * 4 B
constexpr std::int64_t kHeadroom = 512LL * 1024 * 1024;  // textures/fbo/working
constexpr std::int64_t kGiB = 1024LL * 1024 * 1024;

TEST(ChooseStatePrecision, AmpleVramKeepsFp32) {
    bool fits = false;
    EXPECT_EQ(choose_state_precision(16 * kGiB, kNumStates, kBytesPerStateFp32,
                                     kHeadroom, &fits),
              GpuPrecision::Fp32);
    EXPECT_TRUE(fits);
}

TEST(ChooseStatePrecision, Fp32OverflowsButFp16Fits) {
    // 8 GB card: need32 ~= 12.2 GB > budget, need16 ~= 6.1 GB <= budget.
    bool fits = false;
    EXPECT_EQ(choose_state_precision(8 * kGiB, kNumStates, kBytesPerStateFp32,
                                     kHeadroom, &fits),
              GpuPrecision::Fp16);
    EXPECT_TRUE(fits);
}

TEST(ChooseStatePrecision, EvenFp16OverflowsStillPicksFp16AndFlags) {
    // 4 GB card: even the 6.1 GB fp16 atlas will not fit -- pick the smallest
    // footprint anyway and signal the caller to warn (it may reduce the box).
    bool fits = true;
    EXPECT_EQ(choose_state_precision(4 * kGiB, kNumStates, kBytesPerStateFp32,
                                     kHeadroom, &fits),
              GpuPrecision::Fp16);
    EXPECT_FALSE(fits);
}

TEST(ChooseStatePrecision, UnmeasurableBudgetDefaultsToFp32) {
    // The GL query returns kVramUnknown when neither NVX nor ATI meminfo is
    // present: do NOT silently downgrade physics fidelity on a budget we
    // cannot measure.
    bool fits = false;
    EXPECT_EQ(choose_state_precision(ses::kVramUnknown, kNumStates,
                                     kBytesPerStateFp32, kHeadroom, &fits),
              GpuPrecision::Fp32);
    EXPECT_TRUE(fits);
    // Any other negative (nonsense) free value is treated the same way.
    EXPECT_EQ(choose_state_precision(-5, kNumStates, kBytesPerStateFp32,
                                     kHeadroom, nullptr),
              GpuPrecision::Fp32);
}

TEST(ChooseStatePrecision, OutFitsPointerIsOptional) {
    // The overload must be callable without the fits flag.
    EXPECT_EQ(choose_state_precision(16 * kGiB, kNumStates, kBytesPerStateFp32,
                                     kHeadroom),
              GpuPrecision::Fp32);
}

}  // namespace
