// sesolver_vkcheck: framework-free Vulkan verification harness (M5).
//
// The raw-Vulkan analog of sesolver_qrhicheck and the seed of the eventual
// framework-free compute core: NO Qt anywhere in this binary. volk loads the
// loader, VMA allocates, the kernels are the SAME Vulkan-GLSL sources the
// QRhi engine bakes with qsb -- here compiled offline by glslangValidator to
// plain SPIR-V and embedded as C arrays (tools/cmake/bin2h.cmake). Every
// check reproduces its qrhicheck twin: same oracle data, same tolerance, so
// the two harnesses cross-check each other kernel by kernel as the M5 port
// proceeds. Stage 1 covers the full individual-kernel set: element-wise ops,
// the four reductions, the fp16 codec (a compute-to-compute hazard), the
// line FFT at N=64/256, and the 3-axis fft3 orchestration (three dispatches
// aliasing one buffer -- the barrier pattern at the heart of the Strang
// step, here hand-authored and policed by the validation layer).
//
// Validation layers: set SES_VK_VALIDATION=1 (and have the layer discoverable
// via VK_ADD_LAYER_PATH or the SDK registry). Any validation ERROR fails the
// run even if the numbers pass.
//
// Exit codes: 0 = all checks PASS, 1 = FAIL, 77 = SKIP (no Vulkan runtime /
// device on this machine; ctest maps 77 to SKIP).

#define VMA_IMPLEMENTATION
#include "vk_engine.hpp"

#include <core/complex.hpp>
#include <core/fft.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/drive.hpp>
#include <core/harmonics.hpp>
#include <core/imaginary_time.hpp>
#include <core/magnetic.hpp>
#include <core/radial.hpp>
#include <core/potential.hpp>
#include <core/projection.hpp>
#include <core/propagator.hpp>
#include <core/rotation.hpp>
#include <core/vec.hpp>
#include <core/wavepacket.hpp>

#include <phase_multiply_spv.h>
#include <conj_scale_spv.h>
#include <scale_spv.h>
#include <norm_peak_spv.h>
#include <inner_product_spv.h>
#include <dipole_kick_spv.h>
#include <mean_force_spv.h>
#include <dipole_spv.h>
#include <pack_half_spv.h>
#include <unpack_half_spv.h>
#include <shear_spv.h>
#include <axpy_spv.h>
#include <copy_state_spv.h>
#include <synth_spv.h>
#include <project_deposit_spv.h>
#include <fft_line8_spv.h>
#include <fft_line64_spv.h>
#include <fft_line256_spv.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr VkDescriptorType kStorage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
constexpr VkDescriptorType kUniform = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

// Complex<double> -> interleaved rg32f, byte-identical to the qrhicheck /
// engine upload format so all harnesses see the same fp32 inputs.
std::vector<float> to_rg32f(const std::vector<ses::Complex<double>>& src) {
    std::vector<float> out(2 * src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        out[2 * i] = static_cast<float>(src[i].real());
        out[2 * i + 1] = static_cast<float>(src[i].imag());
    }
    return out;
}

// Tears down a check's Vulkan objects at scope exit (declare LAST in the
// check so its destructor runs first, while the registered objects are
// still alive). Keeps every early return leak-free.
class Scope {
public:
    explicit Scope(ses_vk::DeviceContext& ctx) : ctx_(ctx) {}
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    ~Scope() {
        shot.destroy(ctx_);
        arena.destroy(ctx_);
        for (auto it = kernels.rbegin(); it != kernels.rend(); ++it) {
            (*it)->destroy(ctx_);
        }
        for (auto it = buffers.rbegin(); it != buffers.rend(); ++it) {
            ctx_.destroy_buffer(*it);
        }
    }
    ses_vk::OneShot shot;
    ses_vk::DescriptorArena arena;
    std::vector<ses_vk::Kernel*> kernels;
    std::vector<ses_vk::Buffer*> buffers;

private:
    ses_vk::DeviceContext& ctx_;
};

// staging[src_off .. src_off+bytes) -> dst[0..bytes)
struct UploadCopy {
    const ses_vk::Buffer* dst;
    VkDeviceSize src_off;
    VkDeviceSize bytes;
};

void record_uploads(VkCommandBuffer cb, const ses_vk::Buffer& staging,
                    std::initializer_list<UploadCopy> copies) {
    for (const UploadCopy& c : copies) {
        const VkBufferCopy r{c.src_off, 0, c.bytes};
        vkCmdCopyBuffer(cb, staging.buf, c.dst->buf, 1, &r);
    }
    ses_vk::barrier_transfer_to_compute(cb);
}

// device src[0..bytes) -> staging[0..bytes), fenced for host reads.
void record_readback(VkCommandBuffer cb, const ses_vk::Buffer& src,
                     const ses_vk::Buffer& staging, VkDeviceSize bytes) {
    ses_vk::barrier_compute_to_transfer(cb);
    const VkBufferCopy r{0, 0, bytes};
    vkCmdCopyBuffer(cb, src.buf, staging.buf, 1, &r);
    ses_vk::barrier_transfer_to_host(cb);
}

void flush(ses_vk::DeviceContext& ctx, const ses_vk::Buffer& b) {
    vmaFlushAllocation(ctx.allocator, b.alloc, 0, VK_WHOLE_SIZE);
}

void invalidate(ses_vk::DeviceContext& ctx, const ses_vk::Buffer& b) {
    vmaInvalidateAllocation(ctx.allocator, b.alloc, 0, VK_WHOLE_SIZE);
}

// psi <- psi (complex-*) phase: same data and tolerance (1e-5) as
// qrhicheck's check_phase_multiply.
bool check_phase_multiply(ses_vk::DeviceContext& ctx) {
    const std::size_t n = 4096;
    std::vector<ses::Complex<double>> psi_d(n);
    std::vector<ses::Complex<double>> phase_d(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        psi_d[i] = ses::Complex<double>{std::sin(0.37 * x) + 0.2,
                                        std::cos(1.13 * x) - 0.1};
        phase_d[i] = ses::Complex<double>{std::cos(2.9 * x), std::sin(2.9 * x)};
    }
    const std::vector<float> psi_f = to_rg32f(psi_d);
    const std::vector<float> phase_f = to_rg32f(phase_d);
    const VkDeviceSize bytes = psi_f.size() * sizeof(float);

    struct alignas(16) Params {
        std::uint32_t n, pad0, pad1, pad2;
    };
    ses_vk::Kernel k;
    ses_vk::Buffer psi{}, phase{}, staging{}, ubo{};
    Scope s(ctx);
    s.kernels = {&k};
    s.buffers = {&psi, &phase, &staging, &ubo};

    if (!k.create(ctx, k_phase_multiply_spv, k_phase_multiply_spv_size,
                  {{0, kStorage}, {1, kStorage}, {2, kUniform}})) {
        return false;
    }
    if (!ctx.create_device_buffer(bytes, &psi) ||
        !ctx.create_device_buffer(bytes, &phase) ||
        !ctx.create_host_buffer(2 * bytes,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                &staging) ||
        !ctx.create_host_buffer(sizeof(Params),
                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &ubo)) {
        return false;
    }
    std::memcpy(staging.mapped, psi_f.data(), bytes);
    std::memcpy(static_cast<char*>(staging.mapped) + bytes, phase_f.data(),
                bytes);
    const Params params{static_cast<std::uint32_t>(n), 0, 0, 0};
    std::memcpy(ubo.mapped, &params, sizeof(params));
    flush(ctx, staging);
    flush(ctx, ubo);

    if (!s.arena.create(ctx, 1, 2, 1)) return false;
    VkDescriptorSet set = s.arena.allocate(ctx, k.set_layout());
    if (set == VK_NULL_HANDLE) return false;
    s.arena.write_buffer(ctx, set, 0, kStorage, psi.buf);
    s.arena.write_buffer(ctx, set, 1, kStorage, phase.buf);
    s.arena.write_buffer(ctx, set, 2, kUniform, ubo.buf, sizeof(Params));

    if (!s.shot.begin(ctx)) return false;
    record_uploads(s.shot.cb(), staging,
                   {{&psi, 0, bytes}, {&phase, bytes, bytes}});
    k.bind(s.shot.cb(), set);
    vkCmdDispatch(s.shot.cb(), static_cast<std::uint32_t>((n + 255) / 256), 1,
                  1);
    record_readback(s.shot.cb(), psi, staging, bytes);
    if (!s.shot.submit_and_wait(ctx)) return false;
    invalidate(ctx, staging);

    const float* out = static_cast<const float*>(staging.mapped);
    double max_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const ses::Complex<double> expected = psi_d[i] * phase_d[i];
        max_err = std::max(max_err, std::abs(out[2 * i] - expected.real()));
        max_err = std::max(max_err, std::abs(out[2 * i + 1] - expected.imag()));
    }
    const bool pass = max_err < 1e-5;
    std::printf(
        "phase-multiply kernel (raw Vulkan): max |gpu - cpu| = %.3e  [%s]\n",
        max_err, pass ? "PASS" : "FAIL");
    return pass;
}

// data <- s * data, same data/tolerance as qrhicheck's check_scale.
bool check_scale(ses_vk::DeviceContext& ctx) {
    const std::size_t n = 4096;
    const float sc = 0.5f;
    std::vector<ses::Complex<double>> d0(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        d0[i] = ses::Complex<double>{std::sin(0.7 * x) - 0.3,
                                     std::cos(0.21 * x) + 0.4};
    }
    const std::vector<float> in = to_rg32f(d0);
    const VkDeviceSize bytes = in.size() * sizeof(float);

    struct alignas(16) Params {
        std::uint32_t n;
        float scale;
        float pad0, pad1;
    };
    ses_vk::Kernel k;
    ses_vk::Buffer data{}, staging{}, ubo{};
    Scope s(ctx);
    s.kernels = {&k};
    s.buffers = {&data, &staging, &ubo};

    if (!k.create(ctx, k_scale_spv, k_scale_spv_size,
                  {{0, kStorage}, {1, kUniform}})) {
        return false;
    }
    if (!ctx.create_device_buffer(bytes, &data) ||
        !ctx.create_host_buffer(bytes,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                &staging) ||
        !ctx.create_host_buffer(sizeof(Params),
                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &ubo)) {
        return false;
    }
    std::memcpy(staging.mapped, in.data(), bytes);
    const Params params{static_cast<std::uint32_t>(n), sc, 0.0f, 0.0f};
    std::memcpy(ubo.mapped, &params, sizeof(params));
    flush(ctx, staging);
    flush(ctx, ubo);

    if (!s.arena.create(ctx, 1, 1, 1)) return false;
    VkDescriptorSet set = s.arena.allocate(ctx, k.set_layout());
    if (set == VK_NULL_HANDLE) return false;
    s.arena.write_buffer(ctx, set, 0, kStorage, data.buf);
    s.arena.write_buffer(ctx, set, 1, kUniform, ubo.buf, sizeof(Params));

    if (!s.shot.begin(ctx)) return false;
    record_uploads(s.shot.cb(), staging, {{&data, 0, bytes}});
    k.bind(s.shot.cb(), set);
    vkCmdDispatch(s.shot.cb(), static_cast<std::uint32_t>((n + 255) / 256), 1,
                  1);
    record_readback(s.shot.cb(), data, staging, bytes);
    if (!s.shot.submit_and_wait(ctx)) return false;
    invalidate(ctx, staging);

    const float* out = static_cast<const float*>(staging.mapped);
    double max_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        max_err = std::max(
            max_err, std::abs(out[2 * i] - static_cast<double>(sc) * d0[i].real()));
        max_err = std::max(max_err, std::abs(out[2 * i + 1] -
                                             static_cast<double>(sc) *
                                                 d0[i].imag()));
    }
    const bool pass = max_err < 1e-5;
    std::printf("scale kernel (raw Vulkan): max |gpu - cpu| = %.3e  [%s]\n",
                max_err, pass ? "PASS" : "FAIL");
    return pass;
}

// Fixed-order tree reduction of |psi|^2 into per-workgroup (sum, max)
// partials, finished on the host. Same data/tolerances as qrhicheck.
bool check_norm_peak(ses_vk::DeviceContext& ctx) {
    const std::size_t n = 20000;
    std::vector<ses::Complex<double>> psi_d(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        psi_d[i] = ses::Complex<double>{0.5 * std::sin(0.013 * x),
                                        0.3 * std::cos(0.017 * x) + 0.1};
    }
    double cpu_sum = 0.0;
    double cpu_peak = 0.0;
    for (const ses::Complex<double>& z : psi_d) {
        const double d = z.real() * z.real() + z.imag() * z.imag();
        cpu_sum += d;
        cpu_peak = std::max(cpu_peak, d);
    }
    const std::vector<float> in = to_rg32f(psi_d);
    const VkDeviceSize psi_bytes = in.size() * sizeof(float);
    const int groups = 256;
    const VkDeviceSize part_bytes = 2u * groups * sizeof(float);

    struct alignas(16) Params {
        std::uint32_t n, pad0, pad1, pad2;
    };
    ses_vk::Kernel k;
    ses_vk::Buffer psi{}, partials{}, staging{}, ubo{};
    Scope s(ctx);
    s.kernels = {&k};
    s.buffers = {&psi, &partials, &staging, &ubo};

    if (!k.create(ctx, k_norm_peak_spv, k_norm_peak_spv_size,
                  {{0, kStorage}, {1, kUniform}, {2, kStorage}})) {
        return false;
    }
    if (!ctx.create_device_buffer(psi_bytes, &psi) ||
        !ctx.create_device_buffer(part_bytes, &partials) ||
        !ctx.create_host_buffer(psi_bytes,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                &staging) ||
        !ctx.create_host_buffer(sizeof(Params),
                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &ubo)) {
        return false;
    }
    std::memcpy(staging.mapped, in.data(), psi_bytes);
    const Params params{static_cast<std::uint32_t>(n), 0, 0, 0};
    std::memcpy(ubo.mapped, &params, sizeof(params));
    flush(ctx, staging);
    flush(ctx, ubo);

    if (!s.arena.create(ctx, 1, 2, 1)) return false;
    VkDescriptorSet set = s.arena.allocate(ctx, k.set_layout());
    if (set == VK_NULL_HANDLE) return false;
    s.arena.write_buffer(ctx, set, 0, kStorage, psi.buf);
    s.arena.write_buffer(ctx, set, 1, kUniform, ubo.buf, sizeof(Params));
    s.arena.write_buffer(ctx, set, 2, kStorage, partials.buf);

    if (!s.shot.begin(ctx)) return false;
    record_uploads(s.shot.cb(), staging, {{&psi, 0, psi_bytes}});
    k.bind(s.shot.cb(), set);
    vkCmdDispatch(s.shot.cb(), groups, 1, 1);
    record_readback(s.shot.cb(), partials, staging, part_bytes);
    if (!s.shot.submit_and_wait(ctx)) return false;
    invalidate(ctx, staging);

    const float* p = static_cast<const float*>(staging.mapped);
    double gpu_sum = 0.0;
    double gpu_peak = 0.0;
    for (int g = 0; g < groups; ++g) {
        gpu_sum += p[2 * g];
        gpu_peak = std::max(gpu_peak, static_cast<double>(p[2 * g + 1]));
    }
    const double sum_rel = std::abs(gpu_sum - cpu_sum) / cpu_sum;
    const double peak_rel = std::abs(gpu_peak - cpu_peak) / cpu_peak;
    const bool pass = sum_rel < 1e-5 && peak_rel < 1e-6;
    std::printf(
        "norm/peak reduce (raw Vulkan): rel err sum %.3e, peak %.3e  [%s]\n",
        sum_rel, peak_rel, pass ? "PASS" : "FAIL");
    return pass;
}

// <phi|psi> complex two-input reduction. Same data/tolerance as qrhicheck.
bool check_inner_product(ses_vk::DeviceContext& ctx) {
    const std::size_t n = 20000;
    std::vector<ses::Complex<double>> psi_d(n);
    std::vector<ses::Complex<double>> phi_d(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        psi_d[i] =
            ses::Complex<double>{std::sin(0.019 * x) + 0.1, std::cos(0.023 * x)};
        phi_d[i] =
            ses::Complex<double>{std::cos(0.011 * x), std::sin(0.029 * x) - 0.2};
    }
    double cpu_re = 0.0;
    double cpu_im = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double ar = phi_d[i].real();
        const double ai = phi_d[i].imag();
        const double br = psi_d[i].real();
        const double bi = psi_d[i].imag();
        cpu_re += ar * br + ai * bi;
        cpu_im += ar * bi - ai * br;
    }
    const std::vector<float> psi_f = to_rg32f(psi_d);
    const std::vector<float> phi_f = to_rg32f(phi_d);
    const VkDeviceSize bytes = psi_f.size() * sizeof(float);
    const int groups = 256;
    const VkDeviceSize part_bytes = 2u * groups * sizeof(float);

    struct alignas(16) Params {
        std::uint32_t n, pad0, pad1, pad2;
    };
    ses_vk::Kernel k;
    ses_vk::Buffer psi{}, phi{}, partials{}, staging{}, ubo{};
    Scope s(ctx);
    s.kernels = {&k};
    s.buffers = {&psi, &phi, &partials, &staging, &ubo};

    if (!k.create(ctx, k_inner_product_spv, k_inner_product_spv_size,
                  {{0, kStorage}, {1, kUniform}, {2, kStorage}, {3, kStorage}})) {
        return false;
    }
    if (!ctx.create_device_buffer(bytes, &psi) ||
        !ctx.create_device_buffer(bytes, &phi) ||
        !ctx.create_device_buffer(part_bytes, &partials) ||
        !ctx.create_host_buffer(2 * bytes,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                &staging) ||
        !ctx.create_host_buffer(sizeof(Params),
                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &ubo)) {
        return false;
    }
    std::memcpy(staging.mapped, psi_f.data(), bytes);
    std::memcpy(static_cast<char*>(staging.mapped) + bytes, phi_f.data(), bytes);
    const Params params{static_cast<std::uint32_t>(n), 0, 0, 0};
    std::memcpy(ubo.mapped, &params, sizeof(params));
    flush(ctx, staging);
    flush(ctx, ubo);

    if (!s.arena.create(ctx, 1, 3, 1)) return false;
    VkDescriptorSet set = s.arena.allocate(ctx, k.set_layout());
    if (set == VK_NULL_HANDLE) return false;
    s.arena.write_buffer(ctx, set, 0, kStorage, psi.buf);
    s.arena.write_buffer(ctx, set, 1, kUniform, ubo.buf, sizeof(Params));
    s.arena.write_buffer(ctx, set, 2, kStorage, partials.buf);
    s.arena.write_buffer(ctx, set, 3, kStorage, phi.buf);

    if (!s.shot.begin(ctx)) return false;
    record_uploads(s.shot.cb(), staging,
                   {{&psi, 0, bytes}, {&phi, bytes, bytes}});
    k.bind(s.shot.cb(), set);
    vkCmdDispatch(s.shot.cb(), groups, 1, 1);
    record_readback(s.shot.cb(), partials, staging, part_bytes);
    if (!s.shot.submit_and_wait(ctx)) return false;
    invalidate(ctx, staging);

    const float* p = static_cast<const float*>(staging.mapped);
    double gpu_re = 0.0;
    double gpu_im = 0.0;
    for (int g = 0; g < groups; ++g) {
        gpu_re += p[2 * g];
        gpu_im += p[2 * g + 1];
    }
    const double mag = std::sqrt(cpu_re * cpu_re + cpu_im * cpu_im);
    const double err = std::sqrt((gpu_re - cpu_re) * (gpu_re - cpu_re) +
                                 (gpu_im - cpu_im) * (gpu_im - cpu_im));
    const double rel = (mag > 0.0) ? err / mag : err;
    const bool pass = rel < 1e-5;
    std::printf("inner-product kernel (raw Vulkan): rel err = %.3e  [%s]\n",
                rel, pass ? "PASS" : "FAIL");
    return pass;
}

// One dipole half-kick psi <- exp(-i theta axis.r) psi per grid cell.
// Same data/tolerance as qrhicheck (vec4-padded vec3 geometry UBO).
bool check_dipole_kick(ses_vk::DeviceContext& ctx) {
    const int nx = 8;
    const int ny = 8;
    const int nz = 8;
    const std::size_t n = static_cast<std::size_t>(nx * ny * nz);
    const double box_min[3] = {-4.0, -4.0, -4.0};
    const double cell_h[3] = {1.0, 1.1, 0.9};
    const double axis[3] = {0.3, 0.6, -0.2};
    const double theta = 0.15;

    std::vector<ses::Complex<double>> psi_d(n);
    std::vector<ses::Complex<double>> cpu(n);
    for (std::size_t idx = 0; idx < n; ++idx) {
        const double x = static_cast<double>(idx);
        psi_d[idx] = ses::Complex<double>{std::sin(0.13 * x) + 0.2,
                                          std::cos(0.09 * x) - 0.1};
        const int i = static_cast<int>(idx) % nx;
        const int j = (static_cast<int>(idx) / nx) % ny;
        const int kk = static_cast<int>(idx) / (nx * ny);
        const double rx = box_min[0] + i * cell_h[0];
        const double ry = box_min[1] + j * cell_h[1];
        const double rz = box_min[2] + kk * cell_h[2];
        const double ang = -theta * (axis[0] * rx + axis[1] * ry + axis[2] * rz);
        const double wr = std::cos(ang);
        const double wi = std::sin(ang);
        const double ar = psi_d[idx].real();
        const double ai = psi_d[idx].imag();
        cpu[idx] = ses::Complex<double>{ar * wr - ai * wi, ar * wi + ai * wr};
    }
    const std::vector<float> in = to_rg32f(psi_d);
    const VkDeviceSize bytes = in.size() * sizeof(float);

    struct alignas(16) Params {
        std::uint32_t n;
        std::int32_t nx;
        std::int32_t ny;
        float theta;
        float box_min[4];
        float cell_h[4];
        float axis[4];
    };
    Params params{};
    params.n = static_cast<std::uint32_t>(n);
    params.nx = nx;
    params.ny = ny;
    params.theta = static_cast<float>(theta);
    for (int c = 0; c < 3; ++c) {
        params.box_min[c] = static_cast<float>(box_min[c]);
        params.cell_h[c] = static_cast<float>(cell_h[c]);
        params.axis[c] = static_cast<float>(axis[c]);
    }

    ses_vk::Kernel k;
    ses_vk::Buffer psi{}, staging{}, ubo{};
    Scope s(ctx);
    s.kernels = {&k};
    s.buffers = {&psi, &staging, &ubo};

    if (!k.create(ctx, k_dipole_kick_spv, k_dipole_kick_spv_size,
                  {{0, kStorage}, {1, kUniform}})) {
        return false;
    }
    if (!ctx.create_device_buffer(bytes, &psi) ||
        !ctx.create_host_buffer(bytes,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                &staging) ||
        !ctx.create_host_buffer(sizeof(Params),
                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &ubo)) {
        return false;
    }
    std::memcpy(staging.mapped, in.data(), bytes);
    std::memcpy(ubo.mapped, &params, sizeof(params));
    flush(ctx, staging);
    flush(ctx, ubo);

    if (!s.arena.create(ctx, 1, 1, 1)) return false;
    VkDescriptorSet set = s.arena.allocate(ctx, k.set_layout());
    if (set == VK_NULL_HANDLE) return false;
    s.arena.write_buffer(ctx, set, 0, kStorage, psi.buf);
    s.arena.write_buffer(ctx, set, 1, kUniform, ubo.buf, sizeof(Params));

    if (!s.shot.begin(ctx)) return false;
    record_uploads(s.shot.cb(), staging, {{&psi, 0, bytes}});
    k.bind(s.shot.cb(), set);
    vkCmdDispatch(s.shot.cb(), static_cast<std::uint32_t>((n + 255) / 256), 1,
                  1);
    record_readback(s.shot.cb(), psi, staging, bytes);
    if (!s.shot.submit_and_wait(ctx)) return false;
    invalidate(ctx, staging);

    const float* out = static_cast<const float*>(staging.mapped);
    double max_err = 0.0;
    for (std::size_t idx = 0; idx < n; ++idx) {
        max_err = std::max(max_err, std::abs(out[2 * idx] - cpu[idx].real()));
        max_err =
            std::max(max_err, std::abs(out[2 * idx + 1] - cpu[idx].imag()));
    }
    const bool pass = max_err < 1e-4;
    std::printf(
        "dipole-kick kernel (raw Vulkan): max |gpu - cpu| = %.3e  [%s]\n",
        max_err, pass ? "PASS" : "FAIL");
    return pass;
}

// <grad V> = sum |psi|^2 grad V: vec4 field-input reduction. Same
// data/tolerance as qrhicheck (grad V at binding 4).
bool check_mean_force(ses_vk::DeviceContext& ctx) {
    const std::size_t n = 20000;
    std::vector<ses::Complex<double>> psi_d(n);
    std::vector<float> grad_f(4 * n);
    double cpu[3] = {0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        psi_d[i] = ses::Complex<double>{0.4 * std::sin(0.011 * x),
                                        0.3 * std::cos(0.013 * x) + 0.05};
        const double gx = std::sin(0.007 * x);
        const double gy = std::cos(0.005 * x) - 0.3;
        const double gz = 0.2 * std::sin(0.017 * x);
        grad_f[4 * i + 0] = static_cast<float>(gx);
        grad_f[4 * i + 1] = static_cast<float>(gy);
        grad_f[4 * i + 2] = static_cast<float>(gz);
        grad_f[4 * i + 3] = 0.0f;
        const double d =
            psi_d[i].real() * psi_d[i].real() + psi_d[i].imag() * psi_d[i].imag();
        cpu[0] += d * gx;
        cpu[1] += d * gy;
        cpu[2] += d * gz;
    }
    const std::vector<float> psi_f = to_rg32f(psi_d);
    const VkDeviceSize psi_bytes = psi_f.size() * sizeof(float);
    const VkDeviceSize grad_bytes = grad_f.size() * sizeof(float);
    const int groups = 256;
    const VkDeviceSize part_bytes = 4u * groups * sizeof(float);

    struct alignas(16) Params {
        std::uint32_t n, pad0, pad1, pad2;
    };
    ses_vk::Kernel k;
    ses_vk::Buffer psi{}, grad{}, partials{}, staging{}, ubo{};
    Scope s(ctx);
    s.kernels = {&k};
    s.buffers = {&psi, &grad, &partials, &staging, &ubo};

    if (!k.create(ctx, k_mean_force_spv, k_mean_force_spv_size,
                  {{0, kStorage}, {1, kUniform}, {2, kStorage}, {4, kStorage}})) {
        return false;
    }
    if (!ctx.create_device_buffer(psi_bytes, &psi) ||
        !ctx.create_device_buffer(grad_bytes, &grad) ||
        !ctx.create_device_buffer(part_bytes, &partials) ||
        !ctx.create_host_buffer(psi_bytes + grad_bytes,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                &staging) ||
        !ctx.create_host_buffer(sizeof(Params),
                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &ubo)) {
        return false;
    }
    std::memcpy(staging.mapped, psi_f.data(), psi_bytes);
    std::memcpy(static_cast<char*>(staging.mapped) + psi_bytes, grad_f.data(),
                grad_bytes);
    const Params params{static_cast<std::uint32_t>(n), 0, 0, 0};
    std::memcpy(ubo.mapped, &params, sizeof(params));
    flush(ctx, staging);
    flush(ctx, ubo);

    if (!s.arena.create(ctx, 1, 3, 1)) return false;
    VkDescriptorSet set = s.arena.allocate(ctx, k.set_layout());
    if (set == VK_NULL_HANDLE) return false;
    s.arena.write_buffer(ctx, set, 0, kStorage, psi.buf);
    s.arena.write_buffer(ctx, set, 1, kUniform, ubo.buf, sizeof(Params));
    s.arena.write_buffer(ctx, set, 2, kStorage, partials.buf);
    s.arena.write_buffer(ctx, set, 4, kStorage, grad.buf);

    if (!s.shot.begin(ctx)) return false;
    record_uploads(s.shot.cb(), staging,
                   {{&psi, 0, psi_bytes}, {&grad, psi_bytes, grad_bytes}});
    k.bind(s.shot.cb(), set);
    vkCmdDispatch(s.shot.cb(), groups, 1, 1);
    record_readback(s.shot.cb(), partials, staging, part_bytes);
    if (!s.shot.submit_and_wait(ctx)) return false;
    invalidate(ctx, staging);

    const float* p = static_cast<const float*>(staging.mapped);
    double gpu[3] = {0.0, 0.0, 0.0};
    for (int g = 0; g < groups; ++g) {
        gpu[0] += p[4 * g + 0];
        gpu[1] += p[4 * g + 1];
        gpu[2] += p[4 * g + 2];
    }
    const double mag =
        std::sqrt(cpu[0] * cpu[0] + cpu[1] * cpu[1] + cpu[2] * cpu[2]);
    const double err = std::sqrt((gpu[0] - cpu[0]) * (gpu[0] - cpu[0]) +
                                 (gpu[1] - cpu[1]) * (gpu[1] - cpu[1]) +
                                 (gpu[2] - cpu[2]) * (gpu[2] - cpu[2]));
    const double rel = (mag > 0.0) ? err / mag : err;
    const bool pass = rel < 1e-4;
    std::printf("mean-force <grad V> (raw Vulkan): rel err = %.3e  [%s]\n", rel,
                pass ? "PASS" : "FAIL");
    return pass;
}

// <to| r |from>: three complex reductions with grid coordinates, 6 floats
// per workgroup. Same data/tolerance as qrhicheck.
bool check_dipole(ses_vk::DeviceContext& ctx) {
    const int nx = 8;
    const int ny = 8;
    const int nz = 8;
    const std::size_t n = static_cast<std::size_t>(nx * ny * nz);
    const double box_min[3] = {-4.0, -4.0, -4.0};
    const double cell_h[3] = {1.0, 1.1, 0.9};

    std::vector<ses::Complex<double>> to_d(n);
    std::vector<ses::Complex<double>> from_d(n);
    double cpu[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (std::size_t idx = 0; idx < n; ++idx) {
        const double x = static_cast<double>(idx);
        to_d[idx] =
            ses::Complex<double>{std::sin(0.11 * x) + 0.1, std::cos(0.07 * x)};
        from_d[idx] =
            ses::Complex<double>{std::cos(0.05 * x), std::sin(0.13 * x) - 0.2};
        const int i = static_cast<int>(idx) % nx;
        const int j = (static_cast<int>(idx) / nx) % ny;
        const int kk = static_cast<int>(idx) / (nx * ny);
        const double r[3] = {box_min[0] + i * cell_h[0],
                             box_min[1] + j * cell_h[1],
                             box_min[2] + kk * cell_h[2]};
        const double ar = to_d[idx].real();
        const double ai = to_d[idx].imag();
        const double br = from_d[idx].real();
        const double bi = from_d[idx].imag();
        const double c_re = ar * br + ai * bi;
        const double c_im = ar * bi - ai * br;
        for (int a = 0; a < 3; ++a) {
            cpu[2 * a + 0] += c_re * r[a];
            cpu[2 * a + 1] += c_im * r[a];
        }
    }
    const std::vector<float> to_f = to_rg32f(to_d);
    const std::vector<float> from_f = to_rg32f(from_d);
    const VkDeviceSize bytes = to_f.size() * sizeof(float);
    const int groups = 256;
    const VkDeviceSize part_bytes = 6u * groups * sizeof(float);

    struct alignas(16) Params {
        std::uint32_t n;
        std::int32_t nx;
        std::int32_t ny;
        std::int32_t pad0;
        float box_min[4];
        float cell_h[4];
    };
    Params params{};
    params.n = static_cast<std::uint32_t>(n);
    params.nx = nx;
    params.ny = ny;
    for (int c = 0; c < 3; ++c) {
        params.box_min[c] = static_cast<float>(box_min[c]);
        params.cell_h[c] = static_cast<float>(cell_h[c]);
    }

    ses_vk::Kernel k;
    ses_vk::Buffer fto{}, ffrom{}, partials{}, staging{}, ubo{};
    Scope s(ctx);
    s.kernels = {&k};
    s.buffers = {&fto, &ffrom, &partials, &staging, &ubo};

    if (!k.create(ctx, k_dipole_spv, k_dipole_spv_size,
                  {{0, kStorage}, {1, kUniform}, {2, kStorage}, {3, kStorage}})) {
        return false;
    }
    if (!ctx.create_device_buffer(bytes, &fto) ||
        !ctx.create_device_buffer(bytes, &ffrom) ||
        !ctx.create_device_buffer(part_bytes, &partials) ||
        !ctx.create_host_buffer(part_bytes + 2 * bytes,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                &staging) ||
        !ctx.create_host_buffer(sizeof(Params),
                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &ubo)) {
        return false;
    }
    std::memcpy(staging.mapped, to_f.data(), bytes);
    std::memcpy(static_cast<char*>(staging.mapped) + bytes, from_f.data(),
                bytes);
    std::memcpy(ubo.mapped, &params, sizeof(params));
    flush(ctx, staging);
    flush(ctx, ubo);

    if (!s.arena.create(ctx, 1, 3, 1)) return false;
    VkDescriptorSet set = s.arena.allocate(ctx, k.set_layout());
    if (set == VK_NULL_HANDLE) return false;
    s.arena.write_buffer(ctx, set, 0, kStorage, fto.buf);
    s.arena.write_buffer(ctx, set, 1, kUniform, ubo.buf, sizeof(Params));
    s.arena.write_buffer(ctx, set, 2, kStorage, partials.buf);
    s.arena.write_buffer(ctx, set, 3, kStorage, ffrom.buf);

    if (!s.shot.begin(ctx)) return false;
    record_uploads(s.shot.cb(), staging,
                   {{&fto, 0, bytes}, {&ffrom, bytes, bytes}});
    k.bind(s.shot.cb(), set);
    vkCmdDispatch(s.shot.cb(), groups, 1, 1);
    record_readback(s.shot.cb(), partials, staging, part_bytes);
    if (!s.shot.submit_and_wait(ctx)) return false;
    invalidate(ctx, staging);

    const float* p = static_cast<const float*>(staging.mapped);
    double gpu[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int g = 0; g < groups; ++g) {
        for (int c = 0; c < 6; ++c) {
            gpu[c] += p[6 * g + c];
        }
    }
    double mag2 = 0.0;
    double err2 = 0.0;
    for (int c = 0; c < 6; ++c) {
        mag2 += cpu[c] * cpu[c];
        err2 += (gpu[c] - cpu[c]) * (gpu[c] - cpu[c]);
    }
    const double rel = (mag2 > 0.0) ? std::sqrt(err2 / mag2) : std::sqrt(err2);
    const bool pass = rel < 1e-4;
    std::printf("dipole <to|r|from> (raw Vulkan): rel err = %.3e  [%s]\n", rel,
                pass ? "PASS" : "FAIL");
    return pass;
}

// fp16 storage codec roundtrip: pack fp32 -> half, unpack half -> fp32. Two
// dispatches with an EXPLICIT compute-to-compute barrier on the half buffer
// -- the read-after-write edge QRhi ordered automatically. Tolerance is fp16
// precision, same as qrhicheck.
bool check_fp16_roundtrip(ses_vk::DeviceContext& ctx) {
    const std::size_t n = 4096;
    std::vector<ses::Complex<double>> src_d(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        src_d[i] = ses::Complex<double>{0.5 * std::sin(0.05 * x),
                                        0.5 * std::cos(0.03 * x)};
    }
    const std::vector<float> src_f = to_rg32f(src_d);
    const VkDeviceSize fp32_bytes = src_f.size() * sizeof(float);
    const VkDeviceSize half_bytes = n * sizeof(std::uint32_t);

    struct alignas(16) Params {
        std::uint32_t n, pad0, pad1, pad2;
    };
    ses_vk::Kernel pack;
    ses_vk::Kernel unpack;
    ses_vk::Buffer src{}, half{}, dst{}, staging{}, ubo{};
    Scope s(ctx);
    s.kernels = {&pack, &unpack};
    s.buffers = {&src, &half, &dst, &staging, &ubo};

    if (!pack.create(ctx, k_pack_half_spv, k_pack_half_spv_size,
                     {{0, kStorage}, {1, kUniform}, {6, kStorage}}) ||
        !unpack.create(ctx, k_unpack_half_spv, k_unpack_half_spv_size,
                       {{0, kStorage}, {1, kUniform}, {6, kStorage}})) {
        return false;
    }
    if (!ctx.create_device_buffer(fp32_bytes, &src) ||
        !ctx.create_device_buffer(half_bytes, &half) ||
        !ctx.create_device_buffer(fp32_bytes, &dst) ||
        !ctx.create_host_buffer(fp32_bytes,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                &staging) ||
        !ctx.create_host_buffer(sizeof(Params),
                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &ubo)) {
        return false;
    }
    std::memcpy(staging.mapped, src_f.data(), fp32_bytes);
    const Params params{static_cast<std::uint32_t>(n), 0, 0, 0};
    std::memcpy(ubo.mapped, &params, sizeof(params));
    flush(ctx, staging);
    flush(ctx, ubo);

    if (!s.arena.create(ctx, 2, 4, 2)) return false;
    VkDescriptorSet pack_set = s.arena.allocate(ctx, pack.set_layout());
    VkDescriptorSet unpack_set = s.arena.allocate(ctx, unpack.set_layout());
    if (pack_set == VK_NULL_HANDLE || unpack_set == VK_NULL_HANDLE) {
        return false;
    }
    s.arena.write_buffer(ctx, pack_set, 0, kStorage, src.buf);
    s.arena.write_buffer(ctx, pack_set, 1, kUniform, ubo.buf, sizeof(Params));
    s.arena.write_buffer(ctx, pack_set, 6, kStorage, half.buf);
    s.arena.write_buffer(ctx, unpack_set, 0, kStorage, dst.buf);
    s.arena.write_buffer(ctx, unpack_set, 1, kUniform, ubo.buf, sizeof(Params));
    s.arena.write_buffer(ctx, unpack_set, 6, kStorage, half.buf);

    const std::uint32_t groups = static_cast<std::uint32_t>((n + 255) / 256);
    if (!s.shot.begin(ctx)) return false;
    record_uploads(s.shot.cb(), staging, {{&src, 0, fp32_bytes}});
    pack.bind(s.shot.cb(), pack_set);
    vkCmdDispatch(s.shot.cb(), groups, 1, 1);
    ses_vk::barrier_compute_to_compute(s.shot.cb());
    unpack.bind(s.shot.cb(), unpack_set);
    vkCmdDispatch(s.shot.cb(), groups, 1, 1);
    record_readback(s.shot.cb(), dst, staging, fp32_bytes);
    if (!s.shot.submit_and_wait(ctx)) return false;
    invalidate(ctx, staging);

    const float* out = static_cast<const float*>(staging.mapped);
    double max_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        max_err = std::max(max_err, std::abs(out[2 * i] - src_d[i].real()));
        max_err = std::max(max_err, std::abs(out[2 * i + 1] - src_d[i].imag()));
    }
    const bool pass = max_err < 5e-3;
    std::printf(
        "fp16 pack/unpack roundtrip (raw Vulkan): max |gpu - cpu| = %.3e  "
        "[%s]\n",
        max_err, pass ? "PASS" : "FAIL");
    return pass;
}

// Radix-2 shared-memory line FFT at a baked N: forward unnormalized DFT of
// one contiguous line vs ses::fft. Same data/tolerance as qrhicheck.
bool check_line_fft(ses_vk::DeviceContext& ctx, int N, const unsigned char* spv,
                    std::size_t spv_size) {
    std::vector<ses::Complex<double>> line(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        line[static_cast<std::size_t>(i)] =
            ses::Complex<double>{std::sin(0.3 * i) + 0.1 * i,
                                 std::cos(0.7 * i) - 0.2};
    }
    std::vector<ses::Complex<double>> cpu = line;
    ses::fft(cpu);  // forward, unnormalized -- same convention as the kernel

    const std::vector<float> in = to_rg32f(line);
    const VkDeviceSize bytes = in.size() * sizeof(float);

    struct alignas(16) Params {
        std::int32_t mod_a, mul_b, mul_c, stride, n_lines, pad0, pad1, pad2;
    };
    Params params{};
    params.mod_a = N;  // l=0 -> base = (0 % N)*1 + (0 / N)*0 = 0
    params.mul_b = 1;
    params.mul_c = 0;
    params.stride = 1;
    params.n_lines = 1;

    ses_vk::Kernel k;
    ses_vk::Buffer data{}, staging{}, ubo{};
    Scope s(ctx);
    s.kernels = {&k};
    s.buffers = {&data, &staging, &ubo};

    if (!k.create(ctx, spv, spv_size, {{0, kStorage}, {1, kUniform}})) {
        return false;
    }
    if (!ctx.create_device_buffer(bytes, &data) ||
        !ctx.create_host_buffer(bytes,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                &staging) ||
        !ctx.create_host_buffer(sizeof(Params),
                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &ubo)) {
        return false;
    }
    std::memcpy(staging.mapped, in.data(), bytes);
    std::memcpy(ubo.mapped, &params, sizeof(params));
    flush(ctx, staging);
    flush(ctx, ubo);

    if (!s.arena.create(ctx, 1, 1, 1)) return false;
    VkDescriptorSet set = s.arena.allocate(ctx, k.set_layout());
    if (set == VK_NULL_HANDLE) return false;
    s.arena.write_buffer(ctx, set, 0, kStorage, data.buf);
    s.arena.write_buffer(ctx, set, 1, kUniform, ubo.buf, sizeof(Params));

    if (!s.shot.begin(ctx)) return false;
    record_uploads(s.shot.cb(), staging, {{&data, 0, bytes}});
    k.bind(s.shot.cb(), set);
    vkCmdDispatch(s.shot.cb(), 1, 1, 1);  // n_lines = 1 workgroup
    record_readback(s.shot.cb(), data, staging, bytes);
    if (!s.shot.submit_and_wait(ctx)) return false;
    invalidate(ctx, staging);

    const float* out = static_cast<const float*>(staging.mapped);
    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        max_err = std::max(
            max_err,
            std::abs(out[2 * i] - cpu[static_cast<std::size_t>(i)].real()));
        max_err = std::max(
            max_err,
            std::abs(out[2 * i + 1] - cpu[static_cast<std::size_t>(i)].imag()));
    }
    const bool pass = max_err < 1e-3;
    std::printf("line FFT N=%d (raw Vulkan): max |gpu - cpu| = %.3e  [%s]\n", N,
                max_err, pass ? "PASS" : "FAIL");
    return pass;
}

// 3-D forward FFT of an 8x8x8 cube: the line FFT once per axis, three
// dispatches aliasing ONE buffer with explicit compute-to-compute barriers
// between axes -- the multi-axis orchestration at the heart of the engine's
// Strang step, exactly where QRhi's automatic tracking used to sit. Three
// descriptor sets share one kernel; per-axis uniforms in three tiny UBOs.
bool check_fft3(ses_vk::DeviceContext& ctx) {
    const int nx = 8;
    const int ny = 8;
    const int nz = 8;
    const ses::Grid1D ax{-4.0, 4.0, 8};
    const ses::Grid3D g{ax, ax, ax};
    ses::Field3D original{g};
    for (int i = 0; i < original.size(); ++i) {
        const double x = static_cast<double>(i);
        original.data()[static_cast<std::size_t>(i)] =
            ses::Complex<double>{std::sin(0.61 * x) + 0.15,
                                 std::cos(1.27 * x) - 0.2};
    }
    ses::Field3D cpu = original;
    ses::fft(cpu);  // 3-D forward, x/y/z line FFTs (same convention)

    const std::vector<float> in = to_rg32f(original.data());
    const VkDeviceSize bytes = in.size() * sizeof(float);

    struct alignas(16) AxisParams {
        std::int32_t mod_a, mul_b, mul_c, stride, n_lines, p0, p1, p2;
    };
    // {mod_a, mul_b, mul_c, stride, n_lines} per axis (ses_gpu::axis_passes).
    const AxisParams axp[3] = {
        {ny * nz, nx, 0, 1, ny * nz, 0, 0, 0},       // x-lines (contiguous)
        {nx, 1, nx * ny, nx, nx * nz, 0, 0, 0},      // y-lines
        {nx * ny, 1, 0, nx * ny, nx * ny, 0, 0, 0},  // z-lines
    };

    ses_vk::Kernel k;
    ses_vk::Buffer data{}, staging{}, ubo0{}, ubo1{}, ubo2{};
    ses_vk::Buffer* ubos[3] = {&ubo0, &ubo1, &ubo2};
    Scope s(ctx);
    s.kernels = {&k};
    s.buffers = {&data, &staging, &ubo0, &ubo1, &ubo2};

    if (!k.create(ctx, k_fft_line8_spv, k_fft_line8_spv_size,
                  {{0, kStorage}, {1, kUniform}})) {
        return false;
    }
    if (!ctx.create_device_buffer(bytes, &data) ||
        !ctx.create_host_buffer(bytes,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                &staging)) {
        return false;
    }
    for (int a = 0; a < 3; ++a) {
        if (!ctx.create_host_buffer(sizeof(AxisParams),
                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                    ubos[a])) {
            return false;
        }
        std::memcpy(ubos[a]->mapped, &axp[a], sizeof(AxisParams));
        flush(ctx, *ubos[a]);
    }
    std::memcpy(staging.mapped, in.data(), bytes);
    flush(ctx, staging);

    if (!s.arena.create(ctx, 3, 3, 3)) return false;
    VkDescriptorSet sets[3] = {};
    for (int a = 0; a < 3; ++a) {
        sets[a] = s.arena.allocate(ctx, k.set_layout());
        if (sets[a] == VK_NULL_HANDLE) return false;
        s.arena.write_buffer(ctx, sets[a], 0, kStorage, data.buf);
        s.arena.write_buffer(ctx, sets[a], 1, kUniform, ubos[a]->buf,
                             sizeof(AxisParams));
    }

    if (!s.shot.begin(ctx)) return false;
    record_uploads(s.shot.cb(), staging, {{&data, 0, bytes}});
    for (int a = 0; a < 3; ++a) {
        if (a > 0) {
            ses_vk::barrier_compute_to_compute(s.shot.cb());
        }
        k.bind(s.shot.cb(), sets[a]);
        vkCmdDispatch(s.shot.cb(), static_cast<std::uint32_t>(axp[a].n_lines),
                      1, 1);
    }
    record_readback(s.shot.cb(), data, staging, bytes);
    if (!s.shot.submit_and_wait(ctx)) return false;
    invalidate(ctx, staging);

    const float* out = static_cast<const float*>(staging.mapped);
    double max_err = 0.0;
    double max_mag = 0.0;
    for (std::size_t i = 0; i < cpu.data().size(); ++i) {
        max_err = std::max(max_err, std::abs(out[2 * i] - cpu.data()[i].real()));
        max_err =
            std::max(max_err, std::abs(out[2 * i + 1] - cpu.data()[i].imag()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].real()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].imag()));
    }
    const double tol = 1e-3 + 1e-5 * max_mag;
    const bool pass = max_err < tol;
    std::printf(
        "fft3 8x8x8 (raw Vulkan): max |gpu - cpu| = %.3e (tol %.3e)  [%s]\n",
        max_err, tol, pass ? "PASS" : "FAIL");
    return pass;
}

ses_vk::EngineKernels engine_blobs_8() {
    ses_vk::EngineKernels b;
    b.mul = k_phase_multiply_spv;
    b.mul_size = k_phase_multiply_spv_size;
    b.conj = k_conj_scale_spv;
    b.conj_size = k_conj_scale_spv_size;
    b.fft = k_fft_line8_spv;
    b.fft_size = k_fft_line8_spv_size;
    b.norm = k_norm_peak_spv;
    b.norm_size = k_norm_peak_spv_size;
    b.scale = k_scale_spv;
    b.scale_size = k_scale_spv_size;
    b.kick = k_dipole_kick_spv;
    b.kick_size = k_dipole_kick_spv_size;
    b.shear = k_shear_spv;
    b.shear_size = k_shear_spv_size;
    b.inner = k_inner_product_spv;
    b.inner_size = k_inner_product_spv_size;
    b.axpy = k_axpy_spv;
    b.axpy_size = k_axpy_spv_size;
    b.copy = k_copy_state_spv;
    b.copy_size = k_copy_state_spv_size;
    b.synth = k_synth_spv;
    b.synth_size = k_synth_spv_size;
    b.force = k_mean_force_spv;
    b.force_size = k_mean_force_spv_size;
    b.dipole = k_dipole_spv;
    b.dipole_size = k_dipole_spv_size;
    b.project = k_project_deposit_spv;
    b.project_size = k_project_deposit_spv_size;
    return b;
}

// The production Strang step through ses_vk::Engine, 20 steps on an 8x8x8
// soft-Coulomb grid vs SplitOperator3D::step -- the raw-Vulkan analog of
// qrhicheck's check_engine_step (hand-rolled line-FFT path; same oracle).
bool check_engine_step(ses_vk::DeviceContext& ctx) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v =
        ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const double dt = 0.02;
    const ses::SplitOperator3D cpu_prop{g, v, dt};
    ses::Field3D psi0 = ses::gaussian_wavepacket(g, ses::Vec3d{1.0, 0.0, 0.0},
                                                 ses::Vec3d{1.2, 1.2, 1.2},
                                                 ses::Vec3d{0.0, 0.5, 0.0});

    ses_vk::Engine engine;
    if (!engine.initialize(ctx, g, engine_blobs_8(), cpu_prop.half_potential_phase(),
                           cpu_prop.kinetic_phase(), psi0.data())) {
        std::printf("engine 20 steps (raw Vulkan): engine init FAIL\n");
        return false;
    }

    ses::Field3D cpu = psi0;
    cpu_prop.step(cpu, 20);

    // Verify the default step path (native VkFFT when the plan built), THEN
    // force the hand-rolled line-FFT path so both keep coverage.
    bool all_pass = true;
    for (int mode = 0; mode < 2; ++mode) {
        const bool want_vkfft = (mode == 0);
        engine.set_use_vkfft(want_vkfft);
        if (want_vkfft && !engine.vkfft_active()) {
            continue;  // plan unavailable: the hand-rolled pass covers it
        }
        engine.upload_state(psi0.data());
        engine.step(20);
        std::vector<float> gpu_out;
        if (!engine.readback(gpu_out)) {
            std::printf("engine 20 steps (raw Vulkan): readback FAIL\n");
            return false;
        }
        double max_err = 0.0;
        double max_mag = 0.0;
        for (std::size_t i = 0; i < cpu.data().size(); ++i) {
            max_err =
                std::max(max_err, std::abs(gpu_out[2 * i] - cpu.data()[i].real()));
            max_err = std::max(max_err,
                               std::abs(gpu_out[2 * i + 1] - cpu.data()[i].imag()));
            max_mag = std::max(max_mag, std::abs(cpu.data()[i].real()));
            max_mag = std::max(max_mag, std::abs(cpu.data()[i].imag()));
        }
        const double tol = 1e-4 + 1e-5 * max_mag;
        const bool pass = max_err < tol;
        std::printf(
            "engine 20 steps (%s): max |gpu - cpu| = %.3e (tol %.3e)  [%s]\n",
            engine.vkfft_active() ? "raw Vulkan, native VkFFT"
                                  : "raw Vulkan, line FFT",
            max_err, tol, pass ? "PASS" : "FAIL");
        all_pass = all_pass && pass;
    }
    engine.set_use_vkfft(true);
    return all_pass;
}

#ifdef SES_HAVE_VKFFT
// Informational A/B: wall time of step(30) at 64^3, hand-rolled vs native
// VkFFT (every submission fence-waits, so wall time is honest throughput).
// No assert -- numeric parity is check_engine_step's.
bool check_native_vkfft_perf(ses_vk::DeviceContext& ctx) {
    const ses::Grid1D axis{-8.0, 8.0, 64};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v =
        ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const ses::SplitOperator3D prop{g, v, 0.02};
    ses::Field3D psi0 = ses::gaussian_wavepacket(g, ses::Vec3d{2.0, 0.0, 0.0},
                                                 ses::Vec3d{1.5, 1.5, 1.5},
                                                 ses::Vec3d{0.0, 0.3, 0.0});
    ses_vk::EngineKernels b = engine_blobs_8();
    b.fft = k_fft_line64_spv;
    b.fft_size = k_fft_line64_spv_size;
    ses_vk::Engine engine;
    if (!engine.initialize(ctx, g, b, prop.half_potential_phase(),
                           prop.kinetic_phase(), psi0.data())) {
        std::printf("native vkfft perf: engine init FAIL\n");
        return false;
    }
    if (!engine.vkfft_active()) {
        std::printf("native vkfft perf: plan unavailable -- skipped\n");
        return true;
    }
    auto time_steps = [&engine, &psi0](bool vkfft) {
        engine.set_use_vkfft(vkfft);
        engine.upload_state(psi0.data());
        engine.step(5);  // warm-up (pipelines, driver JIT)
        const auto beg = std::chrono::steady_clock::now();
        engine.step(30);
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - beg).count() /
               30.0;
    };
    const double ms_line = time_steps(false);
    const double ms_vkfft = time_steps(true);
    std::printf(
        "native vkfft perf 64^3: line FFT %.2f ms/step, VkFFT %.2f ms/step "
        "(x%.2f)  [PASS]\n",
        ms_line, ms_vkfft, ms_vkfft > 0.0 ? ms_line / ms_vkfft : 0.0);
    return true;
}
#endif  // SES_HAVE_VKFFT

// Imaginary-time relaxation through ses_vk::Engine, 50 steps on an 8x8x8
// harmonic grid vs ImaginaryTimePropagator3D::relax -- the raw-Vulkan analog
// of qrhicheck's check_relax (per-step renormalize: reduce -> host -> scale).
bool check_engine_relax(ses_vk::DeviceContext& ctx) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::harmonic_potential(g, 1.0, ses::Vec3d{});
    const double dtau = 0.05;
    const ses::SplitOperator3D real_prop{g, v, 0.02};  // phase tables for init
    const ses::ImaginaryTimePropagator3D cpu_relaxer{g, v, dtau};
    ses::Field3D psi0 = ses::gaussian_wavepacket(
        g, ses::Vec3d{1.0, 0.0, 0.0}, ses::Vec3d{1.5, 1.5, 1.5}, ses::Vec3d{});

    ses_vk::Engine engine;
    if (!engine.initialize(ctx, g, engine_blobs_8(),
                           real_prop.half_potential_phase(),
                           real_prop.kinetic_phase(), psi0.data())) {
        std::printf("relax 50 steps (raw Vulkan): engine init FAIL\n");
        return false;
    }
    if (!engine.set_relax_tables(cpu_relaxer.half_potential_weight(),
                                 cpu_relaxer.kinetic_weight(), dtau,
                                 g.cell_volume())) {
        std::printf("relax 50 steps (raw Vulkan): set_relax_tables FAIL\n");
        return false;
    }
    const ses_vk::Engine::RelaxStats stats = engine.relax_step(50);
    std::vector<float> gpu_out;
    if (!engine.readback(gpu_out)) {
        std::printf("relax 50 steps (raw Vulkan): readback FAIL\n");
        return false;
    }

    ses::Field3D cpu = psi0;
    cpu_relaxer.relax(cpu, 50);

    double max_err = 0.0;
    double max_mag = 0.0;
    for (std::size_t i = 0; i < cpu.data().size(); ++i) {
        max_err = std::max(max_err, std::abs(gpu_out[2 * i] - cpu.data()[i].real()));
        max_err =
            std::max(max_err, std::abs(gpu_out[2 * i + 1] - cpu.data()[i].imag()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].real()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].imag()));
    }
    const double tol = 1e-4 + 1e-5 * max_mag;
    const bool pass = max_err < tol;
    std::printf(
        "relax 50 steps (raw Vulkan): max |gpu - cpu| = %.3e (tol %.3e), "
        "E~%.3f  [%s]\n",
        max_err, tol, stats.energy, pass ? "PASS" : "FAIL");
    return pass;
}

// Deflated imaginary-time relax vs ImaginaryTimePropagator3D::relax_deflated
// -- Gram-Schmidt projecting the ground state out each step so the flow
// climbs to the first excited level. Exercises create_state_buffer, the
// inner-product reduction, the axpy projection, and copy_into_psi. Same
// oracles/tolerances as qrhicheck's check_deflation.
bool check_engine_deflation(ses_vk::DeviceContext& ctx) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::harmonic_potential(g, 1.0, ses::Vec3d{});
    const double dtau = 0.05;
    const ses::SplitOperator3D real_prop{g, v, 0.02};  // phase tables for init
    const ses::ImaginaryTimePropagator3D cpu_relaxer{g, v, dtau};

    // CPU ground state (the deflation target) + a mixed-parity guess.
    ses::Field3D ground = ses::gaussian_wavepacket(
        g, ses::Vec3d{}, ses::Vec3d{1.2, 1.2, 1.2}, ses::Vec3d{});
    cpu_relaxer.relax(ground, 600);
    ses::Field3D guess = ses::gaussian_wavepacket(
        g, ses::Vec3d{1.0, 0.5, 0.0}, ses::Vec3d{1.2, 1.2, 1.2}, ses::Vec3d{});

    ses_vk::Engine engine;
    if (!engine.initialize(ctx, g, engine_blobs_8(),
                           real_prop.half_potential_phase(),
                           real_prop.kinetic_phase(), guess.data())) {
        std::printf("deflation (raw Vulkan): engine init FAIL\n");
        return false;
    }
    if (!engine.set_relax_tables(cpu_relaxer.half_potential_weight(),
                                 cpu_relaxer.kinetic_weight(), dtau,
                                 g.cell_volume())) {
        std::printf("deflation (raw Vulkan): set_relax_tables FAIL\n");
        return false;
    }
    const int ground_h = engine.create_state_buffer(ground.data());
    if (ground_h < 0) {
        std::printf("deflation (raw Vulkan): create_state_buffer FAIL\n");
        return false;
    }

    // Inner-product kernel: <ground|guess> vs the CPU double reference.
    const ses::Complex<double> cpu_ip = ses::inner_product(ground, guess);
    const ses::Complex<double> gpu_ip = engine.inner_with_psi(ground_h);
    const double ip_err = std::max(std::abs(gpu_ip.real() - cpu_ip.real()),
                                   std::abs(gpu_ip.imag() - cpu_ip.imag()));
    const bool ip_ok = ip_err < 1e-6;
    std::printf("  inner-product <phi|psi> (raw Vulkan): max err %.3e  [%s]\n",
                ip_err, ip_ok ? "PASS" : "FAIL");

    // Deflated relax: 50 GPU steps vs 50 CPU steps (numeric parity).
    engine.relax_deflated_step({ground_h}, 50);
    std::vector<float> gpu_out;
    if (!engine.readback(gpu_out)) {
        return false;
    }
    ses::Field3D cpu = guess;
    cpu_relaxer.relax_deflated(cpu, {&ground}, 50);
    double max_err = 0.0;
    double max_mag = 0.0;
    for (std::size_t i = 0; i < cpu.data().size(); ++i) {
        max_err = std::max(max_err, std::abs(gpu_out[2 * i] - cpu.data()[i].real()));
        max_err =
            std::max(max_err, std::abs(gpu_out[2 * i + 1] - cpu.data()[i].imag()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].real()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].imag()));
    }
    const double tol = 1e-4 + 1e-5 * max_mag;
    const bool relax_ok = max_err < tol;
    std::printf(
        "  deflated relax 50 steps (raw Vulkan): max |gpu - cpu| = %.3e "
        "(tol %.3e)  [%s]\n",
        max_err, tol, relax_ok ? "PASS" : "FAIL");

    // Excited energy estimator (reported; 8^3 too coarse to assert 5w/2).
    const ses_vk::Engine::RelaxStats stats =
        engine.relax_deflated_step({ground_h}, 550);
    std::printf(
        "  deflated energy estimator (raw Vulkan): E = %.4f (5w/2 = 2.5, "
        "coarse grid)\n",
        stats.energy);

    // copy_into_psi (quantum-jump collapse): bitwise copy of the state.
    engine.copy_into_psi(ground_h);
    std::vector<float> copied;
    if (!engine.readback(copied)) {
        return false;
    }
    const std::vector<float> ground_staged = to_rg32f(ground.data());
    double copy_err = 0.0;
    for (std::size_t i = 0; i < copied.size(); ++i) {
        copy_err = std::max(
            copy_err, static_cast<double>(std::abs(copied[i] - ground_staged[i])));
    }
    const bool copy_ok = copy_err == 0.0;
    std::printf("  copy_into_psi (raw Vulkan): max err %.3e  [%s]\n", copy_err,
                copy_ok ? "PASS" : "FAIL");

    const bool pass = ip_ok && relax_ok && copy_ok;
    std::printf("deflation (raw Vulkan): [%s]\n", pass ? "PASS" : "FAIL");
    return pass;
}

// The driven Strang step (dipole half-kicks around the static tables) vs
// core/drive.hpp driven_step -- same adversarial drive as qrhicheck: skew
// (non-unit) polarization axis, nonzero omega, nonzero start time, so every
// kick uniform (axis/box_min/cell_h/theta) is exercised, per-kick through
// the dynamic-offset UBO slots.
bool check_engine_driven(ses_vk::DeviceContext& ctx) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v =
        ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const double dt = 0.02;
    const ses::SplitOperator3D cpu_prop{g, v, dt};
    const ses::DipoleDrive drive{ses::Vec3d{0.3, -0.2, 1.0}, 0.5, 0.6};
    ses::Field3D psi0 = ses::gaussian_wavepacket(g, ses::Vec3d{1.0, 0.0, 0.0},
                                                 ses::Vec3d{1.2, 1.2, 1.2},
                                                 ses::Vec3d{0.0, 0.5, 0.0});

    ses_vk::Engine engine;
    if (!engine.initialize(ctx, g, engine_blobs_8(),
                           cpu_prop.half_potential_phase(),
                           cpu_prop.kinetic_phase(), psi0.data())) {
        std::printf("driven 20 steps (raw Vulkan): engine init FAIL\n");
        return false;
    }
    engine.driven_step(drive, 1.3, dt, 20);
    std::vector<float> gpu_out;
    if (!engine.readback(gpu_out)) {
        std::printf("driven 20 steps (raw Vulkan): readback FAIL\n");
        return false;
    }

    ses::Field3D cpu = psi0;
    ses::driven_step(cpu, cpu_prop, drive, 1.3, 20);

    double max_err = 0.0;
    double max_mag = 0.0;
    for (std::size_t i = 0; i < cpu.data().size(); ++i) {
        max_err = std::max(max_err, std::abs(gpu_out[2 * i] - cpu.data()[i].real()));
        max_err =
            std::max(max_err, std::abs(gpu_out[2 * i + 1] - cpu.data()[i].imag()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].real()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].imag()));
    }
    const double tol = 1e-4 + 1e-5 * max_mag;
    const bool pass = max_err < tol;
    std::printf(
        "driven 20 steps (raw Vulkan): max |gpu - cpu| = %.3e (tol %.3e)  "
        "[%s]\n",
        max_err, tol, pass ? "PASS" : "FAIL");
    return pass;
}

// Magnetic minimal coupling: the exact three-shear rotate_z vs core
// ses::rotate_z, and the full magnetic Strang step (paramagnetic rotation +
// diamagnetic potential) vs MagneticPropagator3D for a field along z and x.
// Same oracles/tolerances as qrhicheck's check_magnetic.
bool check_engine_magnetic(ses_vk::DeviceContext& ctx) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v =
        ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const double dt = 0.02;
    const double b = 0.5;
    ses::Field3D psi0 = ses::gaussian_wavepacket(
        g, ses::Vec3d{1.0, 0.0, 0.5}, ses::Vec3d{1.4, 1.4, 1.4},
        ses::Vec3d{0.0, 0.4, 0.0});

    const ses::SplitOperator3D base{g, v, dt};
    ses_vk::Engine engine;
    if (!engine.initialize(ctx, g, engine_blobs_8(),
                           base.half_potential_phase(), base.kinetic_phase(),
                           psi0.data())) {
        std::printf("magnetic (raw Vulkan): engine init FAIL\n");
        return false;
    }

    // Exact three-shear rotation vs core rotate_z.
    engine.rotate_z_shear(0.6);
    std::vector<float> gpu_out;
    if (!engine.readback(gpu_out)) {
        return false;
    }
    ses::Field3D cpu = psi0;
    ses::rotate_z(cpu, 0.6);
    double rot_err = 0.0;
    double rot_mag = 0.0;
    for (std::size_t i = 0; i < cpu.data().size(); ++i) {
        rot_err = std::max(rot_err, std::abs(gpu_out[2 * i] - cpu.data()[i].real()));
        rot_err =
            std::max(rot_err, std::abs(gpu_out[2 * i + 1] - cpu.data()[i].imag()));
        rot_mag = std::max(rot_mag, std::abs(cpu.data()[i].real()));
        rot_mag = std::max(rot_mag, std::abs(cpu.data()[i].imag()));
    }
    const double rot_tol = 1e-3 + 1e-4 * rot_mag;
    const bool rot_ok = rot_err < rot_tol;
    std::printf(
        "  rotate_z (three-shear, raw Vulkan): max |gpu - cpu| = %.3e "
        "(tol %.3e)  [%s]\n",
        rot_err, rot_tol, rot_ok ? "PASS" : "FAIL");

    // Full magnetic step vs MagneticPropagator3D, field along z then x.
    bool step_ok = true;
    for (int fa = 2; fa >= 0; fa -= 2) {  // axis z then x
        const ses::MagneticPropagator3D mprop{g, v, dt, b, fa};
        const ses::SplitOperator3D core_diamag{g, mprop.effective_potential(),
                                               dt};
        engine.set_half_potential(core_diamag.half_potential_phase());
        engine.upload_state(psi0.data());
        engine.magnetic_step(fa, 0.5 * b * (0.5 * dt), 20);
        if (!engine.readback(gpu_out)) {
            return false;
        }
        ses::Field3D cpu2 = psi0;
        mprop.step(cpu2, 20);
        double err = 0.0;
        double mag = 0.0;
        for (std::size_t i = 0; i < cpu2.data().size(); ++i) {
            err = std::max(err, std::abs(gpu_out[2 * i] - cpu2.data()[i].real()));
            err = std::max(err,
                           std::abs(gpu_out[2 * i + 1] - cpu2.data()[i].imag()));
            mag = std::max(mag, std::abs(cpu2.data()[i].real()));
            mag = std::max(mag, std::abs(cpu2.data()[i].imag()));
        }
        const double tol = 2e-3 + 1e-4 * mag;
        const bool ok = err < tol;
        std::printf(
            "  magnetic step %c (raw Vulkan): max |gpu - cpu| = %.3e "
            "(tol %.3e)  [%s]\n",
            fa == 2 ? 'z' : 'x', err, tol, ok ? "PASS" : "FAIL");
        step_ok = ok && step_ok;
    }

    const bool pass = rot_ok && step_ok;
    std::printf("magnetic (raw Vulkan): [%s]\n", pass ? "PASS" : "FAIL");
    return pass;
}

// Orbital synthesis: psi = (u(|r|)/|r|) Y_lm built on the GPU from a radial
// table, vs core synthesize_orbital, across l = 0..5 and several m -- same
// cases/tolerance as qrhicheck's check_synth.
bool check_engine_synth(ses_vk::DeviceContext& ctx) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v =
        ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const ses::SplitOperator3D prop{g, v, 0.02};
    ses::Field3D seed = ses::gaussian_wavepacket(
        g, ses::Vec3d{}, ses::Vec3d{1.5, 1.5, 1.5}, ses::Vec3d{});
    ses_vk::Engine engine;
    if (!engine.initialize(ctx, g, engine_blobs_8(),
                           prop.half_potential_phase(), prop.kinetic_phase(),
                           seed.data())) {
        std::printf("orbital synthesis (raw Vulkan): engine init FAIL\n");
        return false;
    }

    const ses::RadialGrid rg{8.0, 1599};
    std::vector<double> vr(static_cast<std::size_t>(rg.n));
    for (int i = 0; i < rg.n; ++i) {
        vr[static_cast<std::size_t>(i)] = 0.5 * rg.r(i) * rg.r(i);
    }
    struct SynCase {
        int l, k, m;
    };
    const SynCase cases[] = {{0, 0, 0}, {1, 0, -1}, {2, 0, 1}, {3, 0, -2},
                             {4, 0, 3}, {5, 0, 5},  {5, 0, 0}};
    double worst = 0.0;
    for (const SynCase& c : cases) {
        const ses::RadialState st =
            ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, c.l), c.k);
        const ses::Field3D cpu = ses::synthesize_orbital(g, rg, st.u, c.l, c.m);
        if (!engine.synthesize_into_psi(st.u, c.l, c.m, rg.h(), rg.rmax, rg.n)) {
            std::printf("orbital synthesis (raw Vulkan): synthesize FAIL\n");
            return false;
        }
        std::vector<float> gpu;
        if (!engine.readback(gpu)) {
            return false;
        }
        for (std::size_t i = 0; i < cpu.data().size(); ++i) {
            worst = std::max(worst, std::abs(static_cast<double>(gpu[2 * i]) -
                                             cpu.data()[i].real()));
            worst = std::max(worst, std::abs(static_cast<double>(gpu[2 * i + 1]) -
                                             cpu.data()[i].imag()));
        }
    }
    const bool ok = worst < 1e-4;
    std::printf(
        "orbital synthesis (u/r)Ylm (raw Vulkan): max |gpu - cpu| = %.3e  "
        "[%s]\n",
        worst, ok ? "PASS" : "FAIL");
    return ok;
}

// Engine-level mean force: set_potential_gradient (host central differences,
// periodic) + the <grad V> reduction, vs the same construction in CPU double
// on the fp64 field. Rel tolerance covers the fp32 psi/grad narrowing.
bool check_engine_force(ses_vk::DeviceContext& ctx) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v =
        ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const ses::SplitOperator3D prop{g, v, 0.02};
    ses::Field3D psi0 = ses::gaussian_wavepacket(g, ses::Vec3d{1.0, 0.3, 0.0},
                                                 ses::Vec3d{1.2, 1.2, 1.2},
                                                 ses::Vec3d{});
    ses_vk::Engine engine;
    if (!engine.initialize(ctx, g, engine_blobs_8(),
                           prop.half_potential_phase(), prop.kinetic_phase(),
                           psi0.data())) {
        std::printf("mean force (raw Vulkan): engine init FAIL\n");
        return false;
    }
    if (!engine.set_potential_gradient(v)) {
        std::printf("mean force (raw Vulkan): set_potential_gradient FAIL\n");
        return false;
    }
    const ses::Vec3d gpu = engine.mean_force();

    // CPU double reference with the identical periodic central differences.
    const int n = g.x.n;
    double cpu[3] = {0.0, 0.0, 0.0};
    const double i2h[3] = {1.0 / (2.0 * g.x.spacing()),
                           1.0 / (2.0 * g.y.spacing()),
                           1.0 / (2.0 * g.z.spacing())};
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const std::size_t idx = static_cast<std::size_t>(g.flat(i, j, k));
                const double d = psi0.data()[idx].real() * psi0.data()[idx].real() +
                                 psi0.data()[idx].imag() * psi0.data()[idx].imag();
                const double gx =
                    (v[static_cast<std::size_t>(g.flat((i + 1) % n, j, k))] -
                     v[static_cast<std::size_t>(g.flat((i - 1 + n) % n, j, k))]) *
                    i2h[0];
                const double gy =
                    (v[static_cast<std::size_t>(g.flat(i, (j + 1) % n, k))] -
                     v[static_cast<std::size_t>(g.flat(i, (j - 1 + n) % n, k))]) *
                    i2h[1];
                const double gz =
                    (v[static_cast<std::size_t>(g.flat(i, j, (k + 1) % n))] -
                     v[static_cast<std::size_t>(g.flat(i, j, (k - 1 + n) % n))]) *
                    i2h[2];
                cpu[0] += d * gx;
                cpu[1] += d * gy;
                cpu[2] += d * gz;
            }
        }
    }
    const double dv = g.cell_volume();
    cpu[0] *= dv;
    cpu[1] *= dv;
    cpu[2] *= dv;
    const double mag =
        std::sqrt(cpu[0] * cpu[0] + cpu[1] * cpu[1] + cpu[2] * cpu[2]);
    const double err = std::sqrt((gpu.x - cpu[0]) * (gpu.x - cpu[0]) +
                                 (gpu.y - cpu[1]) * (gpu.y - cpu[1]) +
                                 (gpu.z - cpu[2]) * (gpu.z - cpu[2]));
    const double rel = (mag > 0.0) ? err / mag : err;
    const bool pass = rel < 1e-4;
    std::printf("engine mean force (raw Vulkan): rel err = %.3e  [%s]\n", rel,
                pass ? "PASS" : "FAIL");
    return pass;
}

// Orbital-free angular projection: GPU deposit (sorted-gather per radial
// bin) + CPU-double radial dot vs project_radial_angular, plus determinism
// (second deposit reproduces the amplitude bit-for-bit). Same oracle as
// qrhicheck's check_project.
bool check_engine_project(ses_vk::DeviceContext& ctx) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v =
        ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const ses::SplitOperator3D prop{g, v, 0.02};
    ses::Field3D seed = ses::gaussian_wavepacket(
        g, ses::Vec3d{}, ses::Vec3d{1.5, 1.5, 1.5}, ses::Vec3d{});
    ses_vk::Engine engine;
    if (!engine.initialize(ctx, g, engine_blobs_8(),
                           prop.half_potential_phase(), prop.kinetic_phase(),
                           seed.data())) {
        std::printf("orbital-free projection (raw Vulkan): engine init FAIL\n");
        return false;
    }
    const ses::RadialGrid rg{8.0, 200};
    std::vector<std::vector<double>> u_by_level(2);
    for (int lev = 0; lev < 2; ++lev) {
        u_by_level[static_cast<std::size_t>(lev)].resize(
            static_cast<std::size_t>(rg.n));
        for (int i = 0; i < rg.n; ++i) {
            const double r = rg.r(i);
            u_by_level[static_cast<std::size_t>(lev)][static_cast<std::size_t>(i)] =
                (lev == 0) ? r * std::exp(-r) : r * r * std::exp(-0.5 * r);
        }
    }
    const int l_max = 5;
    const ses::RadialBinIndex idx = ses::build_radial_bin_index(g, rg);
    if (!engine.set_projection_index(idx.sorted_cell, idx.bin_off, rg.n, rg.h(),
                                     l_max)) {
        std::printf(
            "orbital-free projection (raw Vulkan): set_projection_index "
            "FAIL\n");
        return false;
    }

    ses::Field3D psi = ses::gaussian_wavepacket(g, ses::Vec3d{1.0, 0.5, -0.4},
                                                ses::Vec3d{1.7, 1.7, 1.7},
                                                ses::Vec3d{0.3, -0.2, 0.1});
    engine.upload_state(psi.data());
    engine.project_psi();

    const std::vector<ses::ProjectorState> states = {
        {0, 0, 0}, {1, 1, 0}, {1, 1, 1}, {0, 2, -1}, {1, 4, 2}, {0, 5, 3}};
    const ses::RadialAngularProjection cpu =
        ses::project_radial_angular(psi, rg, u_by_level, states, l_max);
    double worst = 0.0;
    for (std::size_t s = 0; s < states.size(); ++s) {
        const ses::ProjectorState& st = states[s];
        const ses::Complex<double> gpu = engine.project_amplitude(
            u_by_level[static_cast<std::size_t>(st.level)], st.l, st.m);
        const ses::Complex<double> ref =
            cpu.amp[s] * std::sqrt(cpu.norm2[static_cast<std::size_t>(s)]);
        const double e = std::max(std::abs(gpu.real() - ref.real()),
                                  std::abs(gpu.imag() - ref.imag())) /
                         (1.0 + std::abs(ref));
        worst = std::max(worst, e);
    }
    const ses::Complex<double> a1 = engine.project_amplitude(u_by_level[1], 1, 0);
    engine.project_psi();
    const ses::Complex<double> a2 = engine.project_amplitude(u_by_level[1], 1, 0);
    const bool deterministic = (a1 == a2);
    const bool ok = worst < 1e-3 && deterministic;
    std::printf(
        "orbital-free projection <n|psi> (raw Vulkan): max rel = %.3e, "
        "deterministic = %d  [%s]\n",
        worst, deterministic ? 1 : 0, ok ? "PASS" : "FAIL");
    return ok;
}

// dipole_between from two resident states (synthesized on the GPU via
// synthesize_state -- also exercising the psi-untouched atlas path) vs the
// CPU double reduction over the same synthesized orbitals.
bool check_engine_dipole_between(ses_vk::DeviceContext& ctx) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v =
        ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const ses::SplitOperator3D prop{g, v, 0.02};
    ses::Field3D seed = ses::gaussian_wavepacket(
        g, ses::Vec3d{}, ses::Vec3d{1.5, 1.5, 1.5}, ses::Vec3d{});
    ses_vk::Engine engine;
    if (!engine.initialize(ctx, g, engine_blobs_8(),
                           prop.half_potential_phase(), prop.kinetic_phase(),
                           seed.data())) {
        std::printf("dipole_between (raw Vulkan): engine init FAIL\n");
        return false;
    }

    // Two harmonic-well eigenstates (1s-like and 2p-like) via the GPU path.
    const ses::RadialGrid rg{8.0, 1599};
    std::vector<double> vr(static_cast<std::size_t>(rg.n));
    for (int i = 0; i < rg.n; ++i) {
        vr[static_cast<std::size_t>(i)] = 0.5 * rg.r(i) * rg.r(i);
    }
    const ses::RadialState s0 =
        ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, 0), 0);
    const ses::RadialState p0 =
        ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, 1), 0);
    const int to_h = engine.synthesize_state(s0.u, 0, 0, rg.h(), rg.rmax, rg.n);
    const int from_h = engine.synthesize_state(p0.u, 1, 0, rg.h(), rg.rmax, rg.n);
    if (to_h < 0 || from_h < 0) {
        std::printf("dipole_between (raw Vulkan): synthesize_state FAIL\n");
        return false;
    }
    const ses::DipoleMatrixElement gpu = engine.dipole_between(to_h, from_h);

    // CPU: same normalized orbitals in double, same reduction.
    ses::Field3D to_f = ses::synthesize_orbital(g, rg, s0.u, 0, 0);
    ses::Field3D from_f = ses::synthesize_orbital(g, rg, p0.u, 1, 0);
    const double dv = g.cell_volume();
    double d6[6] = {0, 0, 0, 0, 0, 0};
    const int n = g.x.n;
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const std::size_t id = static_cast<std::size_t>(g.flat(i, j, k));
                const double r[3] = {g.x.xmin + i * g.x.spacing(),
                                     g.y.xmin + j * g.y.spacing(),
                                     g.z.xmin + k * g.z.spacing()};
                const double ar = to_f.data()[id].real();
                const double ai = to_f.data()[id].imag();
                const double br = from_f.data()[id].real();
                const double bi = from_f.data()[id].imag();
                const double c_re = ar * br + ai * bi;
                const double c_im = ar * bi - ai * br;
                for (int a = 0; a < 3; ++a) {
                    d6[2 * a + 0] += c_re * r[a];
                    d6[2 * a + 1] += c_im * r[a];
                }
            }
        }
    }
    const ses::Complex<double> cpu[3] = {
        ses::Complex<double>{d6[0] * dv, d6[1] * dv},
        ses::Complex<double>{d6[2] * dv, d6[3] * dv},
        ses::Complex<double>{d6[4] * dv, d6[5] * dv}};
    const ses::Complex<double> gv[3] = {gpu.x, gpu.y, gpu.z};
    double mag2 = 0.0;
    double err2 = 0.0;
    for (int c = 0; c < 3; ++c) {
        mag2 += cpu[c].real() * cpu[c].real() + cpu[c].imag() * cpu[c].imag();
        const double dr = gv[c].real() - cpu[c].real();
        const double di = gv[c].imag() - cpu[c].imag();
        err2 += dr * dr + di * di;
    }
    const double rel = (mag2 > 0.0) ? std::sqrt(err2 / mag2) : std::sqrt(err2);
    const bool pass = rel < 1e-4;
    std::printf("dipole_between resident states (raw Vulkan): rel err = %.3e  "
                "[%s]\n",
                rel, pass ? "PASS" : "FAIL");
    engine.release_state(from_h);
    engine.release_state(to_h);
    return pass;
}

}  // namespace

int main() {
    const char* env = std::getenv("SES_VK_VALIDATION");
    const bool want_validation = (env != nullptr && env[0] == '1');

    ses_vk::DeviceContext ctx;
    const ses_vk::Boot boot = ctx.create(want_validation);
    if (boot == ses_vk::Boot::no_driver) {
        std::printf("vkcheck: no Vulkan runtime/device -- SKIP\n");
        return 77;
    }
    if (boot != ses_vk::Boot::ok) {
        std::fprintf(stderr, "vkcheck: device bootstrap failed\n");
        return 1;
    }
    std::printf("vkcheck: device '%s'%s\n", ctx.device_name,
                ctx.validation_active ? " [validation ON]" : "");

    int failures = 0;
    failures += check_phase_multiply(ctx) ? 0 : 1;
    failures += check_scale(ctx) ? 0 : 1;
    failures += check_norm_peak(ctx) ? 0 : 1;
    failures += check_inner_product(ctx) ? 0 : 1;
    failures += check_dipole_kick(ctx) ? 0 : 1;
    failures += check_mean_force(ctx) ? 0 : 1;
    failures += check_dipole(ctx) ? 0 : 1;
    failures += check_fp16_roundtrip(ctx) ? 0 : 1;
    failures += check_line_fft(ctx, 64, k_fft_line64_spv, k_fft_line64_spv_size)
                    ? 0
                    : 1;
    failures +=
        check_line_fft(ctx, 256, k_fft_line256_spv, k_fft_line256_spv_size)
            ? 0
            : 1;
    failures += check_fft3(ctx) ? 0 : 1;
    failures += check_engine_step(ctx) ? 0 : 1;
    failures += check_engine_relax(ctx) ? 0 : 1;
    failures += check_engine_driven(ctx) ? 0 : 1;
    failures += check_engine_magnetic(ctx) ? 0 : 1;
    failures += check_engine_deflation(ctx) ? 0 : 1;
    failures += check_engine_synth(ctx) ? 0 : 1;
    failures += check_engine_force(ctx) ? 0 : 1;
    failures += check_engine_project(ctx) ? 0 : 1;
    failures += check_engine_dipole_between(ctx) ? 0 : 1;
#ifdef SES_HAVE_VKFFT
    failures += check_native_vkfft_perf(ctx) ? 0 : 1;
#endif

    const int verrs = ses_vk::g_validation_errors.load();
    if (ctx.validation_active && verrs != 0) {
        std::fprintf(stderr, "vkcheck: %d validation error(s)  [FAIL]\n", verrs);
        ++failures;
    }
    if (failures == 0) {
        std::printf("vkcheck: all checks PASS\n");
        return 0;
    }
    std::fprintf(stderr, "vkcheck: %d check(s) FAILED\n", failures);
    return 1;
}
