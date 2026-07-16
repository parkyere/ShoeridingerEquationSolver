// sesolver_vkcheck: framework-free Vulkan verification harness.
//
// volk loads the loader, VMA allocates, and the kernels are the production
// shader sources compiled offline to SPIR-V and embedded as C arrays
// (tools/cmake/bin2h.cmake). Every check runs a GPU kernel or an
// ses_vk::Engine path against a CPU double-precision oracle (core/ code or
// an inline double reference) with a tolerance sized to the fp32 (or fp16)
// arithmetic involved. Coverage: element-wise ops, the reductions, the fp16
// codec (a compute-to-compute hazard), the line FFT at N=64/256, the 3-axis
// fft3 orchestration (three dispatches aliasing one buffer -- the barrier
// pattern at the heart of the Strang step, hand-authored and policed by the
// validation layer), and the engine step/relax/measure paths.
//
// Validation layers: set SES_VK_VALIDATION=1 (and have the layer discoverable
// via VK_ADD_LAYER_PATH or the SDK registry). Any validation ERROR fails the
// run even if the numbers pass.
//
// Exit codes: 0 = all checks PASS, 1 = FAIL, 77 = SKIP (no Vulkan runtime /
// device on this machine; ctest maps 77 to SKIP).


// volk + VMA textually first: VK_*/VMA macros never cross module boundaries,
// and the early claim inoculates against GMF/textual redefinitions.
#include <volk.h>
#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#include <complex>

#include <phase_multiply_spv.h>
#include <half_mul_spv.h>
#include <kin_mul_spv.h>
#include <damp_mul_spv.h>
#include <phase_damp_mul_spv.h>
#include <mcwf_axpy_spv.h>
#include <mc_density_spv.h>
#include <mc_classify_spv.h>
#include <mc_scan_spv.h>
#include <mc_emit_spv.h>
#include <conj_scale_spv.h>
#include <scale_spv.h>
#include <scale_buf_spv.h>
#include <norm_peak_spv.h>
#include <norm_finalize_spv.h>
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
#include <bridge_store_spv.h>
#include <bridge_load_spv.h>
#include <flow_velocity_spv.h>
#include <fft_line8_spv.h>
#include <fft_line64_spv.h>
#include <fft_line256_spv.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
import ses.vk.engine;
import ses.magnetic;
import ses.drive;
import ses.projection;
import ses.sampling;
import ses.imaginary_time;
import ses.propagator;
import ses.rotation;
import ses.radial;
import ses.grid;
import ses.vec;
import ses.fft;
import ses.marching_cubes;
import ses.field;
import ses.harmonics;
import ses.wavepacket;
import ses.potential;

namespace {

constexpr VkDescriptorType kStorage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
constexpr VkDescriptorType kUniform = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

// std::complex<double> -> interleaved rg32f, the engine's upload format: the GPU
// sees exactly the fp32 narrowing of the oracle's double inputs.
std::vector<float> to_rg32f(const std::vector<std::complex<double>>& src) {
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

// psi <- psi (complex-*) phase, vs a CPU double reference; 1e-5 absolute
// covers fp32 rounding on these O(1) values.
bool check_phase_multiply(ses_vk::DeviceContext& ctx) {
    const std::size_t n = 4096;
    std::vector<std::complex<double>> psi_d(n);
    std::vector<std::complex<double>> phase_d(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        psi_d[i] = std::complex<double>{std::sin(0.37 * x) + 0.2,
                                        std::cos(1.13 * x) - 0.1};
        phase_d[i] = std::complex<double>{std::cos(2.9 * x), std::sin(2.9 * x)};
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
        const std::complex<double> expected = psi_d[i] * phase_d[i];
        max_err = std::max(max_err, std::abs(out[2 * i] - expected.real()));
        max_err = std::max(max_err, std::abs(out[2 * i + 1] - expected.imag()));
    }
    const bool pass = max_err < 1e-5;
    std::printf(
        "phase-multiply kernel (raw Vulkan): max |gpu - cpu| = %.3e  [%s]\n",
        max_err, pass ? "PASS" : "FAIL");
    return pass;
}

// data <- s * data, vs a CPU double reference (1e-5 absolute: fp32 on O(1)
// values).
bool check_scale(ses_vk::DeviceContext& ctx) {
    const std::size_t n = 4096;
    const float sc = 0.5f;
    std::vector<std::complex<double>> d0(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        d0[i] = std::complex<double>{std::sin(0.7 * x) - 0.3,
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
// partials, finished on the host, vs a CPU double sum/max. Rel tolerance is
// looser for the sum (1e-5) than the peak (1e-6): the sum accumulates fp32
// rounding across the whole field, the peak is one squared value.
bool check_norm_peak(ses_vk::DeviceContext& ctx) {
    const std::size_t n = 20000;
    std::vector<std::complex<double>> psi_d(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        psi_d[i] = std::complex<double>{0.5 * std::sin(0.013 * x),
                                        0.3 * std::cos(0.017 * x) + 0.1};
    }
    double cpu_sum = 0.0;
    double cpu_peak = 0.0;
    for (const std::complex<double>& z : psi_d) {
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

// <phi|psi> complex two-input reduction, vs a CPU double reference
// (rel 1e-5).
bool check_inner_product(ses_vk::DeviceContext& ctx) {
    const std::size_t n = 20000;
    std::vector<std::complex<double>> psi_d(n);
    std::vector<std::complex<double>> phi_d(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        psi_d[i] =
            std::complex<double>{std::sin(0.019 * x) + 0.1, std::cos(0.023 * x)};
        phi_d[i] =
            std::complex<double>{std::cos(0.011 * x), std::sin(0.029 * x) - 0.2};
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

// One dipole half-kick psi <- exp(-i theta axis.r) psi per grid cell, vs a
// CPU double reference; 1e-4 absolute covers fp32 trig on the phase angle.
// The geometry UBO carries vec4-padded vec3s (std140).
bool check_dipole_kick(ses_vk::DeviceContext& ctx) {
    const int nx = 8;
    const int ny = 8;
    const int nz = 8;
    const std::size_t n = static_cast<std::size_t>(nx * ny * nz);
    const double box_min[3] = {-4.0, -4.0, -4.0};
    const double cell_h[3] = {1.0, 1.1, 0.9};
    const double axis[3] = {0.3, 0.6, -0.2};
    const double theta = 0.15;

    std::vector<std::complex<double>> psi_d(n);
    std::vector<std::complex<double>> cpu(n);
    for (std::size_t idx = 0; idx < n; ++idx) {
        const double x = static_cast<double>(idx);
        psi_d[idx] = std::complex<double>{std::sin(0.13 * x) + 0.2,
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
        cpu[idx] = std::complex<double>{ar * wr - ai * wi, ar * wi + ai * wr};
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

// <grad V> = sum |psi|^2 grad V: vec4 field-input reduction (grad V at
// binding 4), vs a CPU double reference (rel 1e-4).
bool check_mean_force(ses_vk::DeviceContext& ctx) {
    // A synthetic PERIODIC 3D grid: the kernel takes central differences of
    // the scalar potential in-shader, so the check feeds V (not a precomputed
    // gradient) and the CPU reference applies the same periodic differences in
    // double.
    const std::uint32_t nx = 40, ny = 25, nz = 20;
    const std::size_t n = std::size_t(nx) * ny * nz;
    const double i2h[3] = {1.0 / (2.0 * 0.5), 1.0 / (2.0 * 0.4),
                           1.0 / (2.0 * 0.25)};
    std::vector<std::complex<double>> psi_d(n);
    std::vector<double> v_d(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        psi_d[i] = std::complex<double>{0.4 * std::sin(0.011 * x),
                                        0.3 * std::cos(0.013 * x) + 0.05};
        const std::size_t ix = i % nx, iy = (i / nx) % ny, iz = i / (nx * ny);
        v_d[i] = std::sin(0.31 * double(ix)) * std::cos(0.17 * double(iy)) +
                 0.4 * std::sin(0.23 * double(iz)) - 0.2;
    }
    double cpu[3] = {0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t ix = i % nx, iy = (i / nx) % ny, iz = i / (nx * ny);
        const std::size_t row = iz * nx * ny + iy * nx;
        const std::size_t col = iz * nx * ny + ix;
        const std::size_t pil = iy * nx + ix;
        const double gx =
            (v_d[row + (ix + 1) % nx] - v_d[row + (ix + nx - 1) % nx]) * i2h[0];
        const double gy = (v_d[col + ((iy + 1) % ny) * nx] -
                           v_d[col + ((iy + ny - 1) % ny) * nx]) *
                          i2h[1];
        const double gz = (v_d[pil + ((iz + 1) % nz) * nx * ny] -
                           v_d[pil + ((iz + nz - 1) % nz) * nx * ny]) *
                          i2h[2];
        const double d =
            psi_d[i].real() * psi_d[i].real() + psi_d[i].imag() * psi_d[i].imag();
        cpu[0] += d * gx;
        cpu[1] += d * gy;
        cpu[2] += d * gz;
    }
    const std::vector<float> psi_f = to_rg32f(psi_d);
    std::vector<float> v_f(n);
    for (std::size_t i = 0; i < n; ++i) {
        v_f[i] = static_cast<float>(v_d[i]);
    }
    const VkDeviceSize psi_bytes = psi_f.size() * sizeof(float);
    const VkDeviceSize v_bytes = v_f.size() * sizeof(float);
    const int groups = 256;
    const VkDeviceSize part_bytes = 4u * groups * sizeof(float);

    struct alignas(16) Params {
        std::uint32_t n, nx, ny, nz;
        float inv_2h[4];
    };
    ses_vk::Kernel k;
    ses_vk::Buffer psi{}, vbuf{}, partials{}, staging{}, ubo{};
    Scope s(ctx);
    s.kernels = {&k};
    s.buffers = {&psi, &vbuf, &partials, &staging, &ubo};

    if (!k.create(ctx, k_mean_force_spv, k_mean_force_spv_size,
                  {{0, kStorage}, {1, kUniform}, {2, kStorage}, {4, kStorage}})) {
        return false;
    }
    if (!ctx.create_device_buffer(psi_bytes, &psi) ||
        !ctx.create_device_buffer(v_bytes, &vbuf) ||
        !ctx.create_device_buffer(part_bytes, &partials) ||
        !ctx.create_host_buffer(psi_bytes + v_bytes,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                &staging) ||
        !ctx.create_host_buffer(sizeof(Params),
                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &ubo)) {
        return false;
    }
    std::memcpy(staging.mapped, psi_f.data(), psi_bytes);
    std::memcpy(static_cast<char*>(staging.mapped) + psi_bytes, v_f.data(),
                v_bytes);
    const Params params{static_cast<std::uint32_t>(n), nx, ny, nz,
                        {static_cast<float>(i2h[0]), static_cast<float>(i2h[1]),
                         static_cast<float>(i2h[2]), 0.0f}};
    std::memcpy(ubo.mapped, &params, sizeof(params));
    flush(ctx, staging);
    flush(ctx, ubo);

    if (!s.arena.create(ctx, 1, 3, 1)) return false;
    VkDescriptorSet set = s.arena.allocate(ctx, k.set_layout());
    if (set == VK_NULL_HANDLE) return false;
    s.arena.write_buffer(ctx, set, 0, kStorage, psi.buf);
    s.arena.write_buffer(ctx, set, 1, kUniform, ubo.buf, sizeof(Params));
    s.arena.write_buffer(ctx, set, 2, kStorage, partials.buf);
    s.arena.write_buffer(ctx, set, 4, kStorage, vbuf.buf);

    if (!s.shot.begin(ctx)) return false;
    record_uploads(s.shot.cb(), staging,
                   {{&psi, 0, psi_bytes}, {&vbuf, psi_bytes, v_bytes}});
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
// per workgroup, vs a CPU double reference (rel 1e-4).
bool check_dipole(ses_vk::DeviceContext& ctx) {
    const int nx = 8;
    const int ny = 8;
    const int nz = 8;
    const std::size_t n = static_cast<std::size_t>(nx * ny * nz);
    const double box_min[3] = {-4.0, -4.0, -4.0};
    const double cell_h[3] = {1.0, 1.1, 0.9};

    std::vector<std::complex<double>> to_d(n);
    std::vector<std::complex<double>> from_d(n);
    double cpu[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (std::size_t idx = 0; idx < n; ++idx) {
        const double x = static_cast<double>(idx);
        to_d[idx] =
            std::complex<double>{std::sin(0.11 * x) + 0.1, std::cos(0.07 * x)};
        from_d[idx] =
            std::complex<double>{std::cos(0.05 * x), std::sin(0.13 * x) - 0.2};
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
// (read-after-write hazard). Tolerance is fp16 storage precision.
bool check_fp16_roundtrip(ses_vk::DeviceContext& ctx) {
    const std::size_t n = 4096;
    std::vector<std::complex<double>> src_d(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        src_d[i] = std::complex<double>{0.5 * std::sin(0.05 * x),
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
// one contiguous line vs ses::fft (CPU double); 1e-3 absolute because the
// unnormalized spectrum magnitudes grow with N, scaling up fp32 rounding.
bool check_line_fft(ses_vk::DeviceContext& ctx, int N, const unsigned char* spv,
                    std::size_t spv_size) {
    std::vector<std::complex<double>> line(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        line[static_cast<std::size_t>(i)] =
            std::complex<double>{std::sin(0.3 * i) + 0.1 * i,
                                 std::cos(0.7 * i) - 0.2};
    }
    std::vector<std::complex<double>> cpu = line;
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
// Strang step. Three descriptor sets share one kernel; per-axis uniforms in
// three tiny UBOs.
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
            std::complex<double>{std::sin(0.61 * x) + 0.15,
                                 std::cos(1.27 * x) - 0.2};
    }
    ses::Field3D cpu = original;
    ses::fft(cpu);  // 3-D forward, x/y/z line FFTs (same convention)

    const std::vector<float> in = to_rg32f(original.data());
    const VkDeviceSize bytes = in.size() * sizeof(float);

    struct alignas(16) AxisParams {
        std::int32_t mod_a, mul_b, mul_c, stride, n_lines, p0, p1, p2;
    };
    // {mod_a, mul_b, mul_c, stride, n_lines} per axis: the fft_line*.comp
    // line addressing, base = (l % mod_a)*mul_b + (l / mod_a)*mul_c with
    // element step = stride.
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
    b.half_mul = k_half_mul_spv;
    b.half_mul_size = k_half_mul_spv_size;
    b.kin_mul = k_kin_mul_spv;
    b.kin_mul_size = k_kin_mul_spv_size;
    b.damp = k_damp_mul_spv;
    b.damp_size = k_damp_mul_spv_size;
    b.pd = k_phase_damp_mul_spv;
    b.pd_size = k_phase_damp_mul_spv_size;
    b.mcwf = k_mcwf_axpy_spv;
    b.mcwf_size = k_mcwf_axpy_spv_size;
    b.mc_density = k_mc_density_spv;
    b.mc_density_size = k_mc_density_spv_size;
    b.mc_classify = k_mc_classify_spv;
    b.mc_classify_size = k_mc_classify_spv_size;
    b.mc_scan = k_mc_scan_spv;
    b.mc_scan_size = k_mc_scan_spv_size;
    b.mc_emit = k_mc_emit_spv;
    b.mc_emit_size = k_mc_emit_spv_size;
    b.conj = k_conj_scale_spv;
    b.conj_size = k_conj_scale_spv_size;
    b.fft = k_fft_line8_spv;
    b.fft_size = k_fft_line8_spv_size;
    b.norm = k_norm_peak_spv;
    b.norm_size = k_norm_peak_spv_size;
    b.scale = k_scale_spv;
    b.scale_size = k_scale_spv_size;
    b.norm_finalize = k_norm_finalize_spv;
    b.norm_finalize_size = k_norm_finalize_spv_size;
    b.scale_buf = k_scale_buf_spv;
    b.scale_buf_size = k_scale_buf_spv_size;
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
    b.bridge_store = k_bridge_store_spv;
    b.bridge_store_size = k_bridge_store_spv_size;
    b.bridge_load = k_bridge_load_spv;
    b.bridge_load_size = k_bridge_load_spv_size;
    b.flow_velocity = k_flow_velocity_spv;
    b.flow_velocity_size = k_flow_velocity_spv_size;
    b.pack = k_pack_half_spv;
    b.pack_size = k_pack_half_spv_size;
    b.unpack = k_unpack_half_spv;
    b.unpack_size = k_unpack_half_spv_size;
    return b;
}

// The production Strang step through ses_vk::Engine, 20 steps on an 8x8x8
// soft-Coulomb grid vs SplitOperator3D::step (CPU double oracle), covering
// both the native-VkFFT and hand-rolled line-FFT paths.
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
    if (!engine.initialize(ctx, g, engine_blobs_8(), v, dt, psi0.data())) {
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

// step_async: identical recorded content submitted WITHOUT waiting on the
// compute queue (the app overlaps it with rendering). Two chained batches
// with the display bridge exercise the internal wait + the volume ping-pong;
// the result must match the CPU oracle like the blocking step, and the
// display view must be live after the flip.
bool check_engine_step_async(ses_vk::DeviceContext& ctx) {
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
    if (!engine.initialize(ctx, g, engine_blobs_8(), v, dt, psi0.data())) {
        std::printf("engine step_async (raw Vulkan): engine init FAIL\n");
        return false;
    }
    if (!engine.write_psi_to_volume()) {  // creates the ping-pong volumes
        std::printf("engine step_async (raw Vulkan): volume FAIL\n");
        return false;
    }
    ses::Field3D cpu = psi0;
    cpu_prop.step(cpu, 20);
    engine.step_async(10, false, true);
    engine.step_async(10, false, true);  // waits + flips internally
    engine.wait_async();
    std::vector<float> gpu_out;
    if (!engine.readback(gpu_out)) {
        std::printf("engine step_async (raw Vulkan): readback FAIL\n");
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
    const bool view_ok = engine.volume_view() != VK_NULL_HANDLE;
    const bool pass = max_err < tol && view_ok;
    std::printf(
        "engine step_async 2x10 + ping-pong (raw Vulkan): max |gpu - cpu| = "
        "%.3e (tol %.3e), display view %s  [%s]\n",
        max_err, tol, view_ok ? "live" : "NULL", pass ? "PASS" : "FAIL");
    return pass;
}

// Real absorber, per-step: step(5, absorb) vs CPU {Strang step, elementwise
// mask multiply} x5 -- the damp interleaves with every step, so absorption
// cannot depend on batch length. fp32 step tolerance.
bool check_engine_absorber(ses_vk::DeviceContext& ctx) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v =
        ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const double dt = 0.02;
    const ses::SplitOperator3D cpu_prop{g, v, dt};
    const std::vector<double> mask = ses::absorbing_mask(g, 2.0);
    ses::Field3D psi0 = ses::gaussian_wavepacket(g, ses::Vec3d{1.0, 0.0, 0.0},
                                                 ses::Vec3d{1.2, 1.2, 1.2},
                                                 ses::Vec3d{0.0, 0.5, 0.0});
    ses_vk::Engine engine;
    if (!engine.initialize(ctx, g, engine_blobs_8(), v, dt, psi0.data()) ||
        !engine.set_absorber(mask)) {
        std::printf("engine absorber per-step: init FAIL\n");
        return false;
    }
    engine.step(5, /*absorb=*/true);
    std::vector<float> gpu_out;
    if (!engine.readback(gpu_out)) {
        std::printf("engine absorber per-step: readback FAIL\n");
        return false;
    }
    ses::Field3D cpu = psi0;
    for (int s = 0; s < 5; ++s) {
        cpu_prop.step(cpu, 1);
        for (std::size_t i = 0; i < cpu.data().size(); ++i) {
            cpu.data()[i] = cpu.data()[i] * std::complex<double>{mask[i], 0.0};
        }
    }
    double max_err = 0.0;
    double max_mag = 0.0;
    for (std::size_t i = 0; i < cpu.data().size(); ++i) {
        max_err = std::max(max_err,
                           std::abs(gpu_out[2 * i] - cpu.data()[i].real()));
        max_err = std::max(
            max_err, std::abs(gpu_out[2 * i + 1] - cpu.data()[i].imag()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].real()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].imag()));
    }
    const double tol = 1e-4 + 1e-5 * max_mag;
    const bool pass = max_err < tol;
    std::printf(
        "engine absorber per-step (real mask): max |gpu - cpu| = %.3e "
        "(tol %.3e)  [%s]\n",
        max_err, tol, pass ? "PASS" : "FAIL");
    return pass;
}

// GPU marching cubes vs the CPU oracle at 64^3 (512 cell blocks: the scan
// stitching is exercised, not just one workgroup). Both sides classify the
// SAME fp32-quantized density against the SAME float-rounded iso, so the
// triangle COUNT must match exactly; the GPU emits block-major, the CPU
// row-major, so both meshes are canonicalized by a grid-rounded triangle key
// before the per-vertex compare (position/normal tight; phase colour compared
// as a CYCLIC hue distance -- the wheel is discontinuous in RGB at its seams,
// where fp32/fp64 atan2 disagree, so a raw channel diff is the wrong metric).
bool check_engine_marching_cubes(ses_vk::DeviceContext& ctx) {
    const ses::Grid1D axis{-8.0, 8.0, 64};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    // Two offset lobes with opposite momenta: a lumpy surface with real
    // phase structure for the color channel.
    ses::Field3D psi0 = ses::gaussian_wavepacket(g, ses::Vec3d{2.0, 0.0, 0.0},
                                                 ses::Vec3d{1.8, 1.8, 1.8},
                                                 ses::Vec3d{0.0, 0.6, 0.0});
    const ses::Field3D lobe2 = ses::gaussian_wavepacket(
        g, ses::Vec3d{-2.0, 1.0, -1.0}, ses::Vec3d{1.4, 1.4, 1.4},
        ses::Vec3d{0.4, 0.0, 0.0});
    for (std::size_t i = 0; i < psi0.data().size(); ++i) {
        psi0.data()[i] = psi0.data()[i] + 0.8 * lobe2.data()[i];
    }
    ses::normalize(psi0);
    // fp32-quantize psi so CPU and GPU classify the identical field.
    ses::Field3D psi{g};
    for (std::size_t i = 0; i < psi0.data().size(); ++i) {
        psi.data()[i] = std::complex<double>{
            static_cast<float>(psi0.data()[i].real()),
            static_cast<float>(psi0.data()[i].imag())};
    }
    std::vector<double> den(psi.data().size());
    double peak = 0.0;
    for (std::size_t i = 0; i < psi.data().size(); ++i) {
        den[i] = static_cast<float>(std::norm(psi.data()[i]));
        peak = std::max(peak, den[i]);
    }
    const double iso = static_cast<float>(0.25 * peak);
    // Knife-edge guard: no corner within fp32 noise of iso (a 1-ulp flip
    // would desync the exact-count assertion, not the physics).
    for (double d : den) {
        if (std::abs(d - iso) < 1e-9 * peak) {
            std::printf("engine marching cubes: knife-edge iso -- adjust the "
                        "test field  [FAIL]\n");
            return false;
        }
    }

    ses_vk::EngineKernels b = engine_blobs_8();
    b.fft = k_fft_line64_spv;
    b.fft_size = k_fft_line64_spv_size;
    ses_vk::Engine engine;
    if (!engine.initialize(ctx, g, b, v, 0.02, psi.data())) {
        std::printf("engine marching cubes: engine init FAIL\n");
        return false;
    }
    constexpr int kMaxTris = 200000;
    if (!engine.mc_prepare(kMaxTris)) {
        std::printf("engine marching cubes: mc_prepare FAIL\n");
        engine.destroy();
        return false;
    }
    const int gpu_tris = engine.mc_extract(iso);
    std::vector<float> gpu_v;
    if (gpu_tris < 0 || !engine.mc_read_vertices(gpu_v, gpu_tris)) {
        std::printf("engine marching cubes: extract/readback FAIL\n");
        engine.destroy();
        return false;
    }
    engine.destroy();

    const ses::Mesh cpu_mesh = ses::marching_cubes(den, g, iso);
    const std::vector<ses::Rgb> cpu_col = ses::phase_colors(cpu_mesh, psi);
    const int cpu_tris = static_cast<int>(cpu_mesh.vertices.size() / 3);

    bool pass = (gpu_tris == cpu_tris);
    double max_pos = 0.0;
    double max_nrm = 0.0;
    double max_col = 0.0;
    if (pass && gpu_tris > 0) {
        // Canonical order: sort by a triangle key rounded to a 1e-3 grid
        // (positions agree to ~1e-6, so both sides round identically). The
        // key is 12 integers (centroid + all three vertices) compared
        // lexicographically via std::array -- a VALID strict-weak-ordering,
        // unlike an abs()<eps compare (that is std::sort UB and can pair
        // triangles differently across platforms/STLs).
        const auto q = [](double x) {
            return static_cast<long long>(std::llround(x * 1000.0));
        };
        struct Key {
            std::array<long long, 12> k;
            int idx;
        };
        const auto make_key = [&q](double x0, double y0, double z0, double x1,
                                   double y1, double z1, double x2, double y2,
                                   double z2, int idx) {
            return Key{{q((x0 + x1 + x2) / 3.0), q((y0 + y1 + y2) / 3.0),
                        q((z0 + z1 + z2) / 3.0), q(x0), q(y0), q(z0), q(x1),
                        q(y1), q(z1), q(x2), q(y2), q(z2)},
                       idx};
        };
        std::vector<Key> ck(static_cast<std::size_t>(cpu_tris));
        std::vector<Key> gk(static_cast<std::size_t>(gpu_tris));
        for (int t = 0; t < cpu_tris; ++t) {
            const ses::Vec3d& a = cpu_mesh.vertices[3 * t];
            const ses::Vec3d& b2 = cpu_mesh.vertices[3 * t + 1];
            const ses::Vec3d& c = cpu_mesh.vertices[3 * t + 2];
            ck[t] = make_key(a.x, a.y, a.z, b2.x, b2.y, b2.z, c.x, c.y, c.z, t);
            const float* tv = &gpu_v[static_cast<std::size_t>(t) * 27u];
            gk[t] = make_key(tv[0], tv[1], tv[2], tv[9], tv[10], tv[11], tv[18],
                             tv[19], tv[20], t);
        }
        const auto less = [](const Key& a, const Key& b) { return a.k < b.k; };
        std::sort(ck.begin(), ck.end(), less);
        std::sort(gk.begin(), gk.end(), less);
        // The vertex colour is the cyclic HSV phase wheel (S=V=1), which is
        // DISCONTINUOUS in RGB at its six seams: near a seam the fp32-GPU and
        // fp64-CPU atan2 land in adjacent sextants and a raw channel diff
        // jumps to ~1.0 -- a coordinate artifact, not a colour error. So
        // compare the HUE ANGLE recovered from RGB, cyclically (a seam flip
        // is ~0 hue distance; a real wheel/sampling bug is large). Both sides
        // are pure wheel outputs, so the standard hexagon inversion is exact.
        const auto hue01 = [](double r, double g, double b) {
            const double mx = std::max({r, g, b});
            const double mn = std::min({r, g, b});
            const double d = mx - mn;
            if (d < 1e-6) return -1.0;  // gray: wheel never emits it
            double h;
            if (mx == r) {
                h = std::fmod((g - b) / d + 6.0, 6.0);
            } else if (mx == g) {
                h = (b - r) / d + 2.0;
            } else {
                h = (r - g) / d + 4.0;
            }
            return h / 6.0;  // [0, 1)
        };
        for (int t = 0; t < cpu_tris; ++t) {
            const int ci = ck[static_cast<std::size_t>(t)].idx;
            const float* tv =
                &gpu_v[static_cast<std::size_t>(gk[static_cast<std::size_t>(t)].idx) * 27u];
            for (int vtx = 0; vtx < 3; ++vtx) {
                const ses::Vec3d& p = cpu_mesh.vertices[3 * ci + vtx];
                const ses::Vec3d& n = cpu_mesh.normals[3 * ci + vtx];
                const ses::Rgb& col = cpu_col[static_cast<std::size_t>(3 * ci + vtx)];
                const float* gv = tv + vtx * 9;
                max_pos = std::max({max_pos, std::abs(gv[0] - p.x),
                                    std::abs(gv[1] - p.y),
                                    std::abs(gv[2] - p.z)});
                max_nrm = std::max({max_nrm, std::abs(gv[3] - n.x),
                                    std::abs(gv[4] - n.y),
                                    std::abs(gv[5] - n.z)});
                const double hg = hue01(gv[6], gv[7], gv[8]);
                const double hc = hue01(col.r, col.g, col.b);
                if (hg < 0.0 || hc < 0.0) {
                    continue;  // degenerate gray (unreachable on the wheel)
                }
                double dh = std::abs(hg - hc);
                dh = std::min(dh, 1.0 - dh);  // cyclic
                max_col = std::max(max_col, dh);
            }
        }
        // max_col is now a cyclic hue distance in [0, 0.5]; 2e-2 = 7.2 deg,
        // far above fp32 seam jitter, far below any real colour bug.
        pass = max_pos < 1e-3 && max_nrm < 2e-2 && max_col < 2e-2;
    }
    std::printf(
        "engine marching cubes (raw Vulkan): tris %d vs cpu %d, max |dpos| = "
        "%.3e, |dnrm| = %.3e, cyclic dhue = %.3e  [%s]\n",
        gpu_tris, cpu_tris, max_pos, max_nrm, max_col, pass ? "PASS" : "FAIL");
    return pass;
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
    if (!engine.initialize(ctx, g, b, v, 0.02, psi0.data())) {
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
// harmonic grid vs ImaginaryTimePropagator3D::relax (per-step renormalize:
// reduce -> host -> scale).
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
    if (!engine.initialize(ctx, g, engine_blobs_8(), v, 0.02, psi0.data())) {
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
// inner-product reduction, the axpy projection, and copy_into_psi.
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
    if (!engine.initialize(ctx, g, engine_blobs_8(), v, 0.02, guess.data())) {
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
    const std::complex<double> cpu_ip = ses::inner_product(ground, guess);
    const std::complex<double> gpu_ip = engine.inner_with_psi(ground_h);
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
// ses.drive driven_step, with an adversarial drive: skew (non-unit)
// polarization axis, nonzero omega, nonzero start time, so every kick
// uniform (axis/box_min/cell_h/theta) is exercised, per-kick through the
// dynamic-offset UBO slots.
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
    if (!engine.initialize(ctx, g, engine_blobs_8(), v, dt, psi0.data())) {
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
    if (!engine.initialize(ctx, g, engine_blobs_8(), v, dt, psi0.data())) {
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
        engine.set_potential(mprop.effective_potential());
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
// table, vs core synthesize_orbital, across l = 0..5 and several m.
bool check_engine_synth(ses_vk::DeviceContext& ctx) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v =
        ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const ses::SplitOperator3D prop{g, v, 0.02};
    ses::Field3D seed = ses::gaussian_wavepacket(
        g, ses::Vec3d{}, ses::Vec3d{1.5, 1.5, 1.5}, ses::Vec3d{});
    ses_vk::Engine engine;
    if (!engine.initialize(ctx, g, engine_blobs_8(), v, 0.02, seed.data())) {
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
    if (!engine.initialize(ctx, g, engine_blobs_8(), v, 0.02, psi0.data())) {
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
// (a second deposit must reproduce the amplitude bit-for-bit).
bool check_engine_project(ses_vk::DeviceContext& ctx) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v =
        ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const ses::SplitOperator3D prop{g, v, 0.02};
    ses::Field3D seed = ses::gaussian_wavepacket(
        g, ses::Vec3d{}, ses::Vec3d{1.5, 1.5, 1.5}, ses::Vec3d{});
    ses_vk::Engine engine;
    if (!engine.initialize(ctx, g, engine_blobs_8(), v, 0.02, seed.data())) {
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
        const std::complex<double> gpu = engine.project_amplitude(
            u_by_level[static_cast<std::size_t>(st.level)], st.l, st.m);
        const std::complex<double> ref =
            cpu.amp[s] * std::sqrt(cpu.norm2[static_cast<std::size_t>(s)]);
        const double e = std::max(std::abs(gpu.real() - ref.real()),
                                  std::abs(gpu.imag() - ref.imag())) /
                         (1.0 + std::abs(ref));
        worst = std::max(worst, e);
    }
    const std::complex<double> a1 = engine.project_amplitude(u_by_level[1], 1, 0);
    engine.project_psi();
    const std::complex<double> a2 = engine.project_amplitude(u_by_level[1], 1, 0);
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
    if (!engine.initialize(ctx, g, engine_blobs_8(), v, 0.02, seed.data())) {
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
    const std::complex<double> cpu[3] = {
        std::complex<double>{d6[0] * dv, d6[1] * dv},
        std::complex<double>{d6[2] * dv, d6[3] * dv},
        std::complex<double>{d6[4] * dv, d6[5] * dv}};
    const std::complex<double> gv[3] = {gpu.x, gpu.y, gpu.z};
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

// fp16 atlas consumers: the same eigenstate synthesized fp32-resident and
// fp16-packed must agree through the tested fp32 consumers (inner_with_psi
// and dipole_between decode fp16 to scratch on demand). Tolerance is fp16
// storage precision.
bool check_engine_fp16_consumers(ses_vk::DeviceContext& ctx) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v =
        ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const ses::SplitOperator3D prop{g, v, 0.02};
    ses::Field3D psi0 = ses::gaussian_wavepacket(g, ses::Vec3d{0.5, 0.3, 0.0},
                                                 ses::Vec3d{1.4, 1.4, 1.4},
                                                 ses::Vec3d{});
    ses_vk::Engine engine;
    if (!engine.initialize(ctx, g, engine_blobs_8(), v, 0.02, psi0.data())) {
        std::printf("fp16 consumers (raw Vulkan): engine init FAIL\n");
        return false;
    }
    const ses::RadialGrid rg{8.0, 1599};
    std::vector<double> vr(static_cast<std::size_t>(rg.n));
    for (int i = 0; i < rg.n; ++i) {
        vr[static_cast<std::size_t>(i)] = 0.5 * rg.r(i) * rg.r(i);
    }
    const ses::RadialState s0 =
        ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, 0), 0);
    const ses::RadialState p0 =
        ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, 1), 0);
    const int s_f32 = engine.synthesize_state(s0.u, 0, 0, rg.h(), rg.rmax, rg.n);
    const int s_f16 =
        engine.synthesize_state_half(s0.u, 0, 0, rg.h(), rg.rmax, rg.n);
    const int p_f32 = engine.synthesize_state(p0.u, 1, 0, rg.h(), rg.rmax, rg.n);
    if (s_f32 < 0 || s_f16 < 0 || p_f32 < 0) {
        std::printf("fp16 consumers (raw Vulkan): synthesize FAIL\n");
        return false;
    }

    // inner_with_psi: fp32 vs fp16 of the SAME state.
    const std::complex<double> i32 = engine.inner_with_psi(s_f32);
    const std::complex<double> i16 = engine.inner_with_psi(s_f16);
    const double inner_err = std::max(std::abs(i32.real() - i16.real()),
                                      std::abs(i32.imag() - i16.imag()));

    // dipole_between: (fp16 to | fp32 from) vs (fp32 to | fp32 from).
    const ses::DipoleMatrixElement d32 = engine.dipole_between(s_f32, p_f32);
    const ses::DipoleMatrixElement d16 = engine.dipole_between(s_f16, p_f32);
    double dip_err = 0.0;
    const std::complex<double> a[3] = {d32.x, d32.y, d32.z};
    const std::complex<double> b[3] = {d16.x, d16.y, d16.z};
    for (int c = 0; c < 3; ++c) {
        dip_err = std::max(dip_err, std::abs(a[c].real() - b[c].real()));
        dip_err = std::max(dip_err, std::abs(a[c].imag() - b[c].imag()));
    }
    const double worst = std::max(inner_err, dip_err);
    const bool pass = worst < 3e-3;  // fp16 storage precision
    std::printf(
        "fp16 atlas consumers (raw Vulkan): max |fp32 - fp16| = %.3e  [%s]\n",
        worst, pass ? "PASS" : "FAIL");
    return pass;
}

// float -> IEEE binary16 bits, round-to-nearest-even: the conversion the
// RG16F imageStore applies. Handles subnormals/inf/nan exactly.
std::uint16_t f32_to_f16_bits(float value) {
    std::uint32_t f;
    std::memcpy(&f, &value, sizeof(f));
    const std::uint32_t sign = (f >> 16) & 0x8000u;
    f &= 0x7fffffffu;
    if (f > 0x7f800000u) {  // NaN
        return static_cast<std::uint16_t>(sign | 0x7e00u);
    }
    if (f >= 0x47800000u) {  // overflow (incl. inf) -> inf
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (f >= 0x38800000u) {  // normal half: RNE on the dropped 13 bits
        f += 0x00000fffu + ((f >> 13) & 1u);
        return static_cast<std::uint16_t>(sign | ((f - 0x38000000u) >> 13));
    }
    if (f < 0x33000000u) {  // below half of the min subnormal -> +-0
        return static_cast<std::uint16_t>(sign);
    }
    // subnormal half: h_mant = RNE(mant24 * 2^(e-126)), shift in [14, 24]
    const std::uint32_t e = f >> 23;
    const std::uint32_t shift = 126u - e;
    const std::uint32_t mant = (f & 0x7fffffu) | 0x800000u;
    const std::uint32_t halfway = 1u << (shift - 1u);
    const std::uint32_t rem = mant & ((1u << shift) - 1u);
    std::uint32_t hm = mant >> shift;
    if (rem > halfway || (rem == halfway && (hm & 1u) != 0u)) {
        ++hm;  // may carry into the min-normal encoding: correct by layout
    }
    return static_cast<std::uint16_t>(sign | hm);
}

// Exact half -> float (every binary16 value is representable in binary32).
float f16_bits_to_f32(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t em = h & 0x7fffu;
    std::uint32_t f;
    if (em >= 0x7c00u) {  // inf/nan
        f = sign | 0x7f800000u | (static_cast<std::uint32_t>(em & 0x3ffu) << 13);
    } else if (em >= 0x0400u) {  // normal: rebias exponent +112
        f = sign | ((em + 0x1c000u) << 13);
    } else if (em == 0u) {
        f = sign;
    } else {  // subnormal: em * 2^-24, exact in binary32
        const float v = static_cast<float>(em) * 5.9604644775390625e-8f;
        float r = (sign != 0u) ? -v : v;
        return r;
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

// float -> binary16 bits, round-toward-zero: the OTHER conversion Vulkan
// permits for narrowing float stores (spec: RTE or RTZ, implementation's
// choice). Finite overflow clamps to max finite (RTZ never makes inf).
std::uint16_t f32_to_f16_bits_rtz(float value) {
    std::uint32_t f;
    std::memcpy(&f, &value, sizeof(f));
    const std::uint32_t sign = (f >> 16) & 0x8000u;
    f &= 0x7fffffffu;
    if (f > 0x7f800000u) {
        return static_cast<std::uint16_t>(sign | 0x7e00u);  // NaN
    }
    if (f == 0x7f800000u) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);  // inf
    }
    if (f >= 0x47800000u) {
        return static_cast<std::uint16_t>(sign | 0x7bffu);  // clamp
    }
    if (f >= 0x38800000u) {  // normal half: truncate the 13 dropped bits
        return static_cast<std::uint16_t>(sign | ((f - 0x38000000u) >> 13));
    }
    if (f < 0x33800000u) {  // below the min subnormal (2^-24) -> +-0
        return static_cast<std::uint16_t>(sign);
    }
    const std::uint32_t e = f >> 23;
    const std::uint32_t shift = 126u - e;
    const std::uint32_t mant = (f & 0x7fffffu) | 0x800000u;
    return static_cast<std::uint16_t>(sign | (mant >> shift));
}

// The two spec-legal RG16F imageStore quantizations of x (fp32 side).
float f16_quantize(float x) { return f16_bits_to_f32(f32_to_f16_bits(x)); }
float f16_quantize_rtz(float x) {
    return f16_bits_to_f32(f32_to_f16_bits_rtz(x));
}

// initialize -> destroy -> initialize on the SAME Engine object: the second
// run must re-create every lazily-wired descriptor set (reset_lazy_state),
// so a step matches the CPU oracle. Without the reset the stale handles bind
// freed sets (validation error / garbage). SES_VK_VALIDATION=1 also asserts
// zero layer errors across the re-init.
bool check_engine_reinit(ses_vk::DeviceContext& ctx) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v =
        ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const double dt = 0.02;
    const std::vector<double> mask = ses::absorbing_mask(g, 2.0);
    ses::Field3D psi0 = ses::gaussian_wavepacket(g, ses::Vec3d{1.0, 0.0, 0.0},
                                                 ses::Vec3d{1.2, 1.2, 1.2},
                                                 ses::Vec3d{0.0, 0.5, 0.0});

    // The op sequence exercises the LAZY descriptor sets reset_lazy_state
    // guards: set_absorber (damp_/pd_full_/pd_half_ sets) + write_psi_to_volume
    // (store_ ping-pong) + a fused absorb+bridge step. A step() with the
    // default absorb=false/bridge=false would touch only EAGER sets and prove
    // nothing about the reset.
    auto run = [&](ses_vk::Engine& e, int nsteps) -> bool {
        if (!e.set_absorber(mask)) {
            return false;
        }
        e.write_psi_to_volume();
        e.step(nsteps, /*absorb=*/true, /*bridge=*/true);
        return true;
    };

    // Reference: a FRESH engine that ran the sequence once -- the deterministic
    // fp32 result a correct re-init must reproduce bit-for-bit.
    ses_vk::Engine ref;
    if (!ref.initialize(ctx, g, engine_blobs_8(), v, dt, psi0.data()) ||
        !run(ref, 20)) {
        std::printf("engine reinit (raw Vulkan): reference FAIL\n");
        return false;
    }
    std::vector<float> ref_out;
    if (!ref.readback(ref_out)) {
        std::printf("engine reinit (raw Vulkan): reference readback FAIL\n");
        return false;
    }

    // Subject: run the lazy sequence, tear down, re-init, RE-run it. This
    // EXERCISES the lazy re-creation paths (a reset bug that makes cycle 2 fail
    // to re-create, or crash, surfaces here) and, under SES_VK_VALIDATION=1,
    // the layer flags a bind of a freed cycle-1 descriptor. LIMIT: with no
    // validation layer the freed-set bind is UB that VMA masks by reusing
    // cycle-1's memory for cycle-2's buffers, so the output can COINCIDE --
    // output comparison alone cannot guarantee the reset is complete; the
    // validation layer is what makes this strict.
    ses_vk::Engine sub;
    if (!sub.initialize(ctx, g, engine_blobs_8(), v, dt, psi0.data()) ||
        !run(sub, 3)) {
        std::printf("engine reinit (raw Vulkan): first cycle FAIL\n");
        return false;
    }
    sub.destroy();
    if (!sub.initialize(ctx, g, engine_blobs_8(), v, dt, psi0.data()) ||
        !run(sub, 20)) {
        std::printf("engine reinit (raw Vulkan): second cycle FAIL\n");
        return false;
    }
    std::vector<float> sub_out;
    if (!sub.readback(sub_out) || sub_out.size() != ref_out.size()) {
        std::printf("engine reinit (raw Vulkan): subject readback FAIL\n");
        return false;
    }
    double max_err = 0.0;
    for (std::size_t i = 0; i < sub_out.size(); ++i) {
        max_err = std::max(max_err, static_cast<double>(
                                        std::abs(sub_out[i] - ref_out[i])));
    }
    // Deterministic ops on one device -> bitwise equal iff every lazy set was
    // rebuilt; a stale-bind yields garbage (>> tol) or a device error.
    const double tol = 1e-6;
    const bool pass = max_err < tol && sub.volume_view() != VK_NULL_HANDLE;
    std::printf(
        "engine reinit (lazy sets: init->run->destroy->init->run vs fresh ref, "
        "raw Vulkan): max |sub - ref| = %.3e (tol %.0e)  [%s]\n",
        max_err, tol, pass ? "PASS" : "FAIL");
    return pass;
}

// Fused MCWF axpy vs the sequential synthesize -> add_state_into_psi chain:
// the same operator sum evaluated in one dispatch (in-register) vs three
// psi round-trips per state -- identical math, small fp32 reorder tolerance.
bool check_engine_mcwf_axpy(ses_vk::DeviceContext& ctx) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v =
        ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    ses::Field3D psi0 = ses::gaussian_wavepacket(g, ses::Vec3d{1.0, 0.0, 0.0},
                                                 ses::Vec3d{1.2, 1.2, 1.2},
                                                 ses::Vec3d{});
    ses_vk::Engine engine;
    if (!engine.initialize(ctx, g, engine_blobs_8(), v, 0.02, psi0.data())) {
        std::printf("mcwf_axpy (raw Vulkan): engine init FAIL\n");
        return false;
    }
    const ses::RadialGrid rg{4.0, 63};
    std::vector<double> vr(static_cast<std::size_t>(rg.n));
    for (int i = 0; i < rg.n; ++i) {
        vr[static_cast<std::size_t>(i)] = -1.0 / rg.r(i);
    }
    const ses::RadialState s0 =
        ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, 0), 0);
    const ses::RadialState p0 =
        ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, 1), 0);
    // A high-l tesseral term too: the Y_lm table duplicated into
    // mcwf_axpy.comp is otherwise only exercised at l = 0..1.
    const ses::RadialState h0 =
        ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, 5), 0);
    double n2s = 0.0;
    double n2p = 0.0;
    double n2h = 0.0;
    const int hs =
        engine.synthesize_state(s0.u, 0, 0, rg.h(), rg.rmax, rg.n, nullptr,
                                &n2s);
    const int hp =
        engine.synthesize_state(p0.u, 1, 0, rg.h(), rg.rmax, rg.n, nullptr,
                                &n2p);
    const int hh =
        engine.synthesize_state(h0.u, 5, -3, rg.h(), rg.rmax, rg.n, nullptr,
                                &n2h);
    if (hs < 0 || hp < 0 || hh < 0 || n2s <= 0.0 || n2p <= 0.0 ||
        n2h <= 0.0) {
        std::printf("mcwf_axpy (raw Vulkan): synthesize FAIL\n");
        return false;
    }
    // Chain path.
    engine.upload_state(psi0.data());
    engine.add_state_into_psi(hs, 0.03, -0.01);
    engine.add_state_into_psi(hp, -0.02, 0.04);
    engine.add_state_into_psi(hh, 0.015, 0.025);
    std::vector<float> chain;
    if (!engine.readback(chain)) {
        std::printf("mcwf_axpy (raw Vulkan): readback FAIL\n");
        return false;
    }
    // Fused path (same coefficients, cached-norm normalizers).
    engine.upload_state(psi0.data());
    const std::vector<ses_vk::Engine::McwfTerm> terms{
        {&s0.u, 0, 0, 0.03, -0.01, 1.0 / std::sqrt(n2s)},
        {&p0.u, 1, 0, -0.02, 0.04, 1.0 / std::sqrt(n2p)},
        {&h0.u, 5, -3, 0.015, 0.025, 1.0 / std::sqrt(n2h)},
    };
    if (!engine.mcwf_axpy(terms, rg.h(), rg.rmax, rg.n)) {
        std::printf("mcwf_axpy (raw Vulkan): fused dispatch FAIL\n");
        return false;
    }
    std::vector<float> fused;
    if (!engine.readback(fused) || fused.size() != chain.size()) {
        std::printf("mcwf_axpy (raw Vulkan): fused readback FAIL\n");
        return false;
    }
    double max_err = 0.0;
    for (std::size_t i = 0; i < chain.size(); ++i) {
        max_err = std::max(
            max_err, static_cast<double>(std::abs(fused[i] - chain[i])));
    }
    const double tol = 1e-5;  // fp32 reorder between the two evaluations
    const bool pass = max_err < tol;
    std::printf(
        "mcwf_axpy fused vs chain (raw Vulkan): max |diff| = %.3e (tol "
        "%.0e)  [%s]\n",
        max_err, tol, pass ? "PASS" : "FAIL");
    return pass;
}

// SSBO -> 3D volume image bridge: copy psi into the RG16F volume via
// imageStore, read it back through an SSBO (imageLoad). The store quantizes
// to fp16 (display-only precision), so the oracle is the CPU-side fp16
// round-to-nearest of psi -- the round-trip must reproduce THAT bit-for-bit.
// The volume is a raw VkImage the render shell can import.
bool check_engine_bridge(ses_vk::DeviceContext& ctx) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v =
        ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const ses::SplitOperator3D prop{g, v, 0.02};
    ses::Field3D psi0 = ses::gaussian_wavepacket(g, ses::Vec3d{2.0, 0.0, 0.0},
                                                 ses::Vec3d{1.2, 1.2, 1.2},
                                                 ses::Vec3d{0.0, 0.5, 0.0});
    ses_vk::Engine engine;
    if (!engine.initialize(ctx, g, engine_blobs_8(), v, 0.02, psi0.data())) {
        std::printf("bridge (raw Vulkan): engine init FAIL\n");
        return false;
    }
    engine.step(20);  // a non-trivial psi to copy

    std::vector<float> psi_ref;
    std::vector<float> roundtrip;
    if (!engine.readback(psi_ref) || !engine.bridge_roundtrip(roundtrip) ||
        roundtrip.size() != psi_ref.size()) {
        std::printf("bridge (raw Vulkan): roundtrip FAIL\n");
        return false;
    }
    // Oracle: every texel must be bitwise one of the two SPEC-LEGAL fp16
    // quantizations of psi (Vulkan allows RTE or RTZ for narrowing stores).
    // Counts identify what this implementation actually does.
    auto bits = [](float x) {
        std::uint32_t b;
        std::memcpy(&b, &x, sizeof(b));
        return b;
    };
    std::size_t n_illegal = 0;
    std::size_t n_rtz_only = 0;  // matched RTZ where RTE differed
    double max_quant = 0.0;      // RNE quantization scale, for the record
    for (std::size_t i = 0; i < psi_ref.size(); ++i) {
        const float rte = f16_quantize(psi_ref[i]);
        const float rtz = f16_quantize_rtz(psi_ref[i]);
        const std::uint32_t got = bits(roundtrip[i]);
        const bool is_rte = got == bits(rte);
        const bool is_rtz = got == bits(rtz);
        if (!is_rte && !is_rtz) {
            ++n_illegal;
        } else if (is_rtz && !is_rte) {
            ++n_rtz_only;
        }
        max_quant = std::max(
            max_quant, static_cast<double>(std::abs(rte - psi_ref[i])));
    }
    const bool pass = n_illegal == 0;
    std::printf(
        "bridge psi -> RG16F volume -> SSBO (raw Vulkan): %zu texel(s) "
        "outside RTE/RTZ, rtz-only %zu, quantization %.3e  [%s]\n",
        n_illegal, n_rtz_only, max_quant, pass ? "PASS" : "FAIL");
    return pass;
}

// The feature negotiation in create_device must ENABLE (not merely advertise)
// the bits the app builds on -- timeline semaphores, synchronization2, dynamic
// rendering, host query reset, 16-bit storage, and
// shaderDemoteToHelperInvocation (SPIR-V 1.6 lowers `discard` to it). All are
// supported on the Pascal floor, so all must read back enabled; a device that
// genuinely lacks one fails here loudly rather than faulting later when a fast
// path (or a render shader) uses it.
bool check_device_features(const ses_vk::DeviceContext& ctx) {
    const bool pass = ctx.feat_timeline_semaphore && ctx.feat_synchronization2 &&
                      ctx.feat_dynamic_rendering && ctx.feat_host_query_reset &&
                      ctx.feat_storage16 && ctx.feat_demote_to_helper;
    std::printf("device feature negotiation (Pascal floor): timeline=%d "
                "sync2=%d dynRender=%d hostQueryReset=%d storage16=%d "
                "demoteToHelper=%d  [%s]\n",
                ctx.feat_timeline_semaphore ? 1 : 0,
                ctx.feat_synchronization2 ? 1 : 0,
                ctx.feat_dynamic_rendering ? 1 : 0,
                ctx.feat_host_query_reset ? 1 : 0, ctx.feat_storage16 ? 1 : 0,
                ctx.feat_demote_to_helper ? 1 : 0, pass ? "PASS" : "FAIL");
    return pass;
}

// GPU timestamp instrumentation. profile_step() must return a valid per-stage
// breakdown of one 256^3 propagation step with strictly-ordered stamps (both
// FFTs > 0, kick/kin >= 0, total > 0). Prints the empirical decomposition of
// the bandwidth-bound step (min-of-5, per the bench policy).
bool check_timestamp_profile(ses_vk::DeviceContext& ctx) {
    const ses::Grid1D axis{-8.0, 8.0, 256};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::harmonic_potential(g, 1.0, ses::Vec3d{});
    ses::Field3D psi0 = ses::gaussian_wavepacket(
        g, ses::Vec3d{1.0, 0.0, 0.0}, ses::Vec3d{2.0, 2.0, 2.0}, ses::Vec3d{});
    ses_vk::EngineKernels b = engine_blobs_8();
    b.fft = k_fft_line256_spv;
    b.fft_size = k_fft_line256_spv_size;
    ses_vk::Engine engine;
    if (!engine.initialize(ctx, g, b, v, 0.02, psi0.data())) {
        std::printf("timestamp profile 256^3: engine init FAIL\n");
        return false;
    }
    ses_vk::Engine::StepProfile best{};
    for (int i = 0; i < 5; ++i) {  // min-of-N: fastest run is the least-noised
        const ses_vk::Engine::StepProfile p = engine.profile_step();
        if (p.valid && (!best.valid || p.total_ms < best.total_ms)) {
            best = p;
        }
    }
    const bool step_pass = best.valid && best.fwd_fft_ms > 0.0 &&
                           best.inv_fft_ms > 0.0 && best.kin_mul_ms >= 0.0 &&
                           best.kick_ms >= 0.0 && best.total_ms > 0.0;
    const double fft = best.fwd_fft_ms + best.inv_fft_ms;
    std::printf(
        "timestamp profile 256^3 step (raw Vulkan): kick %.3f + fwdFFT %.3f + "
        "kin_mul %.3f + invFFT %.3f = %.3f ms (FFT pair %.0f%%)  [%s]\n",
        best.kick_ms, best.fwd_fft_ms, best.kin_mul_ms, best.inv_fft_ms,
        best.total_ms,
        best.total_ms > 0.0 ? 100.0 * fft / best.total_ms : 0.0,
        step_pass ? "PASS" : "FAIL");

    // Relax breakdown -- the imaginary step body vs the per-step norm+peak
    // reduction. Reuses this 256^3 engine with relax tables.
    const double dtau = 0.02;
    const ses::ImaginaryTimePropagator3D relaxer{g, v, dtau};
    bool relax_pass = false;
    if (engine.set_relax_tables(relaxer.half_potential_weight(),
                                relaxer.kinetic_weight(), dtau,
                                g.cell_volume())) {
        ses_vk::Engine::RelaxProfile rbest{};
        for (int i = 0; i < 5; ++i) {
            const ses_vk::Engine::RelaxProfile rp = engine.profile_relax();
            if (rp.valid && (!rbest.valid || rp.total_ms < rbest.total_ms)) {
                rbest = rp;
            }
        }
        relax_pass = rbest.valid && rbest.step_body_ms > 0.0 &&
                     rbest.norm_reduce_ms > 0.0 && rbest.total_ms > 0.0;
        std::printf(
            "timestamp profile 256^3 relax (raw Vulkan): stepBody %.3f + "
            "normReduce %.3f = %.3f ms (reduction %.0f%%)  [%s]\n",
            rbest.step_body_ms, rbest.norm_reduce_ms, rbest.total_ms,
            rbest.total_ms > 0.0 ? 100.0 * rbest.norm_reduce_ms / rbest.total_ms
                                 : 0.0,
            relax_pass ? "PASS" : "FAIL");
    } else {
        std::printf("timestamp profile 256^3 relax: set_relax_tables FAIL\n");
    }
    return step_pass && relax_pass;
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
    failures += check_device_features(ctx) ? 0 : 1;
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
    failures += check_engine_absorber(ctx) ? 0 : 1;
    failures += check_engine_relax(ctx) ? 0 : 1;
    failures += check_engine_driven(ctx) ? 0 : 1;
    failures += check_engine_magnetic(ctx) ? 0 : 1;
    failures += check_engine_deflation(ctx) ? 0 : 1;
    failures += check_engine_synth(ctx) ? 0 : 1;
    failures += check_engine_force(ctx) ? 0 : 1;
    failures += check_engine_project(ctx) ? 0 : 1;
    failures += check_engine_dipole_between(ctx) ? 0 : 1;
    failures += check_engine_fp16_consumers(ctx) ? 0 : 1;
    failures += check_engine_mcwf_axpy(ctx) ? 0 : 1;
    failures += check_engine_reinit(ctx) ? 0 : 1;
    failures += check_engine_marching_cubes(ctx) ? 0 : 1;
    failures += check_engine_bridge(ctx) ? 0 : 1;
    failures += check_engine_step_async(ctx) ? 0 : 1;
#ifdef SES_HAVE_VKFFT
    failures += check_native_vkfft_perf(ctx) ? 0 : 1;
    failures += check_timestamp_profile(ctx) ? 0 : 1;
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
