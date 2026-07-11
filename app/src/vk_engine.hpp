#pragma once

// ses_vk::Engine (M5 Stage 2): the framework-free analog of
// ses_qrhi::QrhiEngine's core propagation -- the split-operator Strang step
// and imaginary-time relaxation, on raw Vulkan via the vk_compute.hpp layer.
// No Qt anywhere. SPIR-V blobs are dependency-injected (EngineKernels), so
// the engine has no resource system; the DeviceContext is passed in, so the
// same engine runs on a self-created device (headless: checks, clusters) or,
// later, on handles adopted from the GUI's QRhi.
//
// Numerical contract: byte-identical to the QRhi engine's hand-rolled path.
// Same kernels (the qsb-decorated Vulkan-GLSL sources, baked offline), same
// dispatch chain (halfV . IFFT . kin . FFT . halfV; the inverse FFT = conj .
// FFT . conj/N), same std140 parameter blocks, same host-double reduction
// finishes. What QRhi did implicitly and this engine does explicitly: a
// compute-to-compute memory barrier before every dispatch that aliases psi
// (all of them), transfer barriers around uploads/readbacks, and a fence
// wait per submission (the analog of endOffscreenFrame's synchronization).

#include "vk_compute.hpp"

#include <core/complex.hpp>
#include <core/grid.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace ses_vk {

// The SPIR-V blobs the engine's core propagation needs. The caller owns the
// storage (embedded C arrays in the harness; a file loader later).
struct EngineKernels {
    const unsigned char* mul = nullptr;    // phase_multiply.comp
    std::size_t mul_size = 0;
    const unsigned char* conj = nullptr;   // conj_scale.comp
    std::size_t conj_size = 0;
    const unsigned char* fft = nullptr;    // fft_line<n>.comp for the grid n
    std::size_t fft_size = 0;
    const unsigned char* norm = nullptr;   // norm_peak.comp
    std::size_t norm_size = 0;
    const unsigned char* scale = nullptr;  // scale.comp
    std::size_t scale_size = 0;
};

class Engine {
public:
    // Free-energy estimate + normalized peak density from the per-step
    // renormalization (QRhi/GL RelaxStats parity).
    struct RelaxStats {
        double energy = 0.0;
        double peak = 0.0;
    };

    Engine() = default;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    ~Engine() { destroy(); }

    // half_v / kinetic are SplitOperator3D's phase tables; psi0 the initial
    // field. Cubic grids only (one baked fft_line<n>), like the QRhi engine.
    bool initialize(DeviceContext& ctx, const ses::Grid3D& grid,
                    const EngineKernels& blobs,
                    const std::vector<ses::Complex<double>>& half_v,
                    const std::vector<ses::Complex<double>>& kinetic,
                    const std::vector<ses::Complex<double>>& psi0) {
        ctx_ = &ctx;
        n_ = grid.x.n;
        cells_ = static_cast<std::size_t>(grid.size());
        cell_volume_ = grid.cell_volume();
        if (grid.y.n != n_ || grid.z.n != n_) {
            std::fprintf(stderr, "ses_vk::Engine: only cubic grids supported\n");
            return false;
        }
        mul_groups_ = static_cast<std::uint32_t>((cells_ + 255) / 256);
        field_bytes_ = 2 * cells_ * sizeof(float);

        if (!mul_.create(ctx, blobs.mul, blobs.mul_size,
                         {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                          {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !conj_.create(ctx, blobs.conj, blobs.conj_size,
                          {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                           {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !fft_.create(ctx, blobs.fft, blobs.fft_size,
                         {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                          {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !norm_.create(ctx, blobs.norm, blobs.norm_size,
                          {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                           {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                           {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !scale_.create(ctx, blobs.scale, blobs.scale_size,
                           {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}})) {
            return false;
        }

        if (!ctx.create_device_buffer(field_bytes_, &psi_) ||
            !ctx.create_device_buffer(field_bytes_, &half_) ||
            !ctx.create_device_buffer(field_bytes_, &kin_) ||
            !ctx.create_device_buffer(2 * kGroups * sizeof(float), &partials_) ||
            !ctx.create_host_buffer(field_bytes_,
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    &staging_)) {
            return false;
        }

        // std140 parameter blocks, written once into host-mapped UBOs (every
        // submission fence-waits, so no in-flight aliasing; the scale UBO is
        // the only one rewritten, between submissions).
        const MulParams muln{static_cast<std::uint32_t>(cells_), 0, 0, 0};
        const ConjParams conj1{static_cast<std::uint32_t>(cells_), 1.0f, 0.0f,
                               0.0f};
        const ConjParams conjN{static_cast<std::uint32_t>(cells_),
                               1.0f / static_cast<float>(cells_), 0.0f, 0.0f};
        const std::int32_t nn = n_ * n_;
        const FftParams fftp[3] = {
            {nn, n_, 0, 1, nn, 0, 0, 0},   // x-lines (contiguous)
            {n_, 1, nn, n_, nn, 0, 0, 0},  // y-lines
            {nn, 1, 0, nn, nn, 0, 0, 0},   // z-lines
        };
        if (!write_ubo(&muln_ubo_, &muln, sizeof(muln)) ||
            !write_ubo(&conj1_ubo_, &conj1, sizeof(conj1)) ||
            !write_ubo(&conjN_ubo_, &conjN, sizeof(conjN)) ||
            !write_ubo(&fft_ubo_[0], &fftp[0], sizeof(FftParams)) ||
            !write_ubo(&fft_ubo_[1], &fftp[1], sizeof(FftParams)) ||
            !write_ubo(&fft_ubo_[2], &fftp[2], sizeof(FftParams)) ||
            !write_ubo(&scale_ubo_, &conj1, sizeof(conj1))) {
            return false;
        }

        // Descriptor sets: 9 now + 2 relax sets later.
        if (!arena_.create(ctx_ ? *ctx_ : ctx, 16, 32, 16)) {
            return false;
        }
        half_set_ = make_mul_set(half_.buf);
        kin_set_ = make_mul_set(kin_.buf);
        conj1_set_ = make_unary_set(conj_, conj1_ubo_, sizeof(ConjParams));
        conjN_set_ = make_unary_set(conj_, conjN_ubo_, sizeof(ConjParams));
        for (int a = 0; a < 3; ++a) {
            fft_set_[a] = make_unary_set(fft_, fft_ubo_[a], sizeof(FftParams));
        }
        scale_set_ = make_unary_set(scale_, scale_ubo_, sizeof(ConjParams));
        norm_set_ = arena_.allocate(*ctx_, norm_.set_layout());
        if (norm_set_ != VK_NULL_HANDLE) {
            arena_.write_buffer(*ctx_, norm_set_, 0,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, psi_.buf);
            arena_.write_buffer(*ctx_, norm_set_, 1,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, muln_ubo_.buf,
                                sizeof(MulParams));
            arena_.write_buffer(*ctx_, norm_set_, 2,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                partials_.buf);
        }
        if (half_set_ == VK_NULL_HANDLE || kin_set_ == VK_NULL_HANDLE ||
            conj1_set_ == VK_NULL_HANDLE || conjN_set_ == VK_NULL_HANDLE ||
            fft_set_[0] == VK_NULL_HANDLE || fft_set_[1] == VK_NULL_HANDLE ||
            fft_set_[2] == VK_NULL_HANDLE || scale_set_ == VK_NULL_HANDLE ||
            norm_set_ == VK_NULL_HANDLE) {
            return false;
        }

        return upload_field(half_, half_v) && upload_field(kin_, kinetic) &&
               upload_field(psi_, psi0);
    }

    // psi <- (halfV . IFFT . kin . FFT . halfV)^nsteps psi. One submission;
    // a compute-to-compute barrier precedes every psi-aliasing dispatch.
    void step(int nsteps) {
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return;
        }
        Recorder r{shot.cb(), true};
        for (int s = 0; s < nsteps; ++s) {
            run_step_body(r, half_set_, kin_set_);
        }
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
    }

    // Imaginary-time weight tables (packed vec2(w,0)) + dtau/dV for the
    // renormalization; call after initialize().
    bool set_relax_tables(const std::vector<double>& half_w,
                          const std::vector<double>& kin_w, double dtau,
                          double cell_volume) {
        dtau_ = dtau;
        cell_volume_ = cell_volume;
        std::vector<float> hf(2 * cells_, 0.0f);
        std::vector<float> kf(2 * cells_, 0.0f);
        for (std::size_t i = 0; i < cells_; ++i) {
            hf[2 * i] = static_cast<float>(half_w[i]);
            kf[2 * i] = static_cast<float>(kin_w[i]);
        }
        if (!ctx_->create_device_buffer(field_bytes_, &relax_half_) ||
            !ctx_->create_device_buffer(field_bytes_, &relax_kin_)) {
            return false;
        }
        if (!upload_raw(relax_half_, hf.data(), field_bytes_) ||
            !upload_raw(relax_kin_, kf.data(), field_bytes_)) {
            return false;
        }
        relax_half_set_ = make_mul_set(relax_half_.buf);
        relax_kin_set_ = make_mul_set(relax_kin_.buf);
        return relax_half_set_ != VK_NULL_HANDLE &&
               relax_kin_set_ != VK_NULL_HANDLE;
    }

    // e^{-H dtau} Strang steps with per-step renormalization. Each step: one
    // submission for the imaginary body + norm reduction + partials readback,
    // a host-double finish, then a submission scaling by 1/sqrt(norm). The
    // pre-renorm norm decays as e^{-2 E dtau} -> free energy estimate.
    RelaxStats relax_step(int nsteps) {
        RelaxStats stats;
        for (int s = 0; s < nsteps; ++s) {
            OneShot shot;
            if (!shot.begin(*ctx_)) {
                return stats;
            }
            Recorder r{shot.cb(), true};
            run_step_body(r, relax_half_set_, relax_kin_set_);
            barrier_compute_to_compute(shot.cb());
            norm_.bind(shot.cb(), norm_set_);
            vkCmdDispatch(shot.cb(), kGroups, 1, 1);
            barrier_compute_to_transfer(shot.cb());
            const VkBufferCopy down{0, 0, 2 * kGroups * sizeof(float)};
            vkCmdCopyBuffer(shot.cb(), partials_.buf, staging_.buf, 1, &down);
            barrier_transfer_to_host(shot.cb());
            const bool ok = shot.submit_and_wait(*ctx_);
            shot.destroy(*ctx_);
            if (!ok) {
                return stats;
            }
            vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                    VK_WHOLE_SIZE);
            const float* p = static_cast<const float*>(staging_.mapped);
            double sum = 0.0;
            double peak = 0.0;
            for (std::uint32_t g = 0; g < kGroups; ++g) {
                sum += p[2 * g];
                peak = std::max(peak, static_cast<double>(p[2 * g + 1]));
            }
            const double norm_sq = sum * cell_volume_;
            const double inv = (norm_sq > 0.0) ? 1.0 / std::sqrt(norm_sq) : 0.0;
            stats.energy = (norm_sq > 0.0 && dtau_ > 0.0)
                               ? -std::log(norm_sq) / (2.0 * dtau_)
                               : 0.0;
            stats.peak = (norm_sq > 0.0) ? peak / norm_sq : 0.0;

            const ConjParams sp{static_cast<std::uint32_t>(cells_),
                                static_cast<float>(inv), 0.0f, 0.0f};
            std::memcpy(scale_ubo_.mapped, &sp, sizeof(sp));
            vmaFlushAllocation(ctx_->allocator, scale_ubo_.alloc, 0,
                               VK_WHOLE_SIZE);
            OneShot s2;
            if (!s2.begin(*ctx_)) {
                return stats;
            }
            scale_.bind(s2.cb(), scale_set_);
            vkCmdDispatch(s2.cb(), mul_groups_, 1, 1);
            s2.submit_and_wait(*ctx_);
            s2.destroy(*ctx_);
        }
        return stats;
    }

    // Reset psi from a host field.
    bool upload_state(const std::vector<ses::Complex<double>>& psi) {
        return upload_field(psi_, psi);
    }

    // Interleaved RG floats, 2 per cell.
    bool readback(std::vector<float>& out) {
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return false;
        }
        barrier_compute_to_transfer(shot.cb());
        const VkBufferCopy down{0, 0, field_bytes_};
        vkCmdCopyBuffer(shot.cb(), psi_.buf, staging_.buf, 1, &down);
        barrier_transfer_to_host(shot.cb());
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return false;
        }
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* p = static_cast<const float*>(staging_.mapped);
        out.assign(p, p + 2 * cells_);
        return true;
    }

    void destroy() {
        if (ctx_ == nullptr) {
            return;
        }
        if (ctx_->device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(ctx_->device);
        }
        arena_.destroy(*ctx_);
        scale_.destroy(*ctx_);
        norm_.destroy(*ctx_);
        fft_.destroy(*ctx_);
        conj_.destroy(*ctx_);
        mul_.destroy(*ctx_);
        ctx_->destroy_buffer(&relax_kin_);
        ctx_->destroy_buffer(&relax_half_);
        ctx_->destroy_buffer(&scale_ubo_);
        for (int a = 0; a < 3; ++a) {
            ctx_->destroy_buffer(&fft_ubo_[a]);
        }
        ctx_->destroy_buffer(&conjN_ubo_);
        ctx_->destroy_buffer(&conj1_ubo_);
        ctx_->destroy_buffer(&muln_ubo_);
        ctx_->destroy_buffer(&staging_);
        ctx_->destroy_buffer(&partials_);
        ctx_->destroy_buffer(&kin_);
        ctx_->destroy_buffer(&half_);
        ctx_->destroy_buffer(&psi_);
        ctx_ = nullptr;
    }

private:
    static constexpr std::uint32_t kGroups = 256;

    struct alignas(16) MulParams {
        std::uint32_t n, p0, p1, p2;
    };
    struct alignas(16) ConjParams {
        std::uint32_t n;
        float scale;
        float p0, p1;
    };
    struct alignas(16) FftParams {
        std::int32_t mod_a, mul_b, mul_c, stride, n_lines, p0, p1, p2;
    };

    // Emits the compute-to-compute hazard edge before every dispatch except
    // the first of a command buffer (prior submissions are fence-complete).
    struct Recorder {
        VkCommandBuffer cb;
        bool first;
        void dispatch(const Kernel& k, VkDescriptorSet set,
                      std::uint32_t groups) {
            if (!first) {
                barrier_compute_to_compute(cb);
            }
            first = false;
            k.bind(cb, set);
            vkCmdDispatch(cb, groups, 1, 1);
        }
    };

    static std::vector<float> to_rg32f(
        const std::vector<ses::Complex<double>>& src) {
        std::vector<float> out(2 * src.size());
        for (std::size_t i = 0; i < src.size(); ++i) {
            out[2 * i] = static_cast<float>(src[i].real());
            out[2 * i + 1] = static_cast<float>(src[i].imag());
        }
        return out;
    }

    bool write_ubo(Buffer* ubo, const void* data, std::size_t size) {
        if (!ctx_->create_host_buffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                      ubo)) {
            return false;
        }
        std::memcpy(ubo->mapped, data, size);
        vmaFlushAllocation(ctx_->allocator, ubo->alloc, 0, VK_WHOLE_SIZE);
        return true;
    }

    VkDescriptorSet make_mul_set(VkBuffer table) {
        VkDescriptorSet set = arena_.allocate(*ctx_, mul_.set_layout());
        if (set == VK_NULL_HANDLE) {
            return set;
        }
        arena_.write_buffer(*ctx_, set, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            psi_.buf);
        arena_.write_buffer(*ctx_, set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            table);
        arena_.write_buffer(*ctx_, set, 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                            muln_ubo_.buf, sizeof(MulParams));
        return set;
    }

    VkDescriptorSet make_unary_set(const Kernel& k, const Buffer& ubo,
                                   std::size_t ubo_size) {
        VkDescriptorSet set = arena_.allocate(*ctx_, k.set_layout());
        if (set == VK_NULL_HANDLE) {
            return set;
        }
        arena_.write_buffer(*ctx_, set, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            psi_.buf);
        arena_.write_buffer(*ctx_, set, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                            ubo.buf, ubo_size);
        return set;
    }

    bool upload_field(Buffer& dst,
                      const std::vector<ses::Complex<double>>& src) {
        const std::vector<float> f = to_rg32f(src);
        return upload_raw(dst, f.data(), f.size() * sizeof(float));
    }

    bool upload_raw(Buffer& dst, const void* data, VkDeviceSize bytes) {
        std::memcpy(staging_.mapped, data, bytes);
        vmaFlushAllocation(ctx_->allocator, staging_.alloc, 0, VK_WHOLE_SIZE);
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return false;
        }
        const VkBufferCopy up{0, 0, bytes};
        vkCmdCopyBuffer(shot.cb(), staging_.buf, dst.buf, 1, &up);
        barrier_transfer_to_compute(shot.cb());
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        return ok;
    }

    // halfV . IFFT . kin . FFT . halfV (the inverse FFT = conj . FFT .
    // conj/N) -- the QRhi engine's hand-rolled chain, barriers explicit.
    void run_step_body(Recorder& r, VkDescriptorSet half_set,
                       VkDescriptorSet kin_set) {
        r.dispatch(mul_, half_set, mul_groups_);
        fft3(r);
        r.dispatch(mul_, kin_set, mul_groups_);
        r.dispatch(conj_, conj1_set_, mul_groups_);
        fft3(r);
        r.dispatch(conj_, conjN_set_, mul_groups_);
        r.dispatch(mul_, half_set, mul_groups_);
    }

    void fft3(Recorder& r) {
        const std::uint32_t nn = static_cast<std::uint32_t>(n_) *
                                 static_cast<std::uint32_t>(n_);
        for (int a = 0; a < 3; ++a) {
            r.dispatch(fft_, fft_set_[a], nn);
        }
    }

    DeviceContext* ctx_ = nullptr;
    int n_ = 0;
    std::size_t cells_ = 0;
    std::uint32_t mul_groups_ = 0;
    VkDeviceSize field_bytes_ = 0;
    double cell_volume_ = 1.0;
    double dtau_ = 0.0;

    Kernel mul_;
    Kernel conj_;
    Kernel fft_;
    Kernel norm_;
    Kernel scale_;
    DescriptorArena arena_;

    Buffer psi_{};
    Buffer half_{};
    Buffer kin_{};
    Buffer partials_{};
    Buffer staging_{};
    Buffer muln_ubo_{};
    Buffer conj1_ubo_{};
    Buffer conjN_ubo_{};
    Buffer fft_ubo_[3]{};
    Buffer scale_ubo_{};
    Buffer relax_half_{};
    Buffer relax_kin_{};

    VkDescriptorSet half_set_ = VK_NULL_HANDLE;
    VkDescriptorSet kin_set_ = VK_NULL_HANDLE;
    VkDescriptorSet conj1_set_ = VK_NULL_HANDLE;
    VkDescriptorSet conjN_set_ = VK_NULL_HANDLE;
    VkDescriptorSet fft_set_[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE,
                                   VK_NULL_HANDLE};
    VkDescriptorSet norm_set_ = VK_NULL_HANDLE;
    VkDescriptorSet scale_set_ = VK_NULL_HANDLE;
    VkDescriptorSet relax_half_set_ = VK_NULL_HANDLE;
    VkDescriptorSet relax_kin_set_ = VK_NULL_HANDLE;
};

}  // namespace ses_vk
