#pragma once

// ses_vk::Engine: the split-operator Strang step and imaginary-time
// relaxation, on raw Vulkan via the vk_compute.hpp layer. No Qt anywhere.
// SPIR-V blobs are dependency-injected (EngineKernels), so the engine has
// no resource system; the DeviceContext is passed in, so the same engine
// runs on a self-created device (headless: checks, clusters) or on handles
// adopted from the GUI shell.
//
// Numerical contract: Vulkan-GLSL kernels baked offline to SPIR-V, the
// dispatch chain halfV . IFFT . kin . FFT . halfV (the inverse FFT = conj .
// FFT . conj/N), std140 parameter blocks, host-double reduction finishes.
// Synchronization is fully explicit: a compute-to-compute memory barrier
// before every dispatch that aliases psi (all of them), transfer barriers
// around uploads/readbacks, and a fence wait per submission.

#include "vk_compute.hpp"

#ifdef SES_HAVE_VKFFT
// volk (via vk_device.hpp) already defined VK_NO_PROTOTYPES and declared the
// canonical vk* names as global function pointers, so header-only VkFFT
// compiles and links against volk's dispatch unmodified -- no include-order
// hack, no external command-buffer blocks: the engine owns the device and
// records VkFFTAppend straight into its own primary command buffers.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4702)  // unreachable code inside vkFFT.h at /O2
#endif
#include <VkFFT/vkFFT.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#endif

#include <core/complex.hpp>
#include <core/decay.hpp>
#include <core/drive.hpp>
#include <core/grid.hpp>
#include <core/vec.hpp>

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
    const unsigned char* kick = nullptr;   // dipole_kick.comp
    std::size_t kick_size = 0;
    const unsigned char* shear = nullptr;  // shear.comp
    std::size_t shear_size = 0;
    const unsigned char* inner = nullptr;  // inner_product.comp
    std::size_t inner_size = 0;
    const unsigned char* axpy = nullptr;   // axpy.comp
    std::size_t axpy_size = 0;
    const unsigned char* copy = nullptr;   // copy_state.comp
    std::size_t copy_size = 0;
    const unsigned char* synth = nullptr;  // synth.comp
    std::size_t synth_size = 0;
    const unsigned char* force = nullptr;  // mean_force.comp
    std::size_t force_size = 0;
    const unsigned char* dipole = nullptr;   // dipole.comp
    std::size_t dipole_size = 0;
    const unsigned char* project = nullptr;  // project_deposit.comp
    std::size_t project_size = 0;
    const unsigned char* bridge_store = nullptr;  // bridge_store.comp
    std::size_t bridge_store_size = 0;
    const unsigned char* bridge_load = nullptr;   // bridge_load.comp (checks)
    std::size_t bridge_load_size = 0;
    const unsigned char* pack = nullptr;    // pack_half.comp (fp16 atlas)
    std::size_t pack_size = 0;
    const unsigned char* unpack = nullptr;  // unpack_half.comp
    std::size_t unpack_size = 0;
};

class Engine {
public:
    // Free-energy estimate + normalized peak density from the per-step
    // renormalization.
    struct RelaxStats {
        double energy = 0.0;
        double peak = 0.0;
    };
    // GPU-reduced norm (sum |psi|^2 dV) and raw peak density.
    struct NormPeak {
        double sum = 0.0;
        double peak = 0.0;
    };

    Engine() = default;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    ~Engine() { destroy(); }

    // half_v / kinetic are SplitOperator3D's phase tables; psi0 the initial
    // field. Cubic grids only (one baked fft_line<n>).
    bool initialize(DeviceContext& ctx, const ses::Grid3D& grid,
                    const EngineKernels& blobs,
                    const std::vector<ses::Complex<double>>& half_v,
                    const std::vector<ses::Complex<double>>& kinetic,
                    const std::vector<ses::Complex<double>>& psi0) {
        ctx_ = &ctx;
        grid_ = grid;
        n_ = grid.x.n;
        cells_ = static_cast<std::size_t>(grid.size());
        cell_volume_ = grid.cell_volume();
        if (grid.y.n != n_ || grid.z.n != n_) {
            std::fprintf(stderr, "ses_vk::Engine: only cubic grids supported\n");
            return false;
        }
        mul_groups_ = static_cast<std::uint32_t>((cells_ + 255) / 256);
        field_bytes_ = 2 * cells_ * sizeof(float);

        // Dynamic-offset UBO slot stride: the device's minimum offset
        // alignment, grown to hold one KickParams block.
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(ctx.phys_dev, &props);
        kick_stride_ = static_cast<std::uint32_t>(
            props.limits.minUniformBufferOffsetAlignment);
        if (kick_stride_ == 0) {
            kick_stride_ = 256;
        }
        while (kick_stride_ < sizeof(KickParams)) {
            kick_stride_ *= 2;
        }

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
                            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !kick_.create(ctx, blobs.kick, blobs.kick_size,
                          {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                           {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC}}) ||
            !shear_.create(ctx, blobs.shear, blobs.shear_size,
                           {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !inner_.create(ctx, blobs.inner, blobs.inner_size,
                           {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !axpy_.create(ctx, blobs.axpy, blobs.axpy_size,
                          {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                           {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                           {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !copy_.create(ctx, blobs.copy, blobs.copy_size,
                          {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                           {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                           {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !synth_.create(ctx, blobs.synth, blobs.synth_size,
                           {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                            {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !force_.create(ctx, blobs.force, blobs.force_size,
                           {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                            {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !dipole_.create(ctx, blobs.dipole, blobs.dipole_size,
                            {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                             {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                             {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                             {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !project_.create(ctx, blobs.project, blobs.project_size,
                             {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                              {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                              {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                              {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                              {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !bridge_store_.create(ctx, blobs.bridge_store,
                                  blobs.bridge_store_size,
                                  {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                                   {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
                                   {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !bridge_load_.create(ctx, blobs.bridge_load, blobs.bridge_load_size,
                                 {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                                  {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
                                  {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !pack_.create(ctx, blobs.pack, blobs.pack_size,
                          {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                           {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                           {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !unpack_.create(ctx, blobs.unpack, blobs.unpack_size,
                            {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                             {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                             {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}})) {
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
        staging_bytes_ = field_bytes_;

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
        // conjA: the per-axis inverse-FFT scale (1/n, the single transformed
        // axis) used by the shear path.
        const ConjParams conjA{static_cast<std::uint32_t>(cells_),
                               1.0f / static_cast<float>(n_), 0.0f, 0.0f};
        const ShearParams shear_zero{};
        if (!write_ubo(&muln_ubo_, &muln, sizeof(muln)) ||
            !write_ubo(&conj1_ubo_, &conj1, sizeof(conj1)) ||
            !write_ubo(&conjN_ubo_, &conjN, sizeof(conjN)) ||
            !write_ubo(&fft_ubo_[0], &fftp[0], sizeof(FftParams)) ||
            !write_ubo(&fft_ubo_[1], &fftp[1], sizeof(FftParams)) ||
            !write_ubo(&fft_ubo_[2], &fftp[2], sizeof(FftParams)) ||
            !write_ubo(&scale_ubo_, &conj1, sizeof(conj1)) ||
            !write_ubo(&conjA_ubo_, &conjA, sizeof(conjA)) ||
            !write_ubo(&shear_ubo_[0], &shear_zero, sizeof(shear_zero)) ||
            !write_ubo(&shear_ubo_[1], &shear_zero, sizeof(shear_zero))) {
            return false;
        }
        if (!ctx.create_host_buffer(2 * kick_stride_,
                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                    &kick_ubo_)) {
            return false;
        }
        kick_slots_ = 2;

        // conj (0,0) coefficients: the axpy UBO is rewritten per projection;
        // the synth UBO per synthesis.
        const AxpyParams axpy0{static_cast<std::uint32_t>(cells_), 0, 0.0f,
                               0.0f};
        const SynthParams synth0{};
        if (!write_ubo(&axpy_ubo_, &axpy0, sizeof(axpy0)) ||
            !write_ubo(&synth_ubo_, &synth0, sizeof(synth0))) {
            return false;
        }

        // Descriptor pool shape: base sets + any-target sets + relax sets +
        // per-resident-state sets; the arena chains more pools of the same
        // shape when one runs dry.
        if (!arena_.create(ctx_ ? *ctx_ : ctx, 96, 192, 96, 2, 4)) {
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
        conjA_set_ = make_unary_set(conj_, conjA_ubo_, sizeof(ConjParams));
        shear_set_[0] = make_unary_set(shear_, shear_ubo_[0], sizeof(ShearParams));
        shear_set_[1] = make_unary_set(shear_, shear_ubo_[1], sizeof(ShearParams));
        kick_set_ = arena_.allocate(*ctx_, kick_.set_layout());
        if (kick_set_ != VK_NULL_HANDLE) {
            arena_.write_buffer(*ctx_, kick_set_, 0,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, psi_.buf);
            arena_.write_buffer(*ctx_, kick_set_, 1,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                                kick_ubo_.buf, sizeof(KickParams));
        }
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
        synth_any_set_ = arena_.allocate(*ctx_, synth_.set_layout());
        norm_any_set_ = arena_.allocate(*ctx_, norm_.set_layout());
        scale_any_set_ = arena_.allocate(*ctx_, scale_.set_layout());
        if (half_set_ == VK_NULL_HANDLE || kin_set_ == VK_NULL_HANDLE ||
            conj1_set_ == VK_NULL_HANDLE || conjN_set_ == VK_NULL_HANDLE ||
            fft_set_[0] == VK_NULL_HANDLE || fft_set_[1] == VK_NULL_HANDLE ||
            fft_set_[2] == VK_NULL_HANDLE || scale_set_ == VK_NULL_HANDLE ||
            norm_set_ == VK_NULL_HANDLE || conjA_set_ == VK_NULL_HANDLE ||
            shear_set_[0] == VK_NULL_HANDLE || shear_set_[1] == VK_NULL_HANDLE ||
            kick_set_ == VK_NULL_HANDLE || synth_any_set_ == VK_NULL_HANDLE ||
            norm_any_set_ == VK_NULL_HANDLE ||
            scale_any_set_ == VK_NULL_HANDLE) {
            return false;
        }

        if (!upload_field(half_, half_v) || !upload_field(kin_, kinetic) ||
            !upload_field(psi_, psi0)) {
            return false;
        }
        // Plan the VkFFT 3D transform directly on psi_'s VkBuffer (plan
        // creation compiles shaders via glslang). Failure leaves the engine
        // on the hand-rolled line FFT.
        ensure_vkfft();
        return true;
    }

    // Force the hand-rolled line-FFT path (A/B coverage); no-op without a plan.
    void set_use_vkfft(bool on) { use_vkfft_ = on; }
    bool vkfft_active() const {
#ifdef SES_HAVE_VKFFT
        return use_vkfft_ && vkfft_ready_;
#else
        return false;
#endif
    }

    // psi <- (halfV . IFFT . kin . FFT . halfV)^nsteps psi. One submission;
    // a compute-to-compute barrier precedes every psi-aliasing dispatch.
    // mask_handle >= 0 records the absorbing-mask multiply and bridge=true
    // the psi -> volume store into the SAME submission (no extra fences).
    void step(int nsteps, int mask_handle = -1, bool bridge = false) {
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return;
        }
        Recorder r{shot.cb(), true};
        for (int s = 0; s < nsteps; ++s) {
            run_step_body(r, half_set_, kin_set_);
        }
        record_batch_tail(shot.cb(), mask_handle, bridge);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
    }

    // Driven Strang steps: kick(t) . step . kick(t+dt), theta = amplitude
    // cos(omega t) dt/2. Per-kick thetas differ within the batch, so the kick
    // parameters live in dynamic-offset slots of ONE host-mapped UBO and the
    // whole batch records as a single submission.
    void driven_step(const ses::DipoleDrive& d, double t0, double dt,
                     int nsteps, int mask_handle = -1, bool bridge = false) {
        const int kicks = 2 * nsteps;
        if (!ensure_kick_capacity(kicks)) {
            return;
        }
        char* slots = static_cast<char*>(kick_ubo_.mapped);
        for (int s = 0; s < nsteps; ++s) {
            const double t = t0 + s * dt;
            KickParams kp = make_kick_params(
                d.axis, d.amplitude * std::cos(d.omega * t) * 0.5 * dt);
            std::memcpy(slots + static_cast<std::size_t>(2 * s) * kick_stride_,
                        &kp, sizeof(kp));
            kp.theta = static_cast<float>(
                d.amplitude * std::cos(d.omega * (t + dt)) * 0.5 * dt);
            std::memcpy(
                slots + static_cast<std::size_t>(2 * s + 1) * kick_stride_, &kp,
                sizeof(kp));
        }
        vmaFlushAllocation(ctx_->allocator, kick_ubo_.alloc, 0, VK_WHOLE_SIZE);

        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return;
        }
        Recorder r{shot.cb(), true};
        for (int s = 0; s < nsteps; ++s) {
            r.dispatch_dyn(kick_, kick_set_, mul_groups_,
                           static_cast<std::uint32_t>(2 * s) * kick_stride_);
            run_step_body(r, half_set_, kin_set_);
            r.dispatch_dyn(kick_, kick_set_, mul_groups_,
                           static_cast<std::uint32_t>(2 * s + 1) * kick_stride_);
        }
        record_batch_tail(shot.cb(), mask_handle, bridge);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
    }

    // Swap the half-potential phase table (e.g. the diamagnetic-augmented one
    // before a magnetic run).
    bool set_half_potential(const std::vector<ses::Complex<double>>& half_v) {
        return upload_field(half_, half_v);
    }

    // Exact three-shear rotation of psi about coordinate `axis` by theta --
    // the GPU transcription of core/rotation.hpp rotate_axis. One submission.
    void rotate_axis_shear(int axis, double theta) {
        const int b = (axis + 1) % 3;  // in-plane axes (b x c = axis)
        const int c = (axis + 2) % 3;
        stage_rotation_ubos(b, c, theta);
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return;
        }
        Recorder r{shot.cb(), true};
        record_rotation(r, b, c);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
    }
    void rotate_z_shear(double theta) { rotate_axis_shear(2, theta); }

    // Magnetic Strang step: R(a) . real-step . R(a), a = (B/2)(dt/2), about
    // the field axis. half_ must hold the diamagnetic-augmented table
    // (set_half_potential). The half-angle is the same for every rotation in
    // the batch, so the two shear parameter sets are staged once and the
    // whole batch records as one submission.
    void magnetic_step(int axis, double half_angle, int nsteps,
                       int mask_handle = -1, bool bridge = false) {
        const int b = (axis + 1) % 3;
        const int c = (axis + 2) % 3;
        stage_rotation_ubos(b, c, half_angle);
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return;
        }
        Recorder r{shot.cb(), true};
        for (int s = 0; s < nsteps; ++s) {
            record_rotation(r, b, c);
            run_step_body(r, half_set_, kin_set_);
            record_rotation(r, b, c);
        }
        record_batch_tail(shot.cb(), mask_handle, bridge);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
    }

    // Imaginary-time weight tables (packed vec2(w,0)) + dtau/dV for the
    // renormalization. Re-entrant: the tables are TRANSIENT (268 MB) --
    // directors upload them on entering relaxation and release_relax_tables
    // on leaving; the descriptor sets are allocated once and re-pointed.
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
        if (relax_half_.buf == VK_NULL_HANDLE &&
            (!ctx_->create_device_buffer(field_bytes_, &relax_half_) ||
             !ctx_->create_device_buffer(field_bytes_, &relax_kin_))) {
            return false;
        }
        if (!upload_raw(relax_half_, hf.data(), field_bytes_) ||
            !upload_raw(relax_kin_, kf.data(), field_bytes_)) {
            return false;
        }
        if (relax_half_set_ == VK_NULL_HANDLE) {
            relax_half_set_ = make_mul_set(relax_half_.buf);
            relax_kin_set_ = make_mul_set(relax_kin_.buf);
        } else {
            // Re-point the once-allocated sets at the fresh buffers (legal:
            // every submission fence-waits).
            arena_.write_buffer(*ctx_, relax_half_set_, 1,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                relax_half_.buf);
            arena_.write_buffer(*ctx_, relax_kin_set_, 1,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                relax_kin_.buf);
        }
        return relax_half_set_ != VK_NULL_HANDLE &&
               relax_kin_set_ != VK_NULL_HANDLE;
    }

    // Free the transient imaginary-time tables (the sets stay, re-pointed by
    // the next set_relax_tables).
    void release_relax_tables() {
        if (relax_half_.buf == VK_NULL_HANDLE) {
            return;
        }
        vkDeviceWaitIdle(ctx_->device);
        ctx_->destroy_buffer(&relax_half_);
        ctx_->destroy_buffer(&relax_kin_);
    }
    bool relax_tables_ready() const {
        return relax_half_.buf != VK_NULL_HANDLE;
    }

    // e^{-H dtau} Strang steps with per-step renormalization. Each step: one
    // submission for the imaginary body + norm reduction + partials readback,
    // a host-double finish on THAT readback, then the 1/sqrt(norm) scale
    // submission. The pre-renorm norm decays as e^{-2 E dtau} -> free energy.
    RelaxStats relax_step(int nsteps) {
        RelaxStats stats;
        if (!relax_tables_ready()) {
            return stats;
        }
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
            record_partials_readback(shot.cb());
            shot.submit_and_wait(*ctx_);
            shot.destroy(*ctx_);
            stats = finish_renorm_from_staging();
        }
        return stats;
    }

    // Reset psi from a host field.
    bool upload_state(const std::vector<ses::Complex<double>>& psi) {
        return upload_field(psi_, psi);
    }

    // ---- resident states (one int handle space) ---------------------------

    // Upload a CPU state into its own resident fp32 buffer; returns a handle
    // usable with every per-state op, or -1 on failure.
    int create_state_buffer(const std::vector<ses::Complex<double>>& state) {
        const int handle = create_state_buffer_uninit();
        if (handle < 0) {
            return -1;
        }
        State* st = state_at(handle);
        if (!upload_field(st->buf, state)) {
            ctx_->destroy_buffer(&st->buf);
            return -1;
        }
        return handle;
    }

    // Free a resident state's buffer; the slot stays so handles remain
    // stable (its pool sets are simply abandoned until the arena resets).
    void release_state(int handle) {
        State* st = state_at(handle);
        if (st == nullptr) {
            return;
        }
        vkDeviceWaitIdle(ctx_->device);
        ctx_->destroy_buffer(&st->buf);
    }

    // <state|psi> = sum conj(state)*psi * dV (fp16 states decode to scratch).
    ses::Complex<double> inner_with_psi(int handle) {
        State* st = state_at(handle);
        if (st == nullptr) {
            return {};
        }
        VkDescriptorSet set = st->inner_set;
        if (st->is_half) {
            const VkBuffer sbuf = decode(st, 0);
            if (sbuf == VK_NULL_HANDLE) {
                return {};
            }
            const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            arena_.write_buffer(*ctx_, inner_any_set_, 0, storage, psi_.buf);
            arena_.write_buffer(*ctx_, inner_any_set_, 1,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                muln_ubo_.buf, sizeof(MulParams));
            arena_.write_buffer(*ctx_, inner_any_set_, 2, storage,
                                partials_.buf);
            arena_.write_buffer(*ctx_, inner_any_set_, 3, storage, sbuf);
            set = inner_any_set_;
        }
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return {};
        }
        inner_.bind(shot.cb(), set);
        vkCmdDispatch(shot.cb(), kGroups, 1, 1);
        record_partials_readback(shot.cb());
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return {};
        }
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* p = static_cast<const float*>(staging_.mapped);
        double re = 0.0;
        double im = 0.0;
        for (std::uint32_t g = 0; g < kGroups; ++g) {
            re += p[2 * g];
            im += p[2 * g + 1];
        }
        return ses::Complex<double>{re * cell_volume_, im * cell_volume_};
    }

    // Deflated imaginary-time relax: imaginary Strang body, Gram-Schmidt
    // project-out of every `lower` state (psi -= <phi|psi> phi), renorm.
    RelaxStats relax_deflated_step(const std::vector<int>& lower, int nsteps) {
        RelaxStats stats;
        if (!relax_tables_ready()) {
            return stats;
        }
        for (int s = 0; s < nsteps; ++s) {
            OneShot shot;
            if (!shot.begin(*ctx_)) {
                return stats;
            }
            Recorder r{shot.cb(), true};
            run_step_body(r, relax_half_set_, relax_kin_set_);
            shot.submit_and_wait(*ctx_);
            shot.destroy(*ctx_);
            for (int h : lower) {
                const ses::Complex<double> c = inner_with_psi(h);
                subtract_projection(h, c.real(), c.imag());
            }
            stats = renormalize_and_estimate();
        }
        return stats;
    }

    // psi += (cre + i cim) * state: superposition seeding. fp32 states only.
    void add_state_into_psi(int handle, double cre, double cim) {
        subtract_projection(handle, -cre, -cim);
    }

    // psi <- src (bitwise; the quantum-jump collapse path). fp32 states only.
    void copy_into_psi(int handle) {
        State* st = state_at(handle);
        if (st == nullptr || st->is_half) {
            return;
        }
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return;
        }
        copy_.bind(shot.cb(), st->copy_set);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
    }

    // psi <- psi * state (elementwise): the absorbing-boundary damp.
    // Record the real-time batch tail: the absorbing-mask multiply and/or
    // the psi -> volume bridge into an ALREADY-recording command buffer.
    void record_batch_tail(VkCommandBuffer cb, int mask_handle, bool bridge) {
        if (mask_handle >= 0) {
            State* st = state_at(mask_handle);
            if (st != nullptr && !st->is_half) {
                barrier_compute_to_compute(cb);
                mul_.bind(cb, st->mul_set);
                vkCmdDispatch(cb, mul_groups_, 1, 1);
            }
        }
        // store_set_ (not ensure_volume): lazy creation submits its own
        // OneShot, which must never run while THIS cb is recording (the
        // shared OneShot pool would be reset under it). The first bridge
        // always goes through write_psi_to_volume, which creates the volume
        // outside any recording.
        if (bridge && store_set_ != VK_NULL_HANDLE) {
            barrier_compute_to_compute(cb);
            transition_volume(cb, VK_IMAGE_LAYOUT_GENERAL);
            bridge_store_.bind(cb, store_set_);
            vkCmdDispatch(cb, mul_groups_, 1, 1);
            transition_volume(cb,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    }

    void apply_mask(int handle) {
        State* st = state_at(handle);
        if (st == nullptr || st->is_half) {
            return;
        }
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return;
        }
        mul_.bind(shot.cb(), st->mul_set);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
    }

    // GPU-reduced norm (sum |psi|^2 dV) and raw peak density.
    NormPeak norm_and_peak() {
        NormPeak out;
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return out;
        }
        norm_.bind(shot.cb(), norm_set_);
        vkCmdDispatch(shot.cb(), kGroups, 1, 1);
        record_partials_readback(shot.cb());
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return out;
        }
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* p = static_cast<const float*>(staging_.mapped);
        for (std::uint32_t g = 0; g < kGroups; ++g) {
            out.sum += p[2 * g];
            out.peak = std::max(out.peak, static_cast<double>(p[2 * g + 1]));
        }
        out.sum *= cell_volume_;
        return out;
    }

    // ---- SSBO -> 3D volume texture bridge (the renderer feed) -----------
    // The RG32F volume the renderer samples (re, im). Created lazily;
    // the renderer consumes volume_view() directly (FrameInput::psi_volume).
    VkImage volume_image() {
        return ensure_volume() ? volume_.img : VK_NULL_HANDLE;
    }
    VkImageView volume_view() {
        return ensure_volume() ? volume_.view : VK_NULL_HANDLE;
    }

    // Copy psi into the volume (imageStore, one texel per cell). The engine
    // OWNS the layout round-trip: the store runs in GENERAL, then the image
    // is handed to SHADER_READ_ONLY_OPTIMAL -- the layout the renderer's
    // sampled-image descriptors declare for it. Nobody else may transition
    // this image.
    bool write_psi_to_volume() {
        if (!ensure_volume()) {
            return false;
        }
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return false;
        }
        barrier_compute_to_compute(shot.cb());  // psi vs prior compute
        transition_volume(shot.cb(), VK_IMAGE_LAYOUT_GENERAL);
        bridge_store_.bind(shot.cb(), store_set_);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        transition_volume(shot.cb(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        return true;
    }

    // psi -> volume -> scratch SSBO -> host; bit-exact iff imageStore/
    // imageLoad round-trip the RG32F texel losslessly (check only).
    bool bridge_roundtrip(std::vector<float>& out) {
        if (!write_psi_to_volume()) {
            return false;
        }
        if (scratch_bridge_.buf == VK_NULL_HANDLE) {
            if (!ctx_->create_device_buffer(field_bytes_, &scratch_bridge_)) {
                return false;
            }
            load_set_ = arena_.allocate(*ctx_, bridge_load_.set_layout());
            if (load_set_ == VK_NULL_HANDLE) {
                return false;
            }
            arena_.write_buffer(*ctx_, load_set_, 0,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                scratch_bridge_.buf);
            arena_.write_image(*ctx_, load_set_, 1, volume_.view);
            arena_.write_buffer(*ctx_, load_set_, 2,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                bridge_ubo_.buf, sizeof(BridgeParams));
        }
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return false;
        }
        barrier_compute_to_compute(shot.cb());  // image written by the store
        transition_volume(shot.cb(), VK_IMAGE_LAYOUT_GENERAL);  // imageLoad
        bridge_load_.bind(shot.cb(), load_set_);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        record_buffer_readback(shot.cb(), scratch_bridge_, field_bytes_);
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

    // ---- semiclassical radiation (Ehrenfest mean force) -----------------
    // Upload grad V (central differences on the periodic grid, packed vec4)
    // so mean_force can reduce against it.
    bool set_potential_gradient(const std::vector<double>& v) {
        const int nx = grid_.x.n;
        const int ny = grid_.y.n;
        const int nz = grid_.z.n;
        const double i2hx = 1.0 / (2.0 * grid_.x.spacing());
        const double i2hy = 1.0 / (2.0 * grid_.y.spacing());
        const double i2hz = 1.0 / (2.0 * grid_.z.spacing());
        std::vector<float> packed(4 * cells_, 0.0f);
        for (int k = 0; k < nz; ++k) {
            const int kp = (k + 1) % nz;
            const int km = (k - 1 + nz) % nz;
            for (int j = 0; j < ny; ++j) {
                const int jp = (j + 1) % ny;
                const int jm = (j - 1 + ny) % ny;
                for (int i = 0; i < nx; ++i) {
                    const int ip = (i + 1) % nx;
                    const int im = (i - 1 + nx) % nx;
                    const std::size_t idx =
                        static_cast<std::size_t>(grid_.flat(i, j, k));
                    packed[4 * idx + 0] = static_cast<float>(
                        (v[static_cast<std::size_t>(grid_.flat(ip, j, k))] -
                         v[static_cast<std::size_t>(grid_.flat(im, j, k))]) *
                        i2hx);
                    packed[4 * idx + 1] = static_cast<float>(
                        (v[static_cast<std::size_t>(grid_.flat(i, jp, k))] -
                         v[static_cast<std::size_t>(grid_.flat(i, jm, k))]) *
                        i2hy);
                    packed[4 * idx + 2] = static_cast<float>(
                        (v[static_cast<std::size_t>(grid_.flat(i, j, kp))] -
                         v[static_cast<std::size_t>(grid_.flat(i, j, km))]) *
                        i2hz);
                }
            }
        }
        if (grad_buf_.buf == VK_NULL_HANDLE) {
            if (!ctx_->create_device_buffer(4 * cells_ * sizeof(float),
                                            &grad_buf_) ||
                !ctx_->create_device_buffer(4 * kGroups * sizeof(float),
                                            &force_partials_)) {
                return false;
            }
            force_set_ = arena_.allocate(*ctx_, force_.set_layout());
            if (force_set_ == VK_NULL_HANDLE) {
                return false;
            }
            const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            arena_.write_buffer(*ctx_, force_set_, 0, storage, psi_.buf);
            arena_.write_buffer(*ctx_, force_set_, 1,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                muln_ubo_.buf, sizeof(MulParams));
            arena_.write_buffer(*ctx_, force_set_, 2, storage,
                                force_partials_.buf);
            arena_.write_buffer(*ctx_, force_set_, 4, storage, grad_buf_.buf);
        }
        return upload_raw(grad_buf_, packed.data(),
                          packed.size() * sizeof(float));
    }

    // <grad V> = sum |psi|^2 grad V * dV -- the Ehrenfest dipole
    // acceleration. Zero if no gradient was uploaded.
    ses::Vec3d mean_force() {
        if (force_set_ == VK_NULL_HANDLE) {
            return ses::Vec3d{};
        }
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return ses::Vec3d{};
        }
        force_.bind(shot.cb(), force_set_);
        vkCmdDispatch(shot.cb(), kGroups, 1, 1);
        record_buffer_readback(shot.cb(), force_partials_,
                               4 * kGroups * sizeof(float));
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return ses::Vec3d{};
        }
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* p = static_cast<const float*>(staging_.mapped);
        double gx = 0.0;
        double gy = 0.0;
        double gz = 0.0;
        for (std::uint32_t g = 0; g < kGroups; ++g) {
            gx += p[4 * g + 0];
            gy += p[4 * g + 1];
            gz += p[4 * g + 2];
        }
        return ses::Vec3d{gx * cell_volume_, gy * cell_volume_,
                          gz * cell_volume_};
    }

    // ---- orbital synthesis ----------------------------------------------
    // psi <- normalized (u(|r|)/|r|) Y_lm, synthesized on the GPU from the
    // radial table u_nl(r). h_radial = rmax/(n_radial+1).
    bool synthesize_into_psi(const std::vector<double>& u, int l, int m,
                             double h_radial, double rmax, int n_radial) {
        return synthesize_into_buffer(psi_, u, l, m, h_radial, rmax, n_radial);
    }

    // Synthesize a normalized fp32 eigenstate into its OWN resident state
    // buffer (the atlas path; psi is untouched) and return a handle.
    // *out_peak gets the normalized peak |psi|^2; *out_norm2 the
    // PRE-normalization grid norm (the projection population normalizer).
    int synthesize_state(const std::vector<double>& u, int l, int m,
                         double h_radial, double rmax, int n_radial,
                         double* out_peak = nullptr,
                         double* out_norm2 = nullptr) {
        const int handle = create_state_buffer_uninit();
        if (handle < 0) {
            return -1;
        }
        State* st = state_at(handle);
        if (!synthesize_into_buffer(st->buf, u, l, m, h_radial, rmax,
                                    n_radial, out_peak, out_norm2)) {
            ctx_->destroy_buffer(&st->buf);
            return -1;
        }
        return handle;
    }

    // Synthesize + normalize in fp32 (the tested path), then pack to fp16
    // storage (cells uints, half the footprint) and return an fp16 handle.
    // Consumers unpack fp16 to scratch on demand (decode-on-use).
    int synthesize_state_half(const std::vector<double>& u, int l, int m,
                              double h_radial, double rmax, int n_radial,
                              double* out_peak = nullptr,
                              double* out_norm2 = nullptr) {
        if (!ensure_fp16()) {
            return -1;
        }
        Buffer tmp{};
        if (!ctx_->create_device_buffer(field_bytes_, &tmp)) {
            return -1;
        }
        if (!synthesize_into_buffer(tmp, u, l, m, h_radial, rmax, n_radial,
                                    out_peak, out_norm2)) {
            ctx_->destroy_buffer(&tmp);
            return -1;
        }
        State st;
        st.is_half = true;
        if (!ctx_->create_device_buffer(cells_ * sizeof(std::uint32_t),
                                        &st.buf)) {
            ctx_->destroy_buffer(&tmp);
            return -1;
        }
        // pack tmp(fp32, binding 0) -> st.buf(fp16, binding 6).
        const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        arena_.write_buffer(*ctx_, pack_set_, 0, storage, tmp.buf);
        arena_.write_buffer(*ctx_, pack_set_, 6, storage, st.buf.buf);
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            ctx_->destroy_buffer(&st.buf);
            ctx_->destroy_buffer(&tmp);
            return -1;
        }
        barrier_compute_to_compute(shot.cb());
        pack_.bind(shot.cb(), pack_set_);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        ctx_->destroy_buffer(&tmp);
        states_.push_back(st);
        return static_cast<int>(states_.size()) - 1;
    }

    // <to| r |from> = sum conj(to)*(x,y,z)*from * dV from two resident
    // states (fp32, or fp16 decoded to scratch), component-wise complex.
    ses::DipoleMatrixElement dipole_between(int to_h, int from_h) {
        State* to = state_at(to_h);
        State* from = state_at(from_h);
        if (to == nullptr || from == nullptr || !ensure_dipole()) {
            return {};
        }
        VkBuffer to_buf = decode(to, 0);
        VkBuffer from_buf = decode(from, 1);
        if (to_buf == VK_NULL_HANDLE || from_buf == VK_NULL_HANDLE) {
            return {};
        }
        DipoleParams dp{};
        dp.n = static_cast<std::uint32_t>(cells_);
        dp.nx = grid_.x.n;
        dp.ny = grid_.y.n;
        dp.box_min[0] = static_cast<float>(grid_.x.xmin);
        dp.box_min[1] = static_cast<float>(grid_.y.xmin);
        dp.box_min[2] = static_cast<float>(grid_.z.xmin);
        dp.cell_h[0] = static_cast<float>(grid_.x.spacing());
        dp.cell_h[1] = static_cast<float>(grid_.y.spacing());
        dp.cell_h[2] = static_cast<float>(grid_.z.spacing());
        std::memcpy(dipole_ubo_.mapped, &dp, sizeof(dp));
        vmaFlushAllocation(ctx_->allocator, dipole_ubo_.alloc, 0,
                           VK_WHOLE_SIZE);
        // The shared set is re-pointed per call (all submissions fence-wait,
        // so the set is never in flight when rewritten).
        const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        arena_.write_buffer(*ctx_, dipole_set_, 0, storage, to_buf);
        arena_.write_buffer(*ctx_, dipole_set_, 3, storage, from_buf);

        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return {};
        }
        dipole_.bind(shot.cb(), dipole_set_);
        vkCmdDispatch(shot.cb(), kGroups, 1, 1);
        record_buffer_readback(shot.cb(), dipole_partials_,
                               6 * kGroups * sizeof(float));
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return {};
        }
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* p = static_cast<const float*>(staging_.mapped);
        double d6[6] = {0, 0, 0, 0, 0, 0};
        for (std::uint32_t g = 0; g < kGroups; ++g) {
            for (int c = 0; c < 6; ++c) {
                d6[c] += p[6 * g + c];
            }
        }
        return ses::DipoleMatrixElement{
            ses::Complex<double>{d6[0] * cell_volume_, d6[1] * cell_volume_},
            ses::Complex<double>{d6[2] * cell_volume_, d6[3] * cell_volume_},
            ses::Complex<double>{d6[4] * cell_volume_, d6[5] * cell_volume_}};
    }

    // ---- orbital-free angular projection --------------------------------
    // Upload the static counting-sort geometry (ses::build_radial_bin_index)
    // and allocate g_lm[ncomp*nr]. Call once after the radial grid is fixed.
    bool set_projection_index(const std::vector<std::uint32_t>& sorted_cell,
                              const std::vector<std::uint32_t>& bin_off,
                              int n_radial, double h_radial, int l_max) {
        proj_nr_ = n_radial;
        proj_ncomp_ = (l_max + 1) * (l_max + 1);
        proj_h_radial_ = h_radial;
        const VkDeviceSize glm_bytes =
            2ull * proj_ncomp_ * proj_nr_ * sizeof(float);
        const ProjectParams pp0{};
        if (!ctx_->create_device_buffer(sorted_cell.size() * 4,
                                        &proj_sorted_buf_) ||
            !ctx_->create_device_buffer(bin_off.size() * 4, &proj_binoff_buf_) ||
            !ctx_->create_device_buffer(glm_bytes, &glm_buf_) ||
            !write_ubo(&proj_ubo_, &pp0, sizeof(pp0))) {
            return false;
        }
        if (!upload_raw(proj_sorted_buf_, sorted_cell.data(),
                        sorted_cell.size() * 4) ||
            !upload_raw(proj_binoff_buf_, bin_off.data(), bin_off.size() * 4)) {
            return false;
        }
        proj_set_ = arena_.allocate(*ctx_, project_.set_layout());
        if (proj_set_ == VK_NULL_HANDLE) {
            return false;
        }
        const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        arena_.write_buffer(*ctx_, proj_set_, 0, storage, psi_.buf);
        arena_.write_buffer(*ctx_, proj_set_, 1,
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, proj_ubo_.buf,
                            sizeof(ProjectParams));
        arena_.write_buffer(*ctx_, proj_set_, 6, storage, proj_sorted_buf_.buf);
        arena_.write_buffer(*ctx_, proj_set_, 7, storage, proj_binoff_buf_.buf);
        arena_.write_buffer(*ctx_, proj_set_, 8, storage, glm_buf_.buf);
        return true;
    }

    // Deposit psi -> g_lm (ONE grid pass, independent of state count), read
    // back to glm_host_ as double. Then call project_amplitude per state.
    void project_psi() {
        if (proj_set_ == VK_NULL_HANDLE) {
            return;  // set_projection_index failed/absent
        }
        ProjectParams pp{};
        pp.nx = grid_.x.n;
        pp.ny = grid_.y.n;
        pp.nr = proj_nr_;
        pp.h_radial = static_cast<float>(proj_h_radial_);
        pp.box_min[0] = static_cast<float>(grid_.x.xmin);
        pp.box_min[1] = static_cast<float>(grid_.y.xmin);
        pp.box_min[2] = static_cast<float>(grid_.z.xmin);
        pp.cell_h[0] = static_cast<float>(grid_.x.spacing());
        pp.cell_h[1] = static_cast<float>(grid_.y.spacing());
        pp.cell_h[2] = static_cast<float>(grid_.z.spacing());
        pp.dv = static_cast<float>(cell_volume_);
        std::memcpy(proj_ubo_.mapped, &pp, sizeof(pp));
        vmaFlushAllocation(ctx_->allocator, proj_ubo_.alloc, 0, VK_WHOLE_SIZE);

        const VkDeviceSize glm_bytes =
            2ull * proj_ncomp_ * proj_nr_ * sizeof(float);
        if (!ensure_staging(glm_bytes)) {
            return;
        }
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return;
        }
        project_.bind(shot.cb(), proj_set_);
        vkCmdDispatch(shot.cb(), static_cast<std::uint32_t>(proj_nr_), 1, 1);
        record_buffer_readback(shot.cb(), glm_buf_, glm_bytes);
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return;
        }
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* raw = static_cast<const float*>(staging_.mapped);
        glm_host_.assign(
            static_cast<std::size_t>(proj_ncomp_),
            std::vector<ses::Complex<double>>(static_cast<std::size_t>(proj_nr_)));
        for (int c = 0; c < proj_ncomp_; ++c) {
            for (int j = 0; j < proj_nr_; ++j) {
                const std::size_t o =
                    2 * (static_cast<std::size_t>(c) * proj_nr_ +
                         static_cast<std::size_t>(j));
                glm_host_[static_cast<std::size_t>(c)]
                         [static_cast<std::size_t>(j)] =
                             ses::Complex<double>{raw[o], raw[o + 1]};
            }
        }
    }

    // <n|psi> raw amplitude = sum_j u_nl[j] g_lm[lm(l,m)][j] (double CPU
    // finish). Needs a prior project_psi().
    ses::Complex<double> project_amplitude(const std::vector<double>& u, int l,
                                           int m) const {
        const std::size_t comp = static_cast<std::size_t>(l * l + (l + m));
        if (comp >= glm_host_.size()) {
            return {};
        }
        const std::vector<ses::Complex<double>>& gc = glm_host_[comp];
        ses::Complex<double> raw{};
        const int n = std::min(static_cast<int>(u.size()), proj_nr_);
        for (int j = 0; j < n; ++j) {
            raw += u[static_cast<std::size_t>(j)] * gc[static_cast<std::size_t>(j)];
        }
        return raw;
    }

    // psi <- s * psi (fp32 drift renormalization).
    void scale(float s) {
        const ConjParams sp{static_cast<std::uint32_t>(cells_), s, 0.0f, 0.0f};
        std::memcpy(scale_ubo_.mapped, &sp, sizeof(sp));
        vmaFlushAllocation(ctx_->allocator, scale_ubo_.alloc, 0, VK_WHOLE_SIZE);
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return;
        }
        scale_.bind(shot.cb(), scale_set_);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
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
        release_vkfft();
        for (State& st : states_) {
            ctx_->destroy_buffer(&st.buf);
        }
        states_.clear();
        arena_.destroy(*ctx_);
        unpack_.destroy(*ctx_);
        pack_.destroy(*ctx_);
        bridge_load_.destroy(*ctx_);
        bridge_store_.destroy(*ctx_);
        project_.destroy(*ctx_);
        dipole_.destroy(*ctx_);
        force_.destroy(*ctx_);
        synth_.destroy(*ctx_);
        copy_.destroy(*ctx_);
        axpy_.destroy(*ctx_);
        inner_.destroy(*ctx_);
        shear_.destroy(*ctx_);
        kick_.destroy(*ctx_);
        scale_.destroy(*ctx_);
        norm_.destroy(*ctx_);
        fft_.destroy(*ctx_);
        conj_.destroy(*ctx_);
        mul_.destroy(*ctx_);
        ctx_->destroy_buffer(&relax_kin_);
        ctx_->destroy_buffer(&relax_half_);
        ctx_->destroy_buffer(&decode_scratch_[1]);
        ctx_->destroy_buffer(&decode_scratch_[0]);
        ctx_->destroy_buffer(&scratch_bridge_);
        ctx_->destroy_buffer(&bridge_ubo_);
        ctx_->destroy_image(&volume_);
        ctx_->destroy_buffer(&proj_ubo_);
        ctx_->destroy_buffer(&glm_buf_);
        ctx_->destroy_buffer(&proj_binoff_buf_);
        ctx_->destroy_buffer(&proj_sorted_buf_);
        ctx_->destroy_buffer(&dipole_partials_);
        ctx_->destroy_buffer(&dipole_ubo_);
        ctx_->destroy_buffer(&force_partials_);
        ctx_->destroy_buffer(&grad_buf_);
        ctx_->destroy_buffer(&radial_buf_);
        ctx_->destroy_buffer(&synth_ubo_);
        ctx_->destroy_buffer(&axpy_ubo_);
        ctx_->destroy_buffer(&kick_ubo_);
        ctx_->destroy_buffer(&shear_ubo_[1]);
        ctx_->destroy_buffer(&shear_ubo_[0]);
        ctx_->destroy_buffer(&conjA_ubo_);
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
    // std140 {uint n; vec2 c}: c aligns to offset 8 (uint + 4 bytes pad).
    struct alignas(16) AxpyParams {
        std::uint32_t n, pad;
        float cx, cy;
    };
    struct alignas(16) KickParams {
        std::uint32_t n;
        std::int32_t nx, ny;
        float theta;
        float box_min[4];
        float cell_h[4];
        float axis[4];
    };
    // std140: n@0, nx@4, ny@8, pad@12, box_min@16, cell_h@32 (dipole.comp).
    struct alignas(16) DipoleParams {
        std::uint32_t n;
        std::int32_t nx, ny, pad0;
        float box_min[4];
        float cell_h[4];
    };
    // std140: nx@0, ny@4, nr@8, h_radial@12, box_min@16, cell_h@32, dv@48.
    struct alignas(16) ProjectParams {
        std::int32_t nx, ny, nr;
        float h_radial;
        float box_min[4];
        float cell_h[4];
        float dv, pad0, pad1, pad2;
    };
    // std140: vec4-padded box_min/cell_h at 16/32, matches synth.comp order.
    struct alignas(16) SynthParams {
        std::uint32_t n;
        std::int32_t nx, ny, l;
        float box_min[4];
        float cell_h[4];
        std::int32_t m, n_radial;
        float h_radial, rmax;
    };
    struct alignas(16) BridgeParams {
        std::int32_t nx, ny, nz, pad;
    };
    // std140 all-scalar block (tight 4-byte packing), matches shear.comp;
    // the trailing pad makes the 16-byte alignment explicit (dodges C4324).
    struct alignas(16) ShearParams {
        std::uint32_t n;
        std::int32_t nx, ny, nz;
        std::int32_t freq_axis, coord_axis, nf;
        float kscale, cmin, ch, coeff;
        float pad0;
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
        void dispatch_dyn(const Kernel& k, VkDescriptorSet set,
                          std::uint32_t groups, std::uint32_t offset) {
            if (!first) {
                barrier_compute_to_compute(cb);
            }
            first = false;
            k.bind(cb, set, offset);
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
        if (!ensure_staging(bytes)) {
            return false;
        }
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

    // A resident state: its buffer (fp32, or fp16-packed at half footprint)
    // + the four per-op descriptor sets (fp32 states only; fp16 consumers
    // decode to scratch and go through the shared any-target sets).
    struct State {
        bool is_half = false;
        Buffer buf{};
        VkDescriptorSet inner_set = VK_NULL_HANDLE;
        VkDescriptorSet axpy_set = VK_NULL_HANDLE;
        VkDescriptorSet copy_set = VK_NULL_HANDLE;
        VkDescriptorSet mul_set = VK_NULL_HANDLE;
    };

    State* state_at(int handle) {
        if (handle < 0 || handle >= static_cast<int>(states_.size())) {
            return nullptr;
        }
        State* st = &states_[static_cast<std::size_t>(handle)];
        return (st->buf.buf == VK_NULL_HANDLE) ? nullptr : st;
    }

    // Record: compute -> transfer edge, src -> staging copy, host edge.
    void record_buffer_readback(VkCommandBuffer cb, const Buffer& src,
                                VkDeviceSize bytes) {
        barrier_compute_to_transfer(cb);
        const VkBufferCopy down{0, 0, bytes};
        vkCmdCopyBuffer(cb, src.buf, staging_.buf, 1, &down);
        barrier_transfer_to_host(cb);
    }
    void record_partials_readback(VkCommandBuffer cb) {
        record_buffer_readback(cb, partials_, 2 * kGroups * sizeof(float));
    }

    // Allocate a state's buffer + descriptor sets without uploading content
    // (create_state_buffer fills it from the host; synthesize_state on GPU).
    int create_state_buffer_uninit() {
        State st;
        if (!ctx_->create_device_buffer(field_bytes_, &st.buf)) {
            return -1;
        }
        st.inner_set = arena_.allocate(*ctx_, inner_.set_layout());
        st.axpy_set = arena_.allocate(*ctx_, axpy_.set_layout());
        st.copy_set = arena_.allocate(*ctx_, copy_.set_layout());
        st.mul_set = arena_.allocate(*ctx_, mul_.set_layout());
        if (st.inner_set == VK_NULL_HANDLE || st.axpy_set == VK_NULL_HANDLE ||
            st.copy_set == VK_NULL_HANDLE || st.mul_set == VK_NULL_HANDLE) {
            ctx_->destroy_buffer(&st.buf);
            return -1;
        }
        const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        const auto uniform = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        arena_.write_buffer(*ctx_, st.inner_set, 0, storage, psi_.buf);
        arena_.write_buffer(*ctx_, st.inner_set, 1, uniform, muln_ubo_.buf,
                            sizeof(MulParams));
        arena_.write_buffer(*ctx_, st.inner_set, 2, storage, partials_.buf);
        arena_.write_buffer(*ctx_, st.inner_set, 3, storage, st.buf.buf);
        arena_.write_buffer(*ctx_, st.axpy_set, 0, storage, psi_.buf);
        arena_.write_buffer(*ctx_, st.axpy_set, 1, uniform, axpy_ubo_.buf,
                            sizeof(AxpyParams));
        arena_.write_buffer(*ctx_, st.axpy_set, 3, storage, st.buf.buf);
        arena_.write_buffer(*ctx_, st.copy_set, 0, storage, psi_.buf);
        arena_.write_buffer(*ctx_, st.copy_set, 1, uniform, muln_ubo_.buf,
                            sizeof(MulParams));
        arena_.write_buffer(*ctx_, st.copy_set, 3, storage, st.buf.buf);
        arena_.write_buffer(*ctx_, st.mul_set, 0, storage, psi_.buf);
        arena_.write_buffer(*ctx_, st.mul_set, 1, storage, st.buf.buf);
        arena_.write_buffer(*ctx_, st.mul_set, 2, uniform, muln_ubo_.buf,
                            sizeof(MulParams));
        states_.push_back(st);
        return static_cast<int>(states_.size()) - 1;
    }

    // Synthesize (u/r)Ylm into `out` via the shared any-target sets (re-
    // pointed per call; every submission fence-waits so never in flight),
    // then grid-normalize it. Reports pre-norm grid norm + normalized peak.
    bool synthesize_into_buffer(Buffer& out, const std::vector<double>& u,
                                int l, int m, double h_radial, double rmax,
                                int n_radial, double* out_peak = nullptr,
                                double* out_norm2 = nullptr) {
        std::vector<float> uf(u.size());
        for (std::size_t i = 0; i < u.size(); ++i) {
            uf[i] = static_cast<float>(u[i]);
        }
        const VkDeviceSize rbytes = uf.size() * sizeof(float);
        if (radial_buf_.buf == VK_NULL_HANDLE || radial_bytes_ != rbytes) {
            vkDeviceWaitIdle(ctx_->device);
            ctx_->destroy_buffer(&radial_buf_);
            if (!ctx_->create_device_buffer(rbytes, &radial_buf_)) {
                return false;
            }
            radial_bytes_ = rbytes;
        }
        if (!upload_raw(radial_buf_, uf.data(), rbytes)) {
            return false;
        }

        SynthParams sp{};
        sp.n = static_cast<std::uint32_t>(cells_);
        sp.nx = grid_.x.n;
        sp.ny = grid_.y.n;
        sp.l = l;
        sp.m = m;
        sp.n_radial = n_radial;
        sp.h_radial = static_cast<float>(h_radial);
        sp.rmax = static_cast<float>(rmax);
        sp.box_min[0] = static_cast<float>(grid_.x.xmin);
        sp.box_min[1] = static_cast<float>(grid_.y.xmin);
        sp.box_min[2] = static_cast<float>(grid_.z.xmin);
        sp.cell_h[0] = static_cast<float>(grid_.x.spacing());
        sp.cell_h[1] = static_cast<float>(grid_.y.spacing());
        sp.cell_h[2] = static_cast<float>(grid_.z.spacing());
        std::memcpy(synth_ubo_.mapped, &sp, sizeof(sp));
        vmaFlushAllocation(ctx_->allocator, synth_ubo_.alloc, 0, VK_WHOLE_SIZE);

        const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        arena_.write_buffer(*ctx_, synth_any_set_, 0, storage, out.buf);
        arena_.write_buffer(*ctx_, synth_any_set_, 1,
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, synth_ubo_.buf,
                            sizeof(SynthParams));
        arena_.write_buffer(*ctx_, synth_any_set_, 5, storage, radial_buf_.buf);

        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return false;
        }
        synth_.bind(shot.cb(), synth_any_set_);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);

        const NormPeak np = normalize_buffer(out);
        if (out_norm2 != nullptr) {
            *out_norm2 = np.sum;
        }
        if (out_peak != nullptr) {
            *out_peak = (np.sum > 0.0) ? np.peak / np.sum : 0.0;
        }
        return true;
    }

    // Grid normalization of `buf` via the re-pointed any-target norm/scale
    // sets. Returns {pre-normalization grid norm (x dV), raw peak density}.
    NormPeak normalize_buffer(Buffer& buf) {
        NormPeak np;
        const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        arena_.write_buffer(*ctx_, norm_any_set_, 0, storage, buf.buf);
        arena_.write_buffer(*ctx_, norm_any_set_, 1,
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, muln_ubo_.buf,
                            sizeof(MulParams));
        arena_.write_buffer(*ctx_, norm_any_set_, 2, storage, partials_.buf);
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return np;
        }
        norm_.bind(shot.cb(), norm_any_set_);
        vkCmdDispatch(shot.cb(), kGroups, 1, 1);
        record_partials_readback(shot.cb());
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return np;
        }
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* p = static_cast<const float*>(staging_.mapped);
        for (std::uint32_t g = 0; g < kGroups; ++g) {
            np.sum += p[2 * g];
            np.peak = std::max(np.peak, static_cast<double>(p[2 * g + 1]));
        }
        np.sum *= cell_volume_;

        const ConjParams sp{static_cast<std::uint32_t>(cells_),
                            static_cast<float>((np.sum > 0.0)
                                                   ? 1.0 / std::sqrt(np.sum)
                                                   : 0.0),
                            0.0f, 0.0f};
        std::memcpy(scale_ubo_.mapped, &sp, sizeof(sp));
        vmaFlushAllocation(ctx_->allocator, scale_ubo_.alloc, 0, VK_WHOLE_SIZE);
        arena_.write_buffer(*ctx_, scale_any_set_, 0, storage, buf.buf);
        arena_.write_buffer(*ctx_, scale_any_set_, 1,
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, scale_ubo_.buf,
                            sizeof(ConjParams));
        OneShot s2;
        if (!s2.begin(*ctx_)) {
            return np;
        }
        scale_.bind(s2.cb(), scale_any_set_);
        vkCmdDispatch(s2.cb(), mul_groups_, 1, 1);
        s2.submit_and_wait(*ctx_);
        s2.destroy(*ctx_);
        return np;
    }

    // Lazily create the RG32F volume + the store side of the bridge: one
    // submission transitions UNDEFINED -> GENERAL for the compute stores
    // (write_psi_to_volume owns the GENERAL <-> SHADER_READ_ONLY round-trip
    // from then on).
    bool ensure_volume() {
        if (store_set_ != VK_NULL_HANDLE) {
            return true;
        }
        if (!ctx_->create_storage_image_3d(
                static_cast<std::uint32_t>(n_), static_cast<std::uint32_t>(n_),
                static_cast<std::uint32_t>(n_), VK_FORMAT_R32G32_SFLOAT,
                &volume_)) {
            return false;
        }
        const BridgeParams bp{grid_.x.n, grid_.y.n, grid_.z.n, 0};
        if (!write_ubo(&bridge_ubo_, &bp, sizeof(bp))) {
            return false;
        }
        store_set_ = arena_.allocate(*ctx_, bridge_store_.set_layout());
        if (store_set_ == VK_NULL_HANDLE) {
            return false;
        }
        arena_.write_buffer(*ctx_, store_set_, 0,
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, psi_.buf);
        arena_.write_image(*ctx_, store_set_, 1, volume_.view);
        arena_.write_buffer(*ctx_, store_set_, 2,
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, bridge_ubo_.buf,
                            sizeof(BridgeParams));
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return false;
        }
        image_layout_barrier(shot.cb(), volume_.img, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_ACCESS_SHADER_READ_BIT |
                                 VK_ACCESS_SHADER_WRITE_BIT);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        volume_layout_ = VK_IMAGE_LAYOUT_GENERAL;
        return true;
    }

    // Transition the volume to `to` if not already there. Conservative
    // compute+fragment stages/access both sides (the queue is a combined
    // graphics+compute family on every supported path).
    void transition_volume(VkCommandBuffer cb, VkImageLayout to) {
        if (volume_layout_ == to) {
            return;
        }
        const VkPipelineStageFlags stages =
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        const VkAccessFlags access =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        image_layout_barrier(cb, volume_.img, volume_layout_, to, stages,
                             access, stages, access);
        volume_layout_ = to;
    }

    // Lazily build the fp16 codec resources: two decode scratch buffers (a
    // 2-fp16-operand dipole needs both) + the shared re-pointed sets.
    bool ensure_fp16() {
        if (pack_set_ != VK_NULL_HANDLE) {
            return true;
        }
        if (!ctx_->create_device_buffer(field_bytes_, &decode_scratch_[0]) ||
            !ctx_->create_device_buffer(field_bytes_, &decode_scratch_[1])) {
            return false;
        }
        pack_set_ = arena_.allocate(*ctx_, pack_.set_layout());
        unpack_set_ = arena_.allocate(*ctx_, unpack_.set_layout());
        inner_any_set_ = arena_.allocate(*ctx_, inner_.set_layout());
        if (pack_set_ == VK_NULL_HANDLE || unpack_set_ == VK_NULL_HANDLE ||
            inner_any_set_ == VK_NULL_HANDLE) {
            pack_set_ = VK_NULL_HANDLE;
            return false;
        }
        const auto uniform = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        arena_.write_buffer(*ctx_, pack_set_, 1, uniform, muln_ubo_.buf,
                            sizeof(MulParams));
        arena_.write_buffer(*ctx_, unpack_set_, 1, uniform, muln_ubo_.buf,
                            sizeof(MulParams));
        return true;
    }

    // A readable fp32 buffer for `st`: its own buffer if fp32, else the fp16
    // content unpacked into decode_scratch_[slot] (0 or 1).
    VkBuffer decode(State* st, int slot) {
        if (!st->is_half) {
            return st->buf.buf;
        }
        if (!ensure_fp16()) {
            return VK_NULL_HANDLE;
        }
        const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        arena_.write_buffer(*ctx_, unpack_set_, 0, storage,
                            decode_scratch_[slot].buf);
        arena_.write_buffer(*ctx_, unpack_set_, 6, storage, st->buf.buf);
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return VK_NULL_HANDLE;
        }
        barrier_compute_to_compute(shot.cb());
        unpack_.bind(shot.cb(), unpack_set_);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        return decode_scratch_[slot].buf;
    }

    // Lazily build the dipole reduction resources (shared re-pointed set).
    bool ensure_dipole() {
        if (dipole_set_ != VK_NULL_HANDLE) {
            return true;
        }
        const DipoleParams dp0{};
        if (!ctx_->create_device_buffer(6 * kGroups * sizeof(float),
                                        &dipole_partials_) ||
            !write_ubo(&dipole_ubo_, &dp0, sizeof(dp0))) {
            return false;
        }
        dipole_set_ = arena_.allocate(*ctx_, dipole_.set_layout());
        if (dipole_set_ == VK_NULL_HANDLE) {
            return false;
        }
        arena_.write_buffer(*ctx_, dipole_set_, 1,
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, dipole_ubo_.buf,
                            sizeof(DipoleParams));
        arena_.write_buffer(*ctx_, dipole_set_, 2,
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            dipole_partials_.buf);
        return true;
    }

    // Grow-only staging capacity (large glm/radial transfers).
    bool ensure_staging(VkDeviceSize bytes) {
        if (bytes <= staging_bytes_) {
            return true;
        }
        vkDeviceWaitIdle(ctx_->device);
        ctx_->destroy_buffer(&staging_);
        if (!ctx_->create_host_buffer(bytes,
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      &staging_)) {
            return false;
        }
        staging_bytes_ = bytes;
        return true;
    }

    // psi -= (cre + i cim) * state (Gram-Schmidt subtract). fp32 states only.
    void subtract_projection(int handle, double cre, double cim) {
        State* st = state_at(handle);
        if (st == nullptr || st->is_half) {
            return;
        }
        const AxpyParams ap{static_cast<std::uint32_t>(cells_), 0,
                            static_cast<float>(cre), static_cast<float>(cim)};
        std::memcpy(axpy_ubo_.mapped, &ap, sizeof(ap));
        vmaFlushAllocation(ctx_->allocator, axpy_ubo_.alloc, 0, VK_WHOLE_SIZE);
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return;
        }
        axpy_.bind(shot.cb(), st->axpy_set);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
    }

    // Norm reduction + readback -> host finish -> 1/sqrt(norm) scale.
    RelaxStats renormalize_and_estimate() {
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return {};
        }
        norm_.bind(shot.cb(), norm_set_);
        vkCmdDispatch(shot.cb(), kGroups, 1, 1);
        record_partials_readback(shot.cb());
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return {};
        }
        return finish_renorm_from_staging();
    }

    // Host finish over partials ALREADY copied into staging_ (relax_step
    // records the reduction inside the step submission), then the
    // 1/sqrt(norm) scale.
    RelaxStats finish_renorm_from_staging() {
        RelaxStats stats;
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
        scale(static_cast<float>(inv));
        return stats;
    }

    KickParams make_kick_params(const ses::Vec3d& axis, double theta) const {
        KickParams kp{};
        kp.n = static_cast<std::uint32_t>(cells_);
        kp.nx = grid_.x.n;
        kp.ny = grid_.y.n;
        kp.theta = static_cast<float>(theta);
        kp.box_min[0] = static_cast<float>(grid_.x.xmin);
        kp.box_min[1] = static_cast<float>(grid_.y.xmin);
        kp.box_min[2] = static_cast<float>(grid_.z.xmin);
        kp.cell_h[0] = static_cast<float>(grid_.x.spacing());
        kp.cell_h[1] = static_cast<float>(grid_.y.spacing());
        kp.cell_h[2] = static_cast<float>(grid_.z.spacing());
        kp.axis[0] = static_cast<float>(axis.x);
        kp.axis[1] = static_cast<float>(axis.y);
        kp.axis[2] = static_cast<float>(axis.z);
        return kp;
    }

    // Grow the dynamic-offset kick UBO to `kicks` slots and re-point the
    // descriptor set at the new buffer (rewriting a fence-idle set is legal).
    bool ensure_kick_capacity(int kicks) {
        const VkDeviceSize need =
            static_cast<VkDeviceSize>(kick_stride_) * kicks;
        if (kick_ubo_.buf != VK_NULL_HANDLE && kick_slots_ >= kicks) {
            return true;
        }
        ctx_->destroy_buffer(&kick_ubo_);
        if (!ctx_->create_host_buffer(
                need, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &kick_ubo_)) {
            return false;
        }
        kick_slots_ = kicks;
        arena_.write_buffer(*ctx_, kick_set_, 1,
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                            kick_ubo_.buf, sizeof(KickParams));
        return true;
    }

    const ses::Grid1D& axis_grid(int a) const {
        return a == 0 ? grid_.x : (a == 1 ? grid_.y : grid_.z);
    }

    // Shearing lines along freq_axis (frequency space along freq_axis),
    // shift each line by coeff * (its coord_axis coordinate).
    ShearParams make_shear_params(int freq_axis, int coord_axis,
                                  double coeff) const {
        const ses::Grid1D& fa = axis_grid(freq_axis);
        const ses::Grid1D& ca = axis_grid(coord_axis);
        const double two_pi = 6.283185307179586;
        ShearParams sp{};
        sp.n = static_cast<std::uint32_t>(cells_);
        sp.nx = grid_.x.n;
        sp.ny = grid_.y.n;
        sp.nz = grid_.z.n;
        sp.freq_axis = freq_axis;
        sp.coord_axis = coord_axis;
        sp.nf = fa.n;
        sp.kscale = static_cast<float>(two_pi / (fa.xmax - fa.xmin));
        sp.cmin = static_cast<float>(ca.xmin);
        sp.ch = static_cast<float>(ca.spacing());
        sp.coeff = static_cast<float>(coeff);
        return sp;
    }

    // Write the two shear parameter sets one (or many) three-shear
    // rotation(s) about the axis perpendicular to (b, c) need: set0 =
    // (b, c, -tan(theta/2)) used twice, set1 = (c, b, sin(theta)).
    void stage_rotation_ubos(int b, int c, double theta) {
        const double t = std::tan(0.5 * theta);
        const double sn = std::sin(theta);
        const ShearParams s0 = make_shear_params(b, c, -t);
        const ShearParams s1 = make_shear_params(c, b, sn);
        std::memcpy(shear_ubo_[0].mapped, &s0, sizeof(s0));
        std::memcpy(shear_ubo_[1].mapped, &s1, sizeof(s1));
        vmaFlushAllocation(ctx_->allocator, shear_ubo_[0].alloc, 0,
                           VK_WHOLE_SIZE);
        vmaFlushAllocation(ctx_->allocator, shear_ubo_[1].alloc, 0,
                           VK_WHOLE_SIZE);
    }

    // One staged shear: FFT along freq_axis, phase-shift (shear_set_[which]),
    // then the inverse FFT along that axis (conj -> FFT -> conj/n).
    void record_shear(Recorder& r, int which, int freq_axis) {
        const std::uint32_t nn = static_cast<std::uint32_t>(n_) *
                                 static_cast<std::uint32_t>(n_);
        r.dispatch(fft_, fft_set_[freq_axis], nn);
        r.dispatch(shear_, shear_set_[which], mul_groups_);
        r.dispatch(conj_, conj1_set_, mul_groups_);
        r.dispatch(fft_, fft_set_[freq_axis], nn);
        r.dispatch(conj_, conjA_set_, mul_groups_);
    }

    // One full three-shear rotation (set0 on b, set1 on c, set0 on b).
    void record_rotation(Recorder& r, int b, int c) {
        record_shear(r, 0, b);
        record_shear(r, 1, c);
        record_shear(r, 0, b);
    }

    // halfV . IFFT . kin . FFT . halfV. With VkFFT active the two whole-3D
    // transforms are single VkFFTAppend blocks (coalesced transposes; the
    // inverse carries the 1/N normalize), replacing six line-FFT dispatches
    // and both conj-scale dispatches. Else the hand-rolled chain (the
    // inverse FFT = conj . FFT . conj/N), barriers explicit either way.
    void run_step_body(Recorder& r, VkDescriptorSet half_set,
                       VkDescriptorSet kin_set) {
#ifdef SES_HAVE_VKFFT
        if (vkfft_active()) {
            r.dispatch(mul_, half_set, mul_groups_);
            record_vkfft(r, -1);  // forward
            r.dispatch(mul_, kin_set, mul_groups_);
            record_vkfft(r, 1);   // inverse, normalized 1/N by the plan
            r.dispatch(mul_, half_set, mul_groups_);
            return;
        }
#endif
        r.dispatch(mul_, half_set, mul_groups_);
        fft3(r);
        r.dispatch(mul_, kin_set, mul_groups_);
        r.dispatch(conj_, conj1_set_, mul_groups_);
        fft3(r);
        r.dispatch(conj_, conjN_set_, mul_groups_);
        r.dispatch(mul_, half_set, mul_groups_);
    }

#ifdef SES_HAVE_VKFFT
    // Plan a 3D C2C fp32 transform IN PLACE on psi_'s VkBuffer. The
    // configuration stores POINTERS; the pointees are members (or live in
    // the outliving DeviceContext) so VkFFTAppend can dereference per call.
    bool ensure_vkfft() {
        if (vkfft_ready_ || vkfft_failed_) {
            return vkfft_ready_;
        }
        vkfft_failed_ = true;
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = ctx_->queue_family;
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateCommandPool(ctx_->device, &pool_info, nullptr,
                                &vkfft_pool_) != VK_SUCCESS ||
            vkCreateFence(ctx_->device, &fence_info, nullptr, &vkfft_fence_) !=
                VK_SUCCESS) {
            release_vkfft();
            return false;
        }
        vkfft_psi_buf_ = psi_.buf;
        vkfft_buf_size_ = static_cast<std::uint64_t>(field_bytes_);
        VkFFTConfiguration conf{};
        conf.FFTdim = 3;
        conf.size[0] = static_cast<std::uint64_t>(grid_.x.n);
        conf.size[1] = static_cast<std::uint64_t>(grid_.y.n);
        conf.size[2] = static_cast<std::uint64_t>(grid_.z.n);
        conf.physicalDevice = &ctx_->phys_dev;
        conf.device = &ctx_->device;
        conf.queue = &ctx_->queue;
        conf.commandPool = &vkfft_pool_;
        conf.fence = &vkfft_fence_;
        conf.buffer = &vkfft_psi_buf_;
        conf.bufferSize = &vkfft_buf_size_;
        conf.normalize = 1;  // inverse divides by N (replaces the conj/N pass)
        const VkFFTResult res = initializeVkFFT(&vkfft_app_, conf);
        if (res != VKFFT_SUCCESS) {
            std::fprintf(stderr,
                         "ses_vk::Engine: initializeVkFFT = %d -- staying on "
                         "the hand-rolled FFT\n",
                         static_cast<int>(res));
            release_vkfft();
            return false;
        }
        std::fprintf(stderr, "ses_vk::Engine: VkFFT 3D plan active (%dx%dx%d)\n",
                     grid_.x.n, grid_.y.n, grid_.z.n);
        vkfft_ready_ = true;
        vkfft_failed_ = false;
        return true;
    }

    void release_vkfft() {
        if (vkfft_ready_) {
            deleteVkFFT(&vkfft_app_);
            vkfft_ready_ = false;
        }
        if (vkfft_fence_ != VK_NULL_HANDLE) {
            vkDestroyFence(ctx_->device, vkfft_fence_, nullptr);
            vkfft_fence_ = VK_NULL_HANDLE;
        }
        if (vkfft_pool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(ctx_->device, vkfft_pool_, nullptr);
            vkfft_pool_ = VK_NULL_HANDLE;
        }
    }

    // One whole-3D transform straight into the current primary command
    // buffer, hazard-fenced on both sides (VkFFT records plain dispatches).
    // direction: -1 forward, 1 inverse (1/N). After the block the recorder
    // is told a barrier was just emitted, so the next dispatch skips its own.
    void record_vkfft(Recorder& r, int direction) {
        if (!r.first) {
            barrier_compute_to_compute(r.cb);
        }
        VkCommandBuffer cb = r.cb;
        VkFFTLaunchParams lp{};
        lp.buffer = &vkfft_psi_buf_;
        lp.commandBuffer = &cb;
        const VkFFTResult res = VkFFTAppend(&vkfft_app_, direction, &lp);
        if (res != VKFFT_SUCCESS) {
            std::fprintf(stderr, "ses_vk::Engine: VkFFTAppend = %d\n",
                         static_cast<int>(res));
        }
        barrier_compute_to_compute(r.cb);
        r.first = true;  // the trailing barrier covers the next dispatch
    }
#else
    bool ensure_vkfft() { return false; }
    void release_vkfft() {}
#endif  // SES_HAVE_VKFFT

    void fft3(Recorder& r) {
        const std::uint32_t nn = static_cast<std::uint32_t>(n_) *
                                 static_cast<std::uint32_t>(n_);
        for (int a = 0; a < 3; ++a) {
            r.dispatch(fft_, fft_set_[a], nn);
        }
    }

    DeviceContext* ctx_ = nullptr;
    ses::Grid3D grid_{};
    int n_ = 0;
    std::size_t cells_ = 0;
    std::uint32_t mul_groups_ = 0;
    VkDeviceSize field_bytes_ = 0;
    double cell_volume_ = 1.0;
    double dtau_ = 0.0;
    std::uint32_t kick_stride_ = 256;
    int kick_slots_ = 0;

    Kernel mul_;
    Kernel conj_;
    Kernel fft_;
    Kernel norm_;
    Kernel scale_;
    Kernel kick_;
    Kernel shear_;
    Kernel inner_;
    Kernel axpy_;
    Kernel copy_;
    Kernel synth_;
    Kernel force_;
    Kernel dipole_;
    Kernel project_;
    Kernel bridge_store_;
    Kernel bridge_load_;
    Kernel pack_;
    Kernel unpack_;
    DescriptorArena arena_;
    DeviceContext::Image volume_{};
    VkImageLayout volume_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    Buffer decode_scratch_[2]{};
    std::vector<State> states_;
    VkDeviceSize staging_bytes_ = 0;
    VkDeviceSize radial_bytes_ = 0;
    int proj_nr_ = 0;
    int proj_ncomp_ = 0;
    double proj_h_radial_ = 0.0;
    std::vector<std::vector<ses::Complex<double>>> glm_host_;

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
    Buffer conjA_ubo_{};
    Buffer shear_ubo_[2]{};
    Buffer kick_ubo_{};
    Buffer axpy_ubo_{};
    Buffer synth_ubo_{};
    Buffer radial_buf_{};
    Buffer grad_buf_{};
    Buffer force_partials_{};
    Buffer dipole_ubo_{};
    Buffer dipole_partials_{};
    Buffer proj_sorted_buf_{};
    Buffer proj_binoff_buf_{};
    Buffer glm_buf_{};
    Buffer proj_ubo_{};
    Buffer bridge_ubo_{};
    Buffer scratch_bridge_{};
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
    VkDescriptorSet conjA_set_ = VK_NULL_HANDLE;
    VkDescriptorSet shear_set_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet kick_set_ = VK_NULL_HANDLE;
    VkDescriptorSet relax_half_set_ = VK_NULL_HANDLE;
    VkDescriptorSet relax_kin_set_ = VK_NULL_HANDLE;
    VkDescriptorSet force_set_ = VK_NULL_HANDLE;
    VkDescriptorSet dipole_set_ = VK_NULL_HANDLE;
    VkDescriptorSet proj_set_ = VK_NULL_HANDLE;
    VkDescriptorSet store_set_ = VK_NULL_HANDLE;
    VkDescriptorSet load_set_ = VK_NULL_HANDLE;
    VkDescriptorSet pack_set_ = VK_NULL_HANDLE;
    VkDescriptorSet unpack_set_ = VK_NULL_HANDLE;
    VkDescriptorSet inner_any_set_ = VK_NULL_HANDLE;
    VkDescriptorSet synth_any_set_ = VK_NULL_HANDLE;
    VkDescriptorSet norm_any_set_ = VK_NULL_HANDLE;
    VkDescriptorSet scale_any_set_ = VK_NULL_HANDLE;

    bool use_vkfft_ = true;
#ifdef SES_HAVE_VKFFT
    // VkFFT plan on psi_'s VkBuffer; pointees of the configuration.
    bool vkfft_ready_ = false;
    bool vkfft_failed_ = false;
    VkFFTApplication vkfft_app_{};
    VkBuffer vkfft_psi_buf_ = VK_NULL_HANDLE;
    std::uint64_t vkfft_buf_size_ = 0;
    VkCommandPool vkfft_pool_ = VK_NULL_HANDLE;
    VkFence vkfft_fence_ = VK_NULL_HANDLE;
#endif
};

}  // namespace ses_vk
