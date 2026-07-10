#pragma once

// QRhi (Vulkan-backend) analog of ses_gpu::GpuEngine (gpu_engine.hpp): the
// split-operator Strang step orchestrated on QRhi compute. The kernels are the
// same decorated Vulkan-GLSL that sesolver_qrhicheck bakes; the engine loads the
// baked .qsb from the Qt resource system, so the caller must bake them into its
// target (phase_multiply / conj_scale / fft_line<N>).
//
// Scope of this first cut: a CUBIC grid (nx == ny == nz == N) with a baked
// fft_line<N> shader -- enough to verify the step against SplitOperator3D. The
// psi buffer is a single Static VkBuffer (GPU-written in place, read back), which
// is also what a later VkFFT drop-in needs (one durable handle).

#ifdef SES_HAVE_VKFFT
// vulkan.h must be seen WITH prototypes before any Qt header pulls it in
// under VK_NO_PROTOTYPES (include guards would then hide the prototypes from
// header-only VkFFT, which calls vk* directly against the loader import lib).
#include <vulkan/vulkan.h>
#endif

#include <core/complex.hpp>
#include <core/decay.hpp>
#include <core/drive.hpp>
#include <core/grid.hpp>
#include <core/vec.hpp>

#include <rhi/qrhi.h>
#include <rhi/qrhi_platform.h>

#include <QFile>
#include <QString>

#ifdef SES_HAVE_VKFFT
#include <QVulkanFunctions>
#include <QVulkanInstance>
#include <VkFFT/vkFFT.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

namespace ses_qrhi {

inline std::vector<float> to_rg32f(const std::vector<ses::Complex<double>>& src) {
    std::vector<float> out(2 * src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        out[2 * i] = static_cast<float>(src[i].real());
        out[2 * i + 1] = static_cast<float>(src[i].imag());
    }
    return out;
}

inline QShader load_qsb(const QString& path) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) {
        std::fprintf(stderr, "QrhiEngine: cannot open %s\n", qPrintable(path));
        return {};
    }
    return QShader::fromSerialized(f.readAll());
}

class QrhiEngine {
public:
    QrhiEngine() = default;
    ~QrhiEngine() { release_vkfft(); }
    QrhiEngine(const QrhiEngine&) = delete;
    QrhiEngine& operator=(const QrhiEngine&) = delete;

    // Prefer VkFFT for the step body's 3D transforms when the plan built
    // (checks use this to also cover the hand-rolled path). No-op when built
    // without VkFFT support.
    void set_use_vkfft(bool on) {
#ifdef SES_HAVE_VKFFT
        use_vkfft_ = on;
#else
        (void)on;
#endif
    }
    bool vkfft_active() const {
#ifdef SES_HAVE_VKFFT
        return use_vkfft_ && vkfft_ready_;
#else
        return false;
#endif
    }

    // Destroy the raw Vulkan objects (the VkFFT plan + its pool/fence) while
    // the device is still known-alive. QRhiWidget::releaseResources() is the
    // right call site; the destructor also calls this as a backstop.
    void release_native() { release_vkfft(); }

    // Returns false if a shader is missing or a resource fails to build. half_v
    // / kinetic are SplitOperator3D's phase tables; psi0 the initial field.
    bool initialize(QRhi* rhi, const ses::Grid3D& grid,
                    const std::vector<ses::Complex<double>>& half_v,
                    const std::vector<ses::Complex<double>>& kinetic,
                    const std::vector<ses::Complex<double>>& psi0) {
        rhi_ = rhi;
        grid_ = grid;
        n_ = grid.x.n;
        cells_ = static_cast<std::size_t>(grid.size());
        cell_volume_ = grid.cell_volume();  // base resource: norm reduction needs dV
        if (grid.y.n != n_ || grid.z.n != n_) {
            std::fprintf(stderr, "QrhiEngine: only cubic grids supported\n");
            return false;
        }

        const QShader mulcs = load_qsb(QStringLiteral(":/shaders/phase_multiply.comp.qsb"));
        const QShader conjcs = load_qsb(QStringLiteral(":/shaders/conj_scale.comp.qsb"));
        const QShader fftcs =
            load_qsb(QStringLiteral(":/shaders/fft_line%1.comp.qsb").arg(n_));
        const QShader kickcs = load_qsb(QStringLiteral(":/shaders/dipole_kick.comp.qsb"));
        const QShader normcs = load_qsb(QStringLiteral(":/shaders/norm_peak.comp.qsb"));
        const QShader scalecs = load_qsb(QStringLiteral(":/shaders/scale.comp.qsb"));
        if (!mulcs.isValid() || !conjcs.isValid() || !fftcs.isValid() || !kickcs.isValid() ||
            !normcs.isValid() || !scalecs.isValid()) {
            return false;
        }

        const std::vector<float> psi_f = to_rg32f(psi0);
        const std::vector<float> half_f = to_rg32f(half_v);
        const std::vector<float> kin_f = to_rg32f(kinetic);
        const quint32 bytes = static_cast<quint32>(psi_f.size() * sizeof(float));

        psi_.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, bytes));
        half_.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, bytes));
        kin_.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, bytes));
        if (!psi_->create() || !half_->create() || !kin_->create()) { return false; }

        // UBO param blocks (std140). Values are constant; re-uploaded each frame
        // (Dynamic uniform buffers are per-frame-slot in QRhi).
        muln_ = MulParams{ static_cast<quint32>(cells_), 0, 0, 0 };
        conj1_ = ConjParams{ static_cast<quint32>(cells_), 1.0f, 0.0f, 0.0f };
        conjN_ = ConjParams{ static_cast<quint32>(cells_),
                             1.0f / static_cast<float>(cells_), 0.0f, 0.0f };
        const int nn = n_ * n_;
        fftp_[0] = FftParams{ nn, n_, 0, 1, nn, 0, 0, 0 };        // x-lines
        fftp_[1] = FftParams{ n_, 1, nn, n_, nn, 0, 0, 0 };       // y-lines
        fftp_[2] = FftParams{ nn, 1, 0, nn, nn, 0, 0, 0 };        // z-lines

        muln_ubo_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                        sizeof(MulParams)));
        conj1_ubo_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                         sizeof(ConjParams)));
        conjN_ubo_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                         sizeof(ConjParams)));
        if (!muln_ubo_->create() || !conj1_ubo_->create() || !conjN_ubo_->create()) {
            return false;
        }
        for (int a = 0; a < 3; ++a) {
            fft_ubo_[a].reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                              sizeof(FftParams)));
            if (!fft_ubo_[a]->create()) { return false; }
        }
        // Dipole-kick params live in a dynamic-offset UBO so a whole driven
        // batch records as one frame; slots are ubufAlignment() apart.
        kick_stride_ = static_cast<quint32>(rhi_->ubufAlignment());
        while (kick_stride_ < sizeof(KickParams)) { kick_stride_ *= 2; }
        if (!ensure_kick_capacity(2)) { return false; }

        // SRBs. phase_multiply: psi(0) rw, phase(1) ro, n(2) ubo. conj_scale:
        // psi(0) rw, params(1) ubo. fft_line: psi(0) rw, axis(1) ubo.
        using B = QRhiShaderResourceBinding;
        const auto cs = B::ComputeStage;
        mul_half_srb_.reset(rhi_->newShaderResourceBindings());
        mul_half_srb_->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                     B::bufferLoad(1, cs, half_.data()),
                                     B::uniformBuffer(2, cs, muln_ubo_.data()) });
        mul_kin_srb_.reset(rhi_->newShaderResourceBindings());
        mul_kin_srb_->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                    B::bufferLoad(1, cs, kin_.data()),
                                    B::uniformBuffer(2, cs, muln_ubo_.data()) });
        conj1_srb_.reset(rhi_->newShaderResourceBindings());
        conj1_srb_->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                  B::uniformBuffer(1, cs, conj1_ubo_.data()) });
        conjN_srb_.reset(rhi_->newShaderResourceBindings());
        conjN_srb_->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                  B::uniformBuffer(1, cs, conjN_ubo_.data()) });
        for (int a = 0; a < 3; ++a) {
            fft_srb_[a].reset(rhi_->newShaderResourceBindings());
            fft_srb_[a]->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                       B::uniformBuffer(1, cs, fft_ubo_[a].data()) });
        }
        if (!mul_half_srb_->create() || !mul_kin_srb_->create() || !conj1_srb_->create() ||
            !conjN_srb_->create() || !fft_srb_[0]->create() || !fft_srb_[1]->create() ||
            !fft_srb_[2]->create()) {
            return false;
        }

        mul_pipe_.reset(rhi_->newComputePipeline());
        mul_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, mulcs));
        mul_pipe_->setShaderResourceBindings(mul_half_srb_.data());
        conj_pipe_.reset(rhi_->newComputePipeline());
        conj_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, conjcs));
        conj_pipe_->setShaderResourceBindings(conj1_srb_.data());
        fft_pipe_.reset(rhi_->newComputePipeline());
        fft_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, fftcs));
        fft_pipe_->setShaderResourceBindings(fft_srb_[0].data());
        kick_pipe_.reset(rhi_->newComputePipeline());
        kick_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, kickcs));
        kick_pipe_->setShaderResourceBindings(kick_srb_.data());
        if (!mul_pipe_->create() || !conj_pipe_->create() || !fft_pipe_->create() ||
            !kick_pipe_->create()) {
            return false;
        }

        // Norm/peak reduction + scale are BASE resources (used by relax,
        // deflation, and orbital synthesis), so they live here, not in
        // set_relax_tables. partials_ holds one (sum,peak) per workgroup.
        partials_.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer,
                                        static_cast<quint32>(2 * kGroups * sizeof(float))));
        scale_ubo_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                         sizeof(ConjParams)));
        if (!partials_->create() || !scale_ubo_->create()) { return false; }
        norm_srb_.reset(rhi_->newShaderResourceBindings());
        norm_srb_->setBindings({ B::bufferLoad(0, cs, psi_.data()),
                                 B::uniformBuffer(1, cs, muln_ubo_.data()),
                                 B::bufferLoadStore(2, cs, partials_.data()) });
        scale_srb_.reset(rhi_->newShaderResourceBindings());
        scale_srb_->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                  B::uniformBuffer(1, cs, scale_ubo_.data()) });
        if (!norm_srb_->create() || !scale_srb_->create()) { return false; }
        norm_pipe_.reset(rhi_->newComputePipeline());
        norm_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, normcs));
        norm_pipe_->setShaderResourceBindings(norm_srb_.data());
        scale_pipe_.reset(rhi_->newComputePipeline());
        scale_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, scalecs));
        scale_pipe_->setShaderResourceBindings(scale_srb_.data());
        if (!norm_pipe_->create() || !scale_pipe_->create()) { return false; }

        // Upload the static buffers (psi/half/kin) once.
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return false; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->uploadStaticBuffer(psi_.data(), psi_f.data());
        u->uploadStaticBuffer(half_.data(), half_f.data());
        u->uploadStaticBuffer(kin_.data(), kin_f.data());
        cb->resourceUpdate(u);
        rhi_->endOffscreenFrame();

        // M4: plan the VkFFT 3D transform on psi's native VkBuffer (outside any
        // frame -- plan creation compiles shaders via glslang). On failure the
        // step body stays on the hand-rolled line FFT.
        ensure_vkfft();
        return true;
    }

    // psi <- (halfV . IFFT . kin . FFT . halfV)^nsteps psi, the split-operator
    // Strang step -- the same dispatch chain as ses_gpu::GpuEngine::step. All
    // dispatches share the psi buffer, so QRhi inserts the barriers between them.
    void step(int nsteps) {
        const int mul_groups = static_cast<int>((cells_ + 255) / 256);
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        update_step_ubos(u);
        cb->beginComputePass(u, QRhiCommandBuffer::ExternalContent);
        for (int s = 0; s < nsteps; ++s) {
            run_step_body(cb, mul_groups, mul_half_srb_.data(), mul_kin_srb_.data());
        }
        cb->endComputePass();
        rhi_->endOffscreenFrame();
    }

    // Driven Strang steps (T3): kick(t) . step . kick(t+dt), the tested
    // core/drive.hpp composition. theta = amplitude cos(omega t) dt/2. The
    // per-kick thetas differ within the batch, so the kick parameters are
    // packed into ONE dynamic-offset UBO (2*nsteps slots) and the WHOLE batch
    // records as a single offscreen frame -- per-step frame splits serialize
    // on GPU completion and halve the laser-path sim rate (GL parity).
    void driven_step(const ses::DipoleDrive& d, double t0, double dt, int nsteps) {
        const int mul_groups = static_cast<int>((cells_ + 255) / 256);
        const int kicks = 2 * nsteps;
        if (!ensure_kick_capacity(kicks)) { return; }
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        update_step_ubos(u);
        for (int s = 0; s < nsteps; ++s) {
            const double t = t0 + s * dt;
            KickParams kp = make_kick_params(
                d.axis, d.amplitude * std::cos(d.omega * t) * 0.5 * dt);
            u->updateDynamicBuffer(kick_ubo_.data(),
                                   static_cast<quint32>(2 * s) * kick_stride_,
                                   sizeof(KickParams), &kp);
            kp.theta = static_cast<float>(
                d.amplitude * std::cos(d.omega * (t + dt)) * 0.5 * dt);
            u->updateDynamicBuffer(kick_ubo_.data(),
                                   static_cast<quint32>(2 * s + 1) * kick_stride_,
                                   sizeof(KickParams), &kp);
        }
        cb->beginComputePass(u, QRhiCommandBuffer::ExternalContent);
        for (int s = 0; s < nsteps; ++s) {
            record_kick(cb, mul_groups, static_cast<quint32>(2 * s) * kick_stride_);
            run_step_body(cb, mul_groups, mul_half_srb_.data(), mul_kin_srb_.data());
            record_kick(cb, mul_groups, static_cast<quint32>(2 * s + 1) * kick_stride_);
        }
        cb->endComputePass();
        rhi_->endOffscreenFrame();
    }

    // Free-energy estimate + normalized peak density from the per-step
    // renormalization (GL RelaxStats parity: peak drives cloud brightness).
    struct RelaxStats { double energy = 0.0; double peak = 0.0; };
    // GPU-reduced norm (sum |psi|^2 dV) and raw peak density (GL NormPeak).
    struct NormPeak { double sum = 0.0; double peak = 0.0; };

    // Upload the imaginary-time weight tables (ImaginaryTimePropagator3D's, packed
    // vec2(w,0) for the complex-multiply kernel) and build the norm + scale
    // renormalization pipelines. Call after initialize(). cell_volume = grid dV.
    bool set_relax_tables(const std::vector<double>& half_w,
                          const std::vector<double>& kin_w, double dtau, double cell_volume) {
        dtau_ = dtau;
        cell_volume_ = cell_volume;
        const quint32 bytes = static_cast<quint32>(2 * cells_ * sizeof(float));
        std::vector<float> hf(2 * cells_, 0.0f);
        std::vector<float> kf(2 * cells_, 0.0f);
        for (std::size_t i = 0; i < cells_; ++i) {
            hf[2 * i] = static_cast<float>(half_w[i]);
            kf[2 * i] = static_cast<float>(kin_w[i]);
        }
        relax_half_.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, bytes));
        relax_kin_.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, bytes));
        if (!relax_half_->create() || !relax_kin_->create()) { return false; }

        using B = QRhiShaderResourceBinding;
        const auto cs = B::ComputeStage;
        relax_half_srb_.reset(rhi_->newShaderResourceBindings());
        relax_half_srb_->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                       B::bufferLoad(1, cs, relax_half_.data()),
                                       B::uniformBuffer(2, cs, muln_ubo_.data()) });
        relax_kin_srb_.reset(rhi_->newShaderResourceBindings());
        relax_kin_srb_->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                      B::bufferLoad(1, cs, relax_kin_.data()),
                                      B::uniformBuffer(2, cs, muln_ubo_.data()) });
        if (!relax_half_srb_->create() || !relax_kin_srb_->create()) { return false; }

        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return false; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->uploadStaticBuffer(relax_half_.data(), hf.data());
        u->uploadStaticBuffer(relax_kin_.data(), kf.data());
        cb->resourceUpdate(u);
        rhi_->endOffscreenFrame();
        return true;
    }

    // e^{-H dtau} Strang steps with per-step renormalization (imaginary time).
    // Each step: one frame for the imaginary Strang chain + the norm reduction
    // (read the partials back), then a frame that scales by 1/sqrt(norm). The
    // pre-renorm norm decays as e^{-2 E dtau}, so the last step yields the free
    // energy estimate. Mirrors ses_gpu::GpuEngine::relax_step.
    RelaxStats relax_step(int nsteps) {
        const int mul_groups = static_cast<int>((cells_ + 255) / 256);
        RelaxStats stats;
        for (int s = 0; s < nsteps; ++s) {
            QRhiCommandBuffer* cb = nullptr;
            if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return stats; }
            QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
            update_step_ubos(u);
            cb->beginComputePass(u, QRhiCommandBuffer::ExternalContent);
            run_step_body(cb, mul_groups, relax_half_srb_.data(), relax_kin_srb_.data());
            cb->endComputePass();
            rhi_->endOffscreenFrame();
            stats = renormalize_and_estimate(mul_groups);
        }
        return stats;
    }

    // Re-upload psi from a host field (reset the state between runs).
    void upload_state(const std::vector<ses::Complex<double>>& psi) {
        const std::vector<float> pf = to_rg32f(psi);
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->uploadStaticBuffer(psi_.data(), pf.data());
        cb->resourceUpdate(u);
        rhi_->endOffscreenFrame();
    }

    // Interleaved RG floats, 2 per cell.
    void readback(std::vector<float>& out) {
        const quint32 bytes = static_cast<quint32>(2 * cells_ * sizeof(float));
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return; }
        QRhiReadbackResult rb;
        QRhiResourceUpdateBatch* d = rhi_->nextResourceUpdateBatch();
        d->readBackBuffer(psi_.data(), 0, bytes, &rb);
        cb->resourceUpdate(d);
        rhi_->endOffscreenFrame();
        out.assign(reinterpret_cast<const float*>(rb.data.constData()),
                   reinterpret_cast<const float*>(rb.data.constData()) + 2 * cells_);
    }

    // ---- resident states (ONE handle space, mirroring GL's GLuint space) ----
    // Every resident state -- CPU-uploaded (absorber mask, deflation seeds) or
    // GPU-synthesized (atlas eigenstates) -- lives in atlas_ and shares one int
    // handle space, so deflation can run against GPU-synthesized states exactly
    // like the GL engine. Per-state SRBs (inner/axpy/copy/mul) build lazily.

    // Upload a CPU state into its own resident fp32 buffer; returns a handle
    // usable with EVERY per-state op, or -1 on failure.
    int create_state_buffer(const std::vector<ses::Complex<double>>& state) {
        const std::vector<float> sf = to_rg32f(state);
        auto a = std::make_unique<AtlasState>();
        a->buf.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer,
                                     static_cast<quint32>(sf.size() * sizeof(float))));
        if (!a->buf->create()) { return -1; }
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return -1; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->uploadStaticBuffer(a->buf.data(), sf.data());
        cb->resourceUpdate(u);
        rhi_->endOffscreenFrame();
        atlas_.push_back(std::move(a));
        return static_cast<int>(atlas_.size()) - 1;
    }

    // Free a resident state's GPU buffer (transient deflation/pair states). The
    // slot stays so other handles remain stable; per-state ops on a released
    // handle are no-ops / zeros.
    void release_state(int handle) {
        if (handle < 0 || handle >= static_cast<int>(atlas_.size())) { return; }
        AtlasState* a = atlas_[static_cast<std::size_t>(handle)].get();
        a->inner_srb.reset();
        a->axpy_srb.reset();
        a->copy_srb.reset();
        a->mul_srb.reset();
        a->buf.reset();
    }

    // <state|psi> = sum conj(state)*psi * dV, live psi vs resident state
    // `handle` (fp16 states are unpacked to scratch first, decode-on-use).
    ses::Complex<double> inner_with_psi(int handle) {
        AtlasState* a = state_at(handle);
        if (a == nullptr || !ensure_inner()) { return {}; }
        QRhiShaderResourceBindings* srb = nullptr;
        if (a->is_half) {
            // Rare path: decode to scratch, bind transiently.
            QRhiBuffer* sbuf = decode(handle, 0);
            if (sbuf == nullptr) { return {}; }
            using B = QRhiShaderResourceBinding;
            const auto cs = B::ComputeStage;
            QScopedPointer<QRhiShaderResourceBindings> isrb(rhi_->newShaderResourceBindings());
            isrb->setBindings({ B::bufferLoad(0, cs, psi_.data()),
                                B::uniformBuffer(1, cs, muln_ubo_.data()),
                                B::bufferLoadStore(2, cs, partials_.data()),
                                B::bufferLoad(3, cs, sbuf) });
            if (!isrb->create()) { return {}; }
            return finish_inner(isrb.data());
        }
        if (!ensure_state_ops(a)) { return {}; }
        srb = a->inner_srb.data();
        return finish_inner(srb);
    }

    // Deflated imaginary-time relax: each step runs the imaginary Strang body,
    // then Gram-Schmidt projects out every `lower` state (psi -= <phi|psi> phi),
    // then renormalizes. Mirrors GpuEngine::relax_deflated_step; the pre-renorm
    // norm gives the free-energy estimate of the next state up.
    RelaxStats relax_deflated_step(const std::vector<int>& lower, int nsteps) {
        const int mul_groups = static_cast<int>((cells_ + 255) / 256);
        RelaxStats stats;
        for (int s = 0; s < nsteps; ++s) {
            // Frame 1: imaginary Strang body.
            {
                QRhiCommandBuffer* cb = nullptr;
                if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return stats; }
                QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
                update_step_ubos(u);
                cb->beginComputePass(u, QRhiCommandBuffer::ExternalContent);
                run_step_body(cb, mul_groups, relax_half_srb_.data(), relax_kin_srb_.data());
                cb->endComputePass();
                rhi_->endOffscreenFrame();
            }
            // Sequential deflation: inner product then axpy, per lower state.
            for (int h : lower) {
                const ses::Complex<double> c = inner_with_psi(h);
                subtract_projection(h, c.real(), c.imag(), mul_groups);
            }
            // Frame N: norm reduction + readback, then Frame S: scale.
            stats = renormalize_and_estimate(mul_groups);
        }
        return stats;
    }

    // psi <- src, the resident state `handle` copied bitwise into psi (the
    // quantum-jump collapse path; GpuEngine::copy_into_psi via glCopyBufferSubData).
    void copy_into_psi(int handle) {
        AtlasState* a = state_at(handle);
        if (a == nullptr || a->is_half || !ensure_state_ops(a)) { return; }
        const int groups = static_cast<int>((cells_ + 255) / 256);
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->updateDynamicBuffer(muln_ubo_.data(), 0, sizeof(MulParams), &muln_);
        cb->beginComputePass(u);
        cb->setComputePipeline(copy_pipe_.data());
        cb->setShaderResources(a->copy_srb.data());
        cb->dispatch(groups, 1, 1);
        cb->endComputePass();
        rhi_->endOffscreenFrame();
    }

    // psi <- psi * state (elementwise complex multiply): the absorbing boundary
    // damp with the mask uploaded as a (mask, 0) state. GL apply_mask parity.
    void apply_mask(int handle) {
        AtlasState* a = state_at(handle);
        if (a == nullptr || a->is_half || !ensure_state_ops(a)) { return; }
        const int groups = static_cast<int>((cells_ + 255) / 256);
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->updateDynamicBuffer(muln_ubo_.data(), 0, sizeof(MulParams), &muln_);
        cb->beginComputePass(u);
        multiply(cb, a->mul_srb.data(), groups);
        cb->endComputePass();
        rhi_->endOffscreenFrame();
    }

    // GPU-reduced norm (sum |psi|^2 dV) and raw peak density -- a 2 KB readback
    // instead of the full field. GL norm_and_peak parity.
    NormPeak norm_and_peak() {
        const NormPeak raw = reduce_norm_peak();
        return NormPeak{ raw.sum * cell_volume_, raw.peak };
    }

    // psi <- s * psi (fp32 drift renormalization). GL scale parity.
    void scale(float s) {
        const int groups = static_cast<int>((cells_ + 255) / 256);
        const ConjParams sp{ static_cast<quint32>(cells_), s, 0.0f, 0.0f };
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->updateDynamicBuffer(scale_ubo_.data(), 0, sizeof(ConjParams), &sp);
        cb->beginComputePass(u);
        cb->setComputePipeline(scale_pipe_.data());
        cb->setShaderResources(scale_srb_.data());
        cb->dispatch(groups, 1, 1);
        cb->endComputePass();
        rhi_->endOffscreenFrame();
    }

    // ---- semiclassical radiation (Ehrenfest mean force) -----------------
    // Upload grad V (central differences on the periodic grid, packed vec4) so
    // mean_force can reduce against it. Rebuild when the potential changes.
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
                    const std::size_t idx = static_cast<std::size_t>(grid_.flat(i, j, k));
                    packed[4 * idx + 0] = static_cast<float>(
                        (v[static_cast<std::size_t>(grid_.flat(ip, j, k))] -
                         v[static_cast<std::size_t>(grid_.flat(im, j, k))]) * i2hx);
                    packed[4 * idx + 1] = static_cast<float>(
                        (v[static_cast<std::size_t>(grid_.flat(i, jp, k))] -
                         v[static_cast<std::size_t>(grid_.flat(i, jm, k))]) * i2hy);
                    packed[4 * idx + 2] = static_cast<float>(
                        (v[static_cast<std::size_t>(grid_.flat(i, j, kp))] -
                         v[static_cast<std::size_t>(grid_.flat(i, j, km))]) * i2hz);
                }
            }
        }
        if (!ensure_force()) { return false; }
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return false; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->uploadStaticBuffer(grad_v_buf_.data(), packed.data());
        cb->resourceUpdate(u);
        rhi_->endOffscreenFrame();
        return true;
    }

    // <grad V> = sum |psi|^2 grad V * dV -- the Ehrenfest dipole acceleration
    // for the semiclassical radiated power. Zero if no gradient was uploaded.
    ses::Vec3d mean_force() {
        // Gate on the last resource ensure_force builds: a partial failure
        // leaves earlier pointers set, and dispatching a null pipeline is UB.
        if (force_pipe_.isNull() || grad_v_buf_.isNull()) { return ses::Vec3d{}; }
        QRhiReadbackResult rb;
        run_reduce(force_pipe_.data(), force_srb_.data(), force_partials_.data(),
                   static_cast<quint32>(4 * kGroups * sizeof(float)), &rb);
        const float* p = reinterpret_cast<const float*>(rb.data.constData());
        double gx = 0.0;
        double gy = 0.0;
        double gz = 0.0;
        for (int gi = 0; gi < kGroups; ++gi) {
            gx += p[4 * gi + 0];
            gy += p[4 * gi + 1];
            gz += p[4 * gi + 2];
        }
        return ses::Vec3d{ gx * cell_volume_, gy * cell_volume_, gz * cell_volume_ };
    }

    // ---- orbital synthesis ---------------------------------------------
    // psi <- normalized (u(|r|)/|r|) Y_lm, synthesized on the GPU from the radial
    // table u_nl(r) (kSynthSrc). h_radial = rmax/(n_radial+1).
    bool synthesize_into_psi(const std::vector<double>& u, int l, int m, double h_radial,
                             double rmax, int n_radial) {
        return synthesize_into_buffer(psi_.data(), u, l, m, h_radial, rmax, n_radial);
    }

    // Synthesize a normalized fp32 eigenstate into its OWN resident buffer (the
    // atlas builds each state on the GPU, no CPU field) and return a handle.
    // *out_peak (if given) receives the normalized peak |psi|^2; *out_norm2 the
    // PRE-normalization grid norm (the projection population normalizer).
    int synthesize_state(const std::vector<double>& u, int l, int m, double h_radial,
                         double rmax, int n_radial, double* out_peak = nullptr,
                         double* out_norm2 = nullptr) {
        auto a = std::make_unique<AtlasState>();
        a->buf.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer,
                                     static_cast<quint32>(2 * cells_ * sizeof(float))));
        if (!a->buf->create()) { return -1; }
        if (!synthesize_into_buffer(a->buf.data(), u, l, m, h_radial, rmax, n_radial,
                                    out_peak, out_norm2)) {
            return -1;
        }
        atlas_.push_back(std::move(a));
        return static_cast<int>(atlas_.size()) - 1;
    }

    // Synthesize + normalize in fp32 (the tested path), then pack to fp16 storage
    // (cells uints, half the footprint) and return an fp16 handle. The fp32 temp
    // is freed. Consumers unpack fp16 to scratch on demand.
    int synthesize_state_half(const std::vector<double>& u, int l, int m, double h_radial,
                              double rmax, int n_radial, double* out_peak = nullptr,
                              double* out_norm2 = nullptr) {
        if (!ensure_fp16()) { return -1; }
        QScopedPointer<QRhiBuffer> tmp(rhi_->newBuffer(
            QRhiBuffer::Static, QRhiBuffer::StorageBuffer,
            static_cast<quint32>(2 * cells_ * sizeof(float))));
        if (!tmp->create()) { return -1; }
        if (!synthesize_into_buffer(tmp.data(), u, l, m, h_radial, rmax, n_radial,
                                    out_peak, out_norm2)) {
            return -1;
        }
        auto a = std::make_unique<AtlasState>();
        a->is_half = true;
        a->buf.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer,
                                     static_cast<quint32>(cells_ * sizeof(quint32))));
        if (!a->buf->create()) { return -1; }
        // pack tmp(fp32, binding 0) -> a->buf(fp16, binding 6).
        using B = QRhiShaderResourceBinding;
        const auto cs = B::ComputeStage;
        QScopedPointer<QRhiShaderResourceBindings> psrb(rhi_->newShaderResourceBindings());
        psrb->setBindings({ B::bufferLoad(0, cs, tmp.data()),
                            B::uniformBuffer(1, cs, muln_ubo_.data()),
                            B::bufferLoadStore(6, cs, a->buf.data()) });
        if (!psrb->create()) { return -1; }
        run_single(pack_pipe_.data(), psrb.data());
        atlas_.push_back(std::move(a));
        return static_cast<int>(atlas_.size()) - 1;
    }

    // <to| r |from> = sum conj(to)*(x,y,z)*from * dV, from two resident atlas
    // states (each fp32 or fp16-decoded). Component-wise complex.
    ses::DipoleMatrixElement dipole_between(int to_h, int from_h) {
        if (!ensure_dipole()) { return {}; }
        QRhiBuffer* to_buf = decode(to_h, 0);
        QRhiBuffer* from_buf = decode(from_h, 1);
        if (to_buf == nullptr || from_buf == nullptr) { return {}; }
        DipoleParams dp{};
        dp.n = static_cast<quint32>(cells_);
        dp.nx = grid_.x.n;
        dp.ny = grid_.y.n;
        dp.box_min[0] = static_cast<float>(grid_.x.xmin);
        dp.box_min[1] = static_cast<float>(grid_.y.xmin);
        dp.box_min[2] = static_cast<float>(grid_.z.xmin);
        dp.cell_h[0] = static_cast<float>(grid_.x.spacing());
        dp.cell_h[1] = static_cast<float>(grid_.y.spacing());
        dp.cell_h[2] = static_cast<float>(grid_.z.spacing());
        using B = QRhiShaderResourceBinding;
        const auto cs = B::ComputeStage;
        QScopedPointer<QRhiShaderResourceBindings> dsrb(rhi_->newShaderResourceBindings());
        dsrb->setBindings({ B::bufferLoad(0, cs, to_buf),
                            B::uniformBuffer(1, cs, dipole_ubo_.data()),
                            B::bufferLoadStore(2, cs, dipole_partials_.data()),
                            B::bufferLoad(3, cs, from_buf) });
        if (!dsrb->create()) { return {}; }
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return {}; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->updateDynamicBuffer(dipole_ubo_.data(), 0, sizeof(DipoleParams), &dp);
        cb->beginComputePass(u);
        cb->setComputePipeline(dipole_pipe_.data());
        cb->setShaderResources(dsrb.data());
        cb->dispatch(kGroups, 1, 1);
        QRhiReadbackResult rb;
        QRhiResourceUpdateBatch* d = rhi_->nextResourceUpdateBatch();
        d->readBackBuffer(dipole_partials_.data(), 0,
                          static_cast<quint32>(6 * kGroups * sizeof(float)), &rb);
        cb->endComputePass(d);
        rhi_->endOffscreenFrame();
        const float* p = reinterpret_cast<const float*>(rb.data.constData());
        double d6[6] = {0, 0, 0, 0, 0, 0};
        for (int gi = 0; gi < kGroups; ++gi) {
            for (int c = 0; c < 6; ++c) { d6[c] += p[6 * gi + c]; }
        }
        return ses::DipoleMatrixElement{
            ses::Complex<double>{ d6[0] * cell_volume_, d6[1] * cell_volume_ },
            ses::Complex<double>{ d6[2] * cell_volume_, d6[3] * cell_volume_ },
            ses::Complex<double>{ d6[4] * cell_volume_, d6[5] * cell_volume_ } };
    }

    // ---- orbital-free angular projection -------------------------------
    // Upload the static counting-sort geometry (ses::build_radial_bin_index) and
    // allocate g_lm[ncomp*nr]. Call once after the radial grid is fixed.
    // sorted_cell = cells grouped by radial bin, bin_off = CSR offsets.
    bool set_projection_index(const std::vector<quint32>& sorted_cell,
                              const std::vector<quint32>& bin_off, int n_radial,
                              double h_radial, int l_max) {
        proj_nr_ = n_radial;
        proj_ncomp_ = (l_max + 1) * (l_max + 1);
        proj_h_radial_ = h_radial;
        const QShader depcs = load_qsb(QStringLiteral(":/shaders/project_deposit.comp.qsb"));
        if (!depcs.isValid()) { return false; }
        proj_sorted_buf_.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer,
                                               static_cast<quint32>(sorted_cell.size() * 4)));
        proj_binoff_buf_.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer,
                                               static_cast<quint32>(bin_off.size() * 4)));
        glm_buf_.reset(rhi_->newBuffer(
            QRhiBuffer::Static, QRhiBuffer::StorageBuffer,
            static_cast<quint32>(2 * proj_ncomp_ * proj_nr_ * sizeof(float))));
        proj_ubo_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                        sizeof(ProjectParams)));
        if (!proj_sorted_buf_->create() || !proj_binoff_buf_->create() || !glm_buf_->create() ||
            !proj_ubo_->create()) {
            return false;
        }
        using B = QRhiShaderResourceBinding;
        const auto cs = B::ComputeStage;
        proj_srb_.reset(rhi_->newShaderResourceBindings());
        proj_srb_->setBindings({ B::bufferLoad(0, cs, psi_.data()),
                                 B::uniformBuffer(1, cs, proj_ubo_.data()),
                                 B::bufferLoad(6, cs, proj_sorted_buf_.data()),
                                 B::bufferLoad(7, cs, proj_binoff_buf_.data()),
                                 B::bufferLoadStore(8, cs, glm_buf_.data()) });
        if (!proj_srb_->create()) { return false; }
        proj_pipe_.reset(rhi_->newComputePipeline());
        proj_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, depcs));
        proj_pipe_->setShaderResourceBindings(proj_srb_.data());
        if (!proj_pipe_->create()) { return false; }

        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return false; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->uploadStaticBuffer(proj_sorted_buf_.data(), sorted_cell.data());
        u->uploadStaticBuffer(proj_binoff_buf_.data(), bin_off.data());
        cb->resourceUpdate(u);
        rhi_->endOffscreenFrame();
        return true;
    }

    // Deposit psi -> g_lm (ONE grid pass, independent of state count), read back
    // to glm_host_ as double. Then call project_amplitude per state. Mirrors
    // Projector::project.
    void project_psi() {
        if (proj_pipe_.isNull()) { return; }  // set_projection_index failed/absent
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

        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->updateDynamicBuffer(proj_ubo_.data(), 0, sizeof(ProjectParams), &pp);
        cb->beginComputePass(u);
        cb->setComputePipeline(proj_pipe_.data());
        cb->setShaderResources(proj_srb_.data());
        cb->dispatch(proj_nr_, 1, 1);
        QRhiReadbackResult rb;
        QRhiResourceUpdateBatch* d = rhi_->nextResourceUpdateBatch();
        d->readBackBuffer(glm_buf_.data(), 0,
                          static_cast<quint32>(2 * proj_ncomp_ * proj_nr_ * sizeof(float)), &rb);
        cb->endComputePass(d);
        rhi_->endOffscreenFrame();

        const float* raw = reinterpret_cast<const float*>(rb.data.constData());
        glm_host_.assign(static_cast<std::size_t>(proj_ncomp_),
                         std::vector<ses::Complex<double>>(static_cast<std::size_t>(proj_nr_)));
        for (int c = 0; c < proj_ncomp_; ++c) {
            for (int j = 0; j < proj_nr_; ++j) {
                const std::size_t o = 2 * (static_cast<std::size_t>(c) * proj_nr_ +
                                           static_cast<std::size_t>(j));
                glm_host_[static_cast<std::size_t>(c)][static_cast<std::size_t>(j)] =
                    ses::Complex<double>{ raw[o], raw[o + 1] };
            }
        }
    }

    // <n|psi> raw amplitude = sum_j u_nl[j] g_lm[lm(l,m)][j] (double CPU finish).
    // Needs a prior project_psi(). Mirrors Projector::amplitude.
    ses::Complex<double> project_amplitude(const std::vector<double>& u, int l, int m) const {
        const std::size_t comp = static_cast<std::size_t>(l * l + (l + m));
        if (comp >= glm_host_.size()) { return {}; }  // no deposit yet / setup failed
        const std::vector<ses::Complex<double>>& gc = glm_host_[comp];
        ses::Complex<double> raw{};
        const int n = std::min(static_cast<int>(u.size()), proj_nr_);
        for (int j = 0; j < n; ++j) {
            raw += u[static_cast<std::size_t>(j)] * gc[static_cast<std::size_t>(j)];
        }
        return raw;
    }

    // ---- magnetic minimal coupling (real-time B field) -----------------
    // Replace the resident half-potential table (half_) with a new one -- the
    // caller swaps in the diamagnetic-augmented V + (B^2/8) rho_perp^2 before a
    // magnetic run (GpuEngine::set_half_potential). Static-buffer re-upload.
    void set_half_potential(const std::vector<ses::Complex<double>>& half_v) {
        const std::vector<float> hf = to_rg32f(half_v);
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->uploadStaticBuffer(half_.data(), hf.data());
        cb->resourceUpdate(u);
        rhi_->endOffscreenFrame();
    }

    // Exact three-shear rotation of psi about coordinate `axis` by theta -- the
    // GPU transcription of core/rotation.hpp rotate_axis. In place, one frame.
    void rotate_axis_shear(int axis, double theta) {
        if (!ensure_magnetic()) { return; }
        const int mul_groups = static_cast<int>((cells_ + 255) / 256);
        const int b = (axis + 1) % 3;  // in-plane axes (b x c = axis)
        const int c = (axis + 2) % 3;
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        stage_rotation_ubos(u, b, c, theta);
        cb->beginComputePass(u);
        record_rotation(cb, mul_groups, b, c);
        cb->endComputePass();
        rhi_->endOffscreenFrame();
    }
    void rotate_z_shear(double theta) { rotate_axis_shear(2, theta); }

    // Magnetic Strang step: R(a) . real-step . R(a), a = (B/2)(dt/2), rotating
    // about the field `axis`. half_ must hold the diamagnetic-augmented table
    // (set_half_potential). The half-angle is the SAME for every rotation in
    // the batch, so the two shear parameter sets are staged once and the WHOLE
    // batch records as a single offscreen frame (per-step frames serialize on
    // GPU completion and gut the magnetic-path sim rate).
    void magnetic_step(int axis, double half_angle, int nsteps) {
        if (!ensure_magnetic()) { return; }
        const int mul_groups = static_cast<int>((cells_ + 255) / 256);
        const int b = (axis + 1) % 3;
        const int c = (axis + 2) % 3;
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        update_step_ubos(u);
        stage_rotation_ubos(u, b, c, half_angle);
        cb->beginComputePass(u, QRhiCommandBuffer::ExternalContent);
        for (int s = 0; s < nsteps; ++s) {
            record_rotation(cb, mul_groups, b, c);
            run_step_body(cb, mul_groups, mul_half_srb_.data(), mul_kin_srb_.data());
            record_rotation(cb, mul_groups, b, c);
        }
        cb->endComputePass();
        rhi_->endOffscreenFrame();
    }

    // ---- SSBO -> 3D volume texture bridge (renderer seed for M3) --------
    // Copy psi into the owned RGBA32F volume texture (imageStore, .xy = psi),
    // the GPU-to-GPU feed the volume renderer samples. write_psi_to_volume does
    // the store; bridge_roundtrip additionally reads the texture back through an
    // SSBO (imageLoad) so the bridge is verifiable without a per-slice 3D
    // texture readback. Returns false if a resource fails to build.

    // The RGBA32F volume the renderer samples (psi in .xy). Lazily created;
    // nullptr only if creation failed.
    QRhiTexture* volume_texture() {
        if (!ensure_volume()) { return nullptr; }
        return volume_tex_.data();
    }

    bool write_psi_to_volume() {
        if (!ensure_volume()) { return false; }
        const int groups = static_cast<int>((cells_ + 255) / 256);
        const BridgeParams bp{ grid_.x.n, grid_.y.n, grid_.z.n, 0 };
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return false; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->updateDynamicBuffer(bridge_ubo_.data(), 0, sizeof(BridgeParams), &bp);
        cb->beginComputePass(u);
        cb->setComputePipeline(store_pipe_.data());
        cb->setShaderResources(store_srb_.data());
        cb->dispatch(groups, 1, 1);
        cb->endComputePass();
        rhi_->endOffscreenFrame();
        return true;
    }

    // psi -> volume texture -> scratch SSBO -> host. out matches psi bit-for-bit
    // iff imageStore/imageLoad round-trip the RGBA32F texel .xy losslessly.
    bool bridge_roundtrip(std::vector<float>& out) {
        if (!write_psi_to_volume()) { return false; }
        if (!ensure_bridge_check()) { return false; }
        const int groups = static_cast<int>((cells_ + 255) / 256);
        const BridgeParams bp{ grid_.x.n, grid_.y.n, grid_.z.n, 0 };
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return false; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->updateDynamicBuffer(bridge_ubo_.data(), 0, sizeof(BridgeParams), &bp);
        cb->beginComputePass(u);
        cb->setComputePipeline(load_pipe_.data());
        cb->setShaderResources(load_srb_.data());
        cb->dispatch(groups, 1, 1);
        QRhiReadbackResult rb;
        QRhiResourceUpdateBatch* d = rhi_->nextResourceUpdateBatch();
        d->readBackBuffer(scratch_bridge_buf_.data(), 0,
                          static_cast<quint32>(2 * cells_ * sizeof(float)), &rb);
        cb->endComputePass(d);
        rhi_->endOffscreenFrame();
        out.assign(reinterpret_cast<const float*>(rb.data.constData()),
                   reinterpret_cast<const float*>(rb.data.constData()) + 2 * cells_);
        return true;
    }

private:
    static constexpr int kGroups = 256;  // norm/peak workgroups (ses_gpu::kNormPeakGroups)

    struct alignas(16) MulParams { quint32 n, p0, p1, p2; };
    struct alignas(16) ConjParams { quint32 n; float scale; float p0, p1; };
    struct alignas(16) FftParams { qint32 mod_a, mul_b, mul_c, stride, n_lines, p0, p1, p2; };
    struct alignas(16) KickParams {
        quint32 n;
        qint32 nx, ny;
        float theta;
        float box_min[4];
        float cell_h[4];
        float axis[4];
    };
    // std140 {uint n; vec2 c}: c aligns to offset 8 (uint + 4 bytes pad).
    struct alignas(16) AxpyParams { quint32 n, pad; float cx, cy; };
    // std140 all-scalar block (tight 4-byte packing), matches shear.comp order.
    struct alignas(16) ShearParams {
        quint32 n;
        qint32 nx, ny, nz;
        qint32 freq_axis, coord_axis, nf;
        float kscale, cmin, ch, coeff;
    };
    // std140: vec4-padded box_min/cell_h at 16/32, matches synth.comp order.
    struct alignas(16) SynthParams {
        quint32 n;
        qint32 nx, ny, l;
        float box_min[4];
        float cell_h[4];
        qint32 m, n_radial;
        float h_radial, rmax;
    };
    struct alignas(16) BridgeParams { qint32 nx, ny, nz, pad; };
    // std140: n@0, nx@4, ny@8, pad@12, box_min@16, cell_h@32, matches dipole.comp.
    struct alignas(16) DipoleParams {
        quint32 n;
        qint32 nx, ny, pad0;
        float box_min[4];
        float cell_h[4];
    };

    // A resident state: fp32 (2*cells floats) or fp16-packed (cells uints, half
    // the footprint). The tested fp32 consumers read fp16 states by unpacking to
    // a scratch buffer on demand (decode-on-use). Heap-owned (unique_ptr) so the
    // QScopedPointer members never move; the per-state SRBs (this buffer as
    // phi/src/table) build lazily on first use, fp32 states only.
    struct AtlasState {
        bool is_half = false;
        QScopedPointer<QRhiBuffer> buf;
        QScopedPointer<QRhiShaderResourceBindings> inner_srb, axpy_srb, copy_srb, mul_srb;
    };
    // std140: nx@0, ny@4, nr@8, h_radial@12, box_min@16, cell_h@32, dv@48.
    struct alignas(16) ProjectParams {
        qint32 nx, ny, nr;
        float h_radial;
        float box_min[4];
        float cell_h[4];
        float dv, pad0, pad1, pad2;
    };

    void multiply(QRhiCommandBuffer* cb, QRhiShaderResourceBindings* srb, int groups) {
        cb->setComputePipeline(mul_pipe_.data());
        cb->setShaderResources(srb);
        cb->dispatch(groups, 1, 1);
    }
    void conjscale(QRhiCommandBuffer* cb, QRhiShaderResourceBindings* srb, int groups) {
        cb->setComputePipeline(conj_pipe_.data());
        cb->setShaderResources(srb);
        cb->dispatch(groups, 1, 1);
    }
    void fft3(QRhiCommandBuffer* cb) {
        cb->setComputePipeline(fft_pipe_.data());
        for (int a = 0; a < 3; ++a) {
            cb->setShaderResources(fft_srb_[a].data());
            cb->dispatch(n_ * n_, 1, 1);
        }
    }
    // Forward 1D FFT along a single axis (n^2 lines), the shear's freq transform.
    void fft_axis(QRhiCommandBuffer* cb, int a) {
        cb->setComputePipeline(fft_pipe_.data());
        cb->setShaderResources(fft_srb_[a].data());
        cb->dispatch(n_ * n_, 1, 1);
    }

    void update_step_ubos(QRhiResourceUpdateBatch* u) {
        u->updateDynamicBuffer(muln_ubo_.data(), 0, sizeof(MulParams), &muln_);
        u->updateDynamicBuffer(conj1_ubo_.data(), 0, sizeof(ConjParams), &conj1_);
        u->updateDynamicBuffer(conjN_ubo_.data(), 0, sizeof(ConjParams), &conjN_);
        for (int a = 0; a < 3; ++a) {
            u->updateDynamicBuffer(fft_ubo_[a].data(), 0, sizeof(FftParams), &fftp_[a]);
        }
    }

    // One split-operator step in the current pass: halfV . IFFT . kin . FFT .
    // halfV, using the given half/kin multiply bindings (real vs relax tables).
    // With VkFFT active, the two 3D transforms are recorded as external blocks
    // (coalesced transposes; the inverse carries the 1/N normalize), replacing
    // six line-FFT dispatches and both conj-scale dispatches; the enclosing
    // pass must have been begun with QRhiCommandBuffer::ExternalContent.
    void run_step_body(QRhiCommandBuffer* cb, int mul_groups,
                       QRhiShaderResourceBindings* half_srb,
                       QRhiShaderResourceBindings* kin_srb) {
        multiply(cb, half_srb, mul_groups);
#ifdef SES_HAVE_VKFFT
        if (vkfft_active()) {
            record_vkfft(cb, -1);  // forward
            multiply(cb, kin_srb, mul_groups);
            record_vkfft(cb, 1);   // inverse, normalized 1/N by the plan
            multiply(cb, half_srb, mul_groups);
            return;
        }
#endif
        fft3(cb);
        multiply(cb, kin_srb, mul_groups);
        conjscale(cb, conj1_srb_.data(), mul_groups);
        fft3(cb);
        conjscale(cb, conjN_srb_.data(), mul_groups);
        multiply(cb, half_srb, mul_groups);
    }

    // The dipole half-kick parameter block for psi <- exp(-i theta axis.r) psi.
    KickParams make_kick_params(const ses::Vec3d& axis, double theta) const {
        KickParams kp{};
        kp.n = static_cast<quint32>(cells_);
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

    // Record one half-kick dispatch in the current pass, sourcing its params
    // from the dynamic-offset slot at `offset` in kick_ubo_.
    void record_kick(QRhiCommandBuffer* cb, int groups, quint32 offset) {
        cb->setComputePipeline(kick_pipe_.data());
        const QRhiCommandBuffer::DynamicOffset off(1, offset);
        cb->setShaderResources(kick_srb_.data(), 1, &off);
        cb->dispatch(groups, 1, 1);
    }

    // Grow the kick parameter UBO to hold `kicks` slots (kick_stride_ apart, the
    // backend's dynamic-offset alignment). The SRB re-points at the new buffer;
    // the pipeline keeps its (unchanged) layout.
    bool ensure_kick_capacity(int kicks) {
        const quint32 need = kick_stride_ * static_cast<quint32>(kicks);
        if (!kick_ubo_.isNull() && kick_ubo_->size() >= need) { return true; }
        kick_ubo_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, need));
        if (!kick_ubo_->create()) { return false; }
        using B = QRhiShaderResourceBinding;
        const auto cs = B::ComputeStage;
        kick_srb_.reset(rhi_->newShaderResourceBindings());
        kick_srb_->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                 B::uniformBufferWithDynamicOffset(1, cs, kick_ubo_.data(),
                                                                   sizeof(KickParams)) });
        return kick_srb_->create();
    }

    // A live-handle lookup that tolerates released slots (nullptr buf).
    AtlasState* state_at(int handle) {
        if (handle < 0 || handle >= static_cast<int>(atlas_.size())) { return nullptr; }
        AtlasState* a = atlas_[static_cast<std::size_t>(handle)].get();
        return a->buf.isNull() ? nullptr : a;
    }

    // Norm/peak reduction over psi + partials readback (one frame). Raw: the
    // caller applies dV to sum; peak is the max cell density as-is.
    NormPeak reduce_norm_peak() {
        QRhiReadbackResult rb;
        run_reduce(norm_pipe_.data(), norm_srb_.data(), partials_.data(),
                   static_cast<quint32>(2 * kGroups * sizeof(float)), &rb);
        const float* p = reinterpret_cast<const float*>(rb.data.constData());
        NormPeak out;
        for (int gi = 0; gi < kGroups; ++gi) {
            out.sum += p[2 * gi];
            out.peak = std::max(out.peak, static_cast<double>(p[2 * gi + 1]));
        }
        return out;
    }

    // Dispatch inner_pipe_ with `srb` + read the complex partials back (x dV).
    ses::Complex<double> finish_inner(QRhiShaderResourceBindings* srb) {
        QRhiReadbackResult rb;
        run_reduce(inner_pipe_.data(), srb, partials_.data(),
                   static_cast<quint32>(2 * kGroups * sizeof(float)), &rb);
        const float* p = reinterpret_cast<const float*>(rb.data.constData());
        double re = 0.0;
        double im = 0.0;
        for (int gi = 0; gi < kGroups; ++gi) { re += p[2 * gi]; im += p[2 * gi + 1]; }
        return ses::Complex<double>{ re * cell_volume_, im * cell_volume_ };
    }

    // Frame N: norm reduction (sum |psi|^2 dV) + partials readback -> host norm,
    // then Frame S: psi <- psi / sqrt(norm). Returns the free-energy estimate
    // from the pre-renorm norm (e^{-2 E dtau}) plus the normalized peak density.
    // Shared by relax / relax_deflated.
    RelaxStats renormalize_and_estimate(int mul_groups) {
        RelaxStats stats;
        const NormPeak np = reduce_norm_peak();
        const double norm_sq = np.sum * cell_volume_;
        const double inv = (norm_sq > 0.0) ? 1.0 / std::sqrt(norm_sq) : 0.0;
        stats.energy = (norm_sq > 0.0 && dtau_ > 0.0) ? -std::log(norm_sq) / (2.0 * dtau_) : 0.0;
        stats.peak = (norm_sq > 0.0) ? np.peak / norm_sq : 0.0;

        const ConjParams sp{ static_cast<quint32>(cells_), static_cast<float>(inv), 0.0f, 0.0f };
        QRhiCommandBuffer* cb2 = nullptr;
        if (rhi_->beginOffscreenFrame(&cb2) != QRhi::FrameOpSuccess) { return stats; }
        QRhiResourceUpdateBatch* u2 = rhi_->nextResourceUpdateBatch();
        u2->updateDynamicBuffer(scale_ubo_.data(), 0, sizeof(ConjParams), &sp);
        cb2->beginComputePass(u2);
        cb2->setComputePipeline(scale_pipe_.data());
        cb2->setShaderResources(scale_srb_.data());
        cb2->dispatch(mul_groups, 1, 1);
        cb2->endComputePass();
        rhi_->endOffscreenFrame();
        return stats;
    }

    // psi <- psi - c*phi (state `handle`), the Gram-Schmidt projection (own frame).
    void subtract_projection(int handle, double cre, double cim, int groups) {
        AtlasState* a = state_at(handle);
        if (a == nullptr || a->is_half || !ensure_state_ops(a)) { return; }
        const AxpyParams ap{ static_cast<quint32>(cells_), 0, static_cast<float>(cre),
                             static_cast<float>(cim) };
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->updateDynamicBuffer(axpy_ubo_.data(), 0, sizeof(AxpyParams), &ap);
        cb->beginComputePass(u);
        cb->setComputePipeline(axpy_pipe_.data());
        cb->setShaderResources(a->axpy_srb.data());
        cb->dispatch(groups, 1, 1);
        cb->endComputePass();
        rhi_->endOffscreenFrame();
    }

    // Lazily build a state's per-op SRBs (inner/axpy/copy/mul) and, on first
    // need, the shared axpy UBO and the inner/axpy/copy pipelines (the state's
    // SRBs are the layout template; later states share the layout). fp32 only.
    bool ensure_state_ops(AtlasState* a) {
        if (!a->inner_srb.isNull()) { return true; }
        // Shared projection-coefficient UBO BEFORE any SRB names it (a null
        // binding would crash the SRB create).
        if (axpy_ubo_.isNull()) {
            axpy_ubo_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                            sizeof(AxpyParams)));
            if (!axpy_ubo_->create()) { return false; }
        }
        using B = QRhiShaderResourceBinding;
        const auto cs = B::ComputeStage;
        a->inner_srb.reset(rhi_->newShaderResourceBindings());
        a->inner_srb->setBindings({ B::bufferLoad(0, cs, psi_.data()),
                                    B::uniformBuffer(1, cs, muln_ubo_.data()),
                                    B::bufferLoadStore(2, cs, partials_.data()),
                                    B::bufferLoad(3, cs, a->buf.data()) });
        a->axpy_srb.reset(rhi_->newShaderResourceBindings());
        a->axpy_srb->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                   B::uniformBuffer(1, cs, axpy_ubo_.data()),
                                   B::bufferLoad(3, cs, a->buf.data()) });
        a->copy_srb.reset(rhi_->newShaderResourceBindings());
        a->copy_srb->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                   B::uniformBuffer(1, cs, muln_ubo_.data()),
                                   B::bufferLoad(3, cs, a->buf.data()) });
        // Elementwise multiply (absorber mask): phase_multiply layout, so it
        // dispatches against the shared mul_pipe_.
        a->mul_srb.reset(rhi_->newShaderResourceBindings());
        a->mul_srb->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                  B::bufferLoad(1, cs, a->buf.data()),
                                  B::uniformBuffer(2, cs, muln_ubo_.data()) });
        // Success is keyed off inner_srb being non-null, so every failure
        // path must roll the partially built SRBs back for a clean retry.
        const auto fail = [a]() {
            a->inner_srb.reset();
            a->axpy_srb.reset();
            a->copy_srb.reset();
            a->mul_srb.reset();
            return false;
        };
        if (!a->inner_srb->create() || !a->axpy_srb->create() || !a->copy_srb->create() ||
            !a->mul_srb->create()) {
            return fail();
        }
        if (!ensure_inner() || !ensure_axpy_copy(a)) {
            return fail();
        }
        return true;
    }

    // Build the axpy/copy pipelines once from `tpl`'s SRBs as layout template.
    bool ensure_axpy_copy(AtlasState* tpl) {
        if (!axpy_pipe_.isNull()) { return true; }
        const QShader axpycs = load_qsb(QStringLiteral(":/shaders/axpy.comp.qsb"));
        const QShader copycs = load_qsb(QStringLiteral(":/shaders/copy_state.comp.qsb"));
        if (!axpycs.isValid() || !copycs.isValid()) { return false; }
        axpy_pipe_.reset(rhi_->newComputePipeline());
        axpy_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, axpycs));
        axpy_pipe_->setShaderResourceBindings(tpl->axpy_srb.data());
        copy_pipe_.reset(rhi_->newComputePipeline());
        copy_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, copycs));
        copy_pipe_->setShaderResourceBindings(tpl->copy_srb.data());
        return axpy_pipe_->create() && copy_pipe_->create();
    }

    // Lazily build the mean-force reduction (grad V buffer + vec4 partials +
    // SRB + pipeline). Called by set_potential_gradient.
    bool ensure_force() {
        if (!force_pipe_.isNull()) { return true; }
        const auto fail = [this]() {
            force_pipe_.reset();
            force_srb_.reset();
            force_partials_.reset();
            grad_v_buf_.reset();
            return false;
        };
        const QShader forcecs = load_qsb(QStringLiteral(":/shaders/mean_force.comp.qsb"));
        if (!forcecs.isValid()) { return fail(); }
        grad_v_buf_.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer,
                                          static_cast<quint32>(4 * cells_ * sizeof(float))));
        force_partials_.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer,
                                              static_cast<quint32>(4 * kGroups * sizeof(float))));
        if (!grad_v_buf_->create() || !force_partials_->create()) { return fail(); }
        using B = QRhiShaderResourceBinding;
        const auto cs = B::ComputeStage;
        force_srb_.reset(rhi_->newShaderResourceBindings());
        force_srb_->setBindings({ B::bufferLoad(0, cs, psi_.data()),
                                  B::uniformBuffer(1, cs, muln_ubo_.data()),
                                  B::bufferLoadStore(2, cs, force_partials_.data()),
                                  B::bufferLoad(4, cs, grad_v_buf_.data()) });
        if (!force_srb_->create()) { return fail(); }
        force_pipe_.reset(rhi_->newComputePipeline());
        force_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, forcecs));
        force_pipe_->setShaderResourceBindings(force_srb_.data());
        if (!force_pipe_->create()) { return fail(); }
        return true;
    }

    // Lazily build the synth UBO/SRB/pipeline. rebuild_srb re-points the SRB at a
    // freshly (re)sized radial buffer; the pipeline layout is unchanged so it is
    // kept. Call after radial_buf_ exists.
    bool ensure_synth(bool rebuild_srb) {
        using B = QRhiShaderResourceBinding;
        const auto cs = B::ComputeStage;
        if (synth_ubo_.isNull()) {
            synth_ubo_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                             sizeof(SynthParams)));
            if (!synth_ubo_->create()) { return false; }
        }
        if (rebuild_srb || synth_srb_.isNull()) {
            synth_srb_.reset(rhi_->newShaderResourceBindings());
            synth_srb_->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                      B::uniformBuffer(1, cs, synth_ubo_.data()),
                                      B::bufferLoad(5, cs, radial_buf_.data()) });
            if (!synth_srb_->create()) { return false; }
        }
        if (synth_pipe_.isNull()) {
            const QShader synthcs = load_qsb(QStringLiteral(":/shaders/synth.comp.qsb"));
            if (!synthcs.isValid()) { return false; }
            synth_pipe_.reset(rhi_->newComputePipeline());
            synth_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, synthcs));
            synth_pipe_->setShaderResourceBindings(synth_srb_.data());
            if (!synth_pipe_->create()) { return false; }
        }
        return true;
    }

    // Lazily build the shear pipeline + TWO shear param UBOs/SRBs (a rotation
    // is shear0 . shear1 . shear0, so two distinct parameter sets cover it and
    // a whole batch can record in one pass) and the per-axis inverse-FFT
    // conj-scale (scale 1/n, not 1/cells) on the first magnetic call.
    bool ensure_magnetic() {
        if (!shear_pipe_.isNull()) { return true; }
        const QShader shearcs = load_qsb(QStringLiteral(":/shaders/shear.comp.qsb"));
        if (!shearcs.isValid()) { return false; }
        using B = QRhiShaderResourceBinding;
        const auto cs = B::ComputeStage;
        conjA_ubo_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                         sizeof(ConjParams)));
        if (!conjA_ubo_->create()) { return false; }
        for (int i = 0; i < 2; ++i) {
            shear_ubo_[i].reset(rhi_->newBuffer(QRhiBuffer::Dynamic,
                                                QRhiBuffer::UniformBuffer,
                                                sizeof(ShearParams)));
            if (!shear_ubo_[i]->create()) { return false; }
            shear_srb_[i].reset(rhi_->newShaderResourceBindings());
            shear_srb_[i]->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                         B::uniformBuffer(1, cs, shear_ubo_[i].data()) });
            if (!shear_srb_[i]->create()) { return false; }
        }
        conjA_srb_.reset(rhi_->newShaderResourceBindings());
        conjA_srb_->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                  B::uniformBuffer(1, cs, conjA_ubo_.data()) });
        if (!conjA_srb_->create()) { return false; }
        shear_pipe_.reset(rhi_->newComputePipeline());
        shear_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, shearcs));
        shear_pipe_->setShaderResourceBindings(shear_srb_[0].data());
        // Per-axis inverse FFT scales by 1/n (the single transformed axis).
        conjA_ = ConjParams{ static_cast<quint32>(cells_), 1.0f / static_cast<float>(n_),
                             0.0f, 0.0f };
        return shear_pipe_->create();
    }

    // Lazily build the RGBA32F volume texture + the store side of the bridge
    // (the production renderer feed). The load/scratch side is check-only and
    // lives in ensure_bridge_check so the app never pays its 2*cells*4B scratch.
    bool ensure_volume() {
        if (!store_pipe_.isNull()) { return true; }
        const QShader storecs = load_qsb(QStringLiteral(":/shaders/bridge_store.comp.qsb"));
        if (!storecs.isValid()) { return false; }
        volume_tex_.reset(rhi_->newTexture(
            QRhiTexture::RGBA32F, n_, n_, n_, 1,
            QRhiTexture::ThreeDimensional | QRhiTexture::UsedWithLoadStore));
        if (!volume_tex_->create()) { return false; }
        bridge_ubo_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                          sizeof(BridgeParams)));
        if (!bridge_ubo_->create()) { return false; }
        using B = QRhiShaderResourceBinding;
        const auto cs = B::ComputeStage;
        store_srb_.reset(rhi_->newShaderResourceBindings());
        store_srb_->setBindings({ B::bufferLoad(0, cs, psi_.data()),
                                  B::imageStore(1, cs, volume_tex_.data(), 0),
                                  B::uniformBuffer(2, cs, bridge_ubo_.data()) });
        if (!store_srb_->create()) { return false; }
        store_pipe_.reset(rhi_->newComputePipeline());
        store_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, storecs));
        store_pipe_->setShaderResourceBindings(store_srb_.data());
        return store_pipe_->create();
    }

#ifdef SES_HAVE_VKFFT
    // ---- VkFFT (M4): the 3D transforms of the step body ------------------
    // Plan a 3D C2C fp32 transform IN PLACE on psi's native VkBuffer, on the
    // QRhi device/queue. Called once from initialize() (outside any frame:
    // plan creation compiles shaders via glslang). Failure leaves the engine
    // on the hand-rolled path.
    bool ensure_vkfft() {
        if (vkfft_ready_ || vkfft_failed_) { return vkfft_ready_; }
        vkfft_failed_ = true;
        const auto* h =
            static_cast<const QRhiVulkanNativeHandles*>(rhi_->nativeHandles());
        if (h == nullptr || h->inst == nullptr) { return false; }
        vkfft_df_ = h->inst->deviceFunctions(h->dev);
        vkfft_phys_dev_ = h->physDev;
        vkfft_dev_ = h->dev;
        vkfft_queue_ = h->gfxQueue;
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = h->gfxQueueFamilyIdx;
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkfft_df_->vkCreateCommandPool(vkfft_dev_, &pool_info, nullptr,
                                           &vkfft_pool_) != VK_SUCCESS ||
            vkfft_df_->vkCreateFence(vkfft_dev_, &fence_info, nullptr,
                                     &vkfft_fence_) != VK_SUCCESS) {
            release_vkfft();
            return false;
        }
        vkfft_psi_buf_ =
            *reinterpret_cast<const VkBuffer*>(psi_->nativeBuffer().objects[0]);
        vkfft_buf_size_ = static_cast<quint64>(2 * cells_ * sizeof(float));
        // The configuration stores POINTERS; the pointees are members so they
        // outlive the plan (VkFFTAppend dereferences them per call).
        VkFFTConfiguration conf{};
        conf.FFTdim = 3;
        conf.size[0] = static_cast<quint64>(grid_.x.n);
        conf.size[1] = static_cast<quint64>(grid_.y.n);
        conf.size[2] = static_cast<quint64>(grid_.z.n);
        conf.physicalDevice = &vkfft_phys_dev_;
        conf.device = &vkfft_dev_;
        conf.queue = &vkfft_queue_;
        conf.commandPool = &vkfft_pool_;
        conf.fence = &vkfft_fence_;
        conf.buffer = &vkfft_psi_buf_;
        conf.bufferSize = &vkfft_buf_size_;
        conf.normalize = 1;  // inverse divides by N (replaces the conj/N pass)
        const VkFFTResult res = initializeVkFFT(&vkfft_app_, conf);
        if (res != VKFFT_SUCCESS) {
            std::fprintf(stderr, "QrhiEngine: initializeVkFFT = %d -- staying on "
                                 "the hand-rolled FFT\n",
                         static_cast<int>(res));
            release_vkfft();
            return false;
        }
        std::fprintf(stderr, "QrhiEngine: VkFFT 3D plan active (%dx%dx%d)\n",
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
        if (vkfft_df_ != nullptr) {
            if (vkfft_fence_ != VK_NULL_HANDLE) {
                vkfft_df_->vkDestroyFence(vkfft_dev_, vkfft_fence_, nullptr);
                vkfft_fence_ = VK_NULL_HANDLE;
            }
            if (vkfft_pool_ != VK_NULL_HANDLE) {
                vkfft_df_->vkDestroyCommandPool(vkfft_dev_, vkfft_pool_, nullptr);
                vkfft_pool_ = VK_NULL_HANDLE;
            }
        }
    }

    // Record one whole-3D transform as an external block in the current
    // ExternalContent compute pass. direction: -1 forward, 1 inverse (1/N).
    // QRhi's resource tracking is suspended inside the block, so explicit
    // compute-to-compute memory barriers order psi against the surrounding
    // QRhi dispatches on both sides.
    void record_vkfft(QRhiCommandBuffer* cb, int direction) {
        cb->beginExternal();
        // Re-query AFTER beginExternal: on Vulkan this is the pass's live
        // secondary command buffer, not the primary.
        const auto* nh = static_cast<const QRhiVulkanCommandBufferNativeHandles*>(
            cb->nativeHandles());
        VkCommandBuffer vkcb = nh->commandBuffer;
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkfft_df_->vkCmdPipelineBarrier(vkcb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                        &mb, 0, nullptr, 0, nullptr);
        VkFFTLaunchParams lp{};
        lp.buffer = &vkfft_psi_buf_;
        lp.commandBuffer = &vkcb;
        const VkFFTResult res = VkFFTAppend(&vkfft_app_, direction, &lp);
        if (res != VKFFT_SUCCESS) {
            // Should be impossible after a successful plan; surface loudly.
            std::fprintf(stderr, "QrhiEngine: VkFFTAppend = %d\n",
                         static_cast<int>(res));
        }
        vkfft_df_->vkCmdPipelineBarrier(vkcb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                        &mb, 0, nullptr, 0, nullptr);
        cb->endExternal();
    }
#else
    void release_vkfft() {}
    bool ensure_vkfft() { return false; }
#endif  // SES_HAVE_VKFFT

    // Check-only extras: the imageLoad readback path of bridge_roundtrip.
    bool ensure_bridge_check() {
        if (!load_pipe_.isNull()) { return true; }
        const QShader loadcs = load_qsb(QStringLiteral(":/shaders/bridge_load.comp.qsb"));
        if (!loadcs.isValid()) { return false; }
        scratch_bridge_buf_.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer,
                                                  static_cast<quint32>(2 * cells_ * sizeof(float))));
        if (!scratch_bridge_buf_->create()) { return false; }
        using B = QRhiShaderResourceBinding;
        const auto cs = B::ComputeStage;
        load_srb_.reset(rhi_->newShaderResourceBindings());
        load_srb_->setBindings({ B::bufferLoadStore(0, cs, scratch_bridge_buf_.data()),
                                 B::imageLoad(1, cs, volume_tex_.data(), 0),
                                 B::uniformBuffer(2, cs, bridge_ubo_.data()) });
        if (!load_srb_->create()) { return false; }
        load_pipe_.reset(rhi_->newComputePipeline());
        load_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, loadcs));
        load_pipe_->setShaderResourceBindings(load_srb_.data());
        return load_pipe_->create();
    }

    // ---- atlas helpers (synthesize_state / fp16 / inner / dipole) ------
    // Synthesize a normalized fp32 eigenstate into `out` (binding-0 storage
    // buffer), dispatching synth_pipe_ with a per-buffer SRB then normalizing.
    // *out_peak = normalized peak |psi|^2; *out_norm2 = pre-normalization grid
    // norm (the orbital-free projection's population normalizer).
    bool synthesize_into_buffer(QRhiBuffer* out, const std::vector<double>& u, int l, int m,
                                double h_radial, double rmax, int n_radial,
                                double* out_peak = nullptr, double* out_norm2 = nullptr) {
        std::vector<float> uf(u.size());
        for (std::size_t i = 0; i < u.size(); ++i) { uf[i] = static_cast<float>(u[i]); }
        const quint32 rbytes = static_cast<quint32>(uf.size() * sizeof(float));
        bool rebuilt = false;
        if (radial_buf_.isNull() || radial_bytes_ != rbytes) {
            radial_buf_.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer,
                                              rbytes));
            if (!radial_buf_->create()) { return false; }
            radial_bytes_ = rbytes;
            rebuilt = true;
        }
        if (!ensure_synth(rebuilt)) { return false; }

        SynthParams sp{};
        sp.n = static_cast<quint32>(cells_);
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

        using B = QRhiShaderResourceBinding;
        const auto cs = B::ComputeStage;
        QScopedPointer<QRhiShaderResourceBindings> osrb(rhi_->newShaderResourceBindings());
        osrb->setBindings({ B::bufferLoadStore(0, cs, out),
                            B::uniformBuffer(1, cs, synth_ubo_.data()),
                            B::bufferLoad(5, cs, radial_buf_.data()) });
        if (!osrb->create()) { return false; }

        const int groups = static_cast<int>((cells_ + 255) / 256);
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return false; }
        QRhiResourceUpdateBatch* uu = rhi_->nextResourceUpdateBatch();
        uu->uploadStaticBuffer(radial_buf_.data(), uf.data());
        uu->updateDynamicBuffer(synth_ubo_.data(), 0, sizeof(SynthParams), &sp);
        cb->beginComputePass(uu);
        cb->setComputePipeline(synth_pipe_.data());
        cb->setShaderResources(osrb.data());
        cb->dispatch(groups, 1, 1);
        cb->endComputePass();
        rhi_->endOffscreenFrame();

        const NormPeak np = normalize_buffer(out);
        if (out_norm2 != nullptr) { *out_norm2 = np.sum; }
        if (out_peak != nullptr) { *out_peak = (np.sum > 0.0) ? np.peak / np.sum : 0.0; }
        return true;
    }

    // psi'-agnostic grid normalization of `buf` (norm reduction * dV -> scale by
    // 1/sqrt(norm)), via per-buffer SRBs layout-compatible with norm/scale_pipe_.
    // Returns {pre-normalization grid norm (x dV), raw peak density}.
    NormPeak normalize_buffer(QRhiBuffer* buf) {
        const int groups = static_cast<int>((cells_ + 255) / 256);
        using B = QRhiShaderResourceBinding;
        const auto cs = B::ComputeStage;
        QScopedPointer<QRhiShaderResourceBindings> nsrb(rhi_->newShaderResourceBindings());
        nsrb->setBindings({ B::bufferLoad(0, cs, buf),
                            B::uniformBuffer(1, cs, muln_ubo_.data()),
                            B::bufferLoadStore(2, cs, partials_.data()) });
        if (!nsrb->create()) { return {}; }
        QRhiReadbackResult rb;
        run_reduce(norm_pipe_.data(), nsrb.data(), partials_.data(),
                   static_cast<quint32>(2 * kGroups * sizeof(float)), &rb);
        const float* p = reinterpret_cast<const float*>(rb.data.constData());
        NormPeak np;
        for (int gi = 0; gi < kGroups; ++gi) {
            np.sum += p[2 * gi];
            np.peak = std::max(np.peak, static_cast<double>(p[2 * gi + 1]));
        }
        np.sum *= cell_volume_;
        const double inv = (np.sum > 0.0) ? 1.0 / std::sqrt(np.sum) : 0.0;
        const ConjParams sp{ static_cast<quint32>(cells_), static_cast<float>(inv), 0.0f, 0.0f };
        QScopedPointer<QRhiShaderResourceBindings> ssrb(rhi_->newShaderResourceBindings());
        ssrb->setBindings({ B::bufferLoadStore(0, cs, buf),
                            B::uniformBuffer(1, cs, scale_ubo_.data()) });
        if (!ssrb->create()) { return np; }
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return np; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->updateDynamicBuffer(scale_ubo_.data(), 0, sizeof(ConjParams), &sp);
        cb->beginComputePass(u);
        cb->setComputePipeline(scale_pipe_.data());
        cb->setShaderResources(ssrb.data());
        cb->dispatch(groups, 1, 1);
        cb->endComputePass();
        rhi_->endOffscreenFrame();
        return np;
    }

    // Return a readable fp32 buffer for atlas state `handle`: the buffer itself if
    // fp32, else the fp16 state unpacked into decode_scratch_[slot] (0 or 1).
    QRhiBuffer* decode(int handle, int slot) {
        AtlasState* a = state_at(handle);
        if (a == nullptr) { return nullptr; }
        if (!a->is_half) { return a->buf.data(); }
        if (!ensure_fp16()) { return nullptr; }
        using B = QRhiShaderResourceBinding;
        const auto cs = B::ComputeStage;
        QScopedPointer<QRhiShaderResourceBindings> usrb(rhi_->newShaderResourceBindings());
        usrb->setBindings({ B::bufferLoadStore(0, cs, decode_scratch_[slot].data()),
                            B::uniformBuffer(1, cs, muln_ubo_.data()),
                            B::bufferLoad(6, cs, a->buf.data()) });
        if (!usrb->create()) { return nullptr; }
        run_single(unpack_pipe_.data(), usrb.data());
        return decode_scratch_[slot].data();
    }

    // One dispatch (cells/256 groups) in its own frame; updates muln_ubo_ (n).
    void run_single(QRhiComputePipeline* pipe, QRhiShaderResourceBindings* srb) {
        const int groups = static_cast<int>((cells_ + 255) / 256);
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->updateDynamicBuffer(muln_ubo_.data(), 0, sizeof(MulParams), &muln_);
        cb->beginComputePass(u);
        cb->setComputePipeline(pipe);
        cb->setShaderResources(srb);
        cb->dispatch(groups, 1, 1);
        cb->endComputePass();
        rhi_->endOffscreenFrame();
    }

    // A kGroups-wide reduction + readback of `buf` (bytes) into rb; updates muln_.
    void run_reduce(QRhiComputePipeline* pipe, QRhiShaderResourceBindings* srb,
                    QRhiBuffer* buf, quint32 bytes, QRhiReadbackResult* rb) {
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->updateDynamicBuffer(muln_ubo_.data(), 0, sizeof(MulParams), &muln_);
        cb->beginComputePass(u);
        cb->setComputePipeline(pipe);
        cb->setShaderResources(srb);
        cb->dispatch(kGroups, 1, 1);
        QRhiResourceUpdateBatch* d = rhi_->nextResourceUpdateBatch();
        d->readBackBuffer(buf, 0, bytes, rb);
        cb->endComputePass(d);
        rhi_->endOffscreenFrame();
    }

    // Lazily build the inner-product pipeline (if not already built by deflation).
    bool ensure_inner() {
        if (!inner_pipe_.isNull()) { return true; }
        const QShader innercs = load_qsb(QStringLiteral(":/shaders/inner_product.comp.qsb"));
        if (!innercs.isValid()) { return false; }
        using B = QRhiShaderResourceBinding;
        const auto cs = B::ComputeStage;
        inner_layout_srb_.reset(rhi_->newShaderResourceBindings());
        inner_layout_srb_->setBindings({ B::bufferLoad(0, cs, psi_.data()),
                                         B::uniformBuffer(1, cs, muln_ubo_.data()),
                                         B::bufferLoadStore(2, cs, partials_.data()),
                                         B::bufferLoad(3, cs, psi_.data()) });
        if (!inner_layout_srb_->create()) { return false; }
        inner_pipe_.reset(rhi_->newComputePipeline());
        inner_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, innercs));
        inner_pipe_->setShaderResourceBindings(inner_layout_srb_.data());
        return inner_pipe_->create();
    }

    // Lazily build the dipole pipeline + its UBO and 6-float partials buffer.
    bool ensure_dipole() {
        if (!dipole_pipe_.isNull()) { return true; }
        const QShader dipcs = load_qsb(QStringLiteral(":/shaders/dipole.comp.qsb"));
        if (!dipcs.isValid()) { return false; }
        dipole_ubo_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                          sizeof(DipoleParams)));
        dipole_partials_.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer,
                                               static_cast<quint32>(6 * kGroups * sizeof(float))));
        if (!dipole_ubo_->create() || !dipole_partials_->create()) { return false; }
        using B = QRhiShaderResourceBinding;
        const auto cs = B::ComputeStage;
        dipole_layout_srb_.reset(rhi_->newShaderResourceBindings());
        dipole_layout_srb_->setBindings({ B::bufferLoad(0, cs, psi_.data()),
                                          B::uniformBuffer(1, cs, dipole_ubo_.data()),
                                          B::bufferLoadStore(2, cs, dipole_partials_.data()),
                                          B::bufferLoad(3, cs, psi_.data()) });
        if (!dipole_layout_srb_->create()) { return false; }
        dipole_pipe_.reset(rhi_->newComputePipeline());
        dipole_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, dipcs));
        dipole_pipe_->setShaderResourceBindings(dipole_layout_srb_.data());
        return dipole_pipe_->create();
    }

    // Lazily build the fp16 pack/unpack pipelines + the two decode scratch
    // buffers. Layout SRBs use the fp32 scratch as binding-6 placeholders (layout
    // compatibility ignores buffer size).
    bool ensure_fp16() {
        if (!unpack_pipe_.isNull()) { return true; }
        const QShader packcs = load_qsb(QStringLiteral(":/shaders/pack_half.comp.qsb"));
        const QShader unpackcs = load_qsb(QStringLiteral(":/shaders/unpack_half.comp.qsb"));
        if (!packcs.isValid() || !unpackcs.isValid()) { return false; }
        for (int s = 0; s < 2; ++s) {
            decode_scratch_[s].reset(rhi_->newBuffer(
                QRhiBuffer::Static, QRhiBuffer::StorageBuffer,
                static_cast<quint32>(2 * cells_ * sizeof(float))));
            if (!decode_scratch_[s]->create()) { return false; }
        }
        using B = QRhiShaderResourceBinding;
        const auto cs = B::ComputeStage;
        pack_layout_srb_.reset(rhi_->newShaderResourceBindings());
        pack_layout_srb_->setBindings({ B::bufferLoad(0, cs, decode_scratch_[0].data()),
                                        B::uniformBuffer(1, cs, muln_ubo_.data()),
                                        B::bufferLoadStore(6, cs, decode_scratch_[1].data()) });
        unpack_layout_srb_.reset(rhi_->newShaderResourceBindings());
        unpack_layout_srb_->setBindings({ B::bufferLoadStore(0, cs, decode_scratch_[0].data()),
                                          B::uniformBuffer(1, cs, muln_ubo_.data()),
                                          B::bufferLoad(6, cs, decode_scratch_[1].data()) });
        if (!pack_layout_srb_->create() || !unpack_layout_srb_->create()) { return false; }
        pack_pipe_.reset(rhi_->newComputePipeline());
        pack_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, packcs));
        pack_pipe_->setShaderResourceBindings(pack_layout_srb_.data());
        unpack_pipe_.reset(rhi_->newComputePipeline());
        unpack_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, unpackcs));
        unpack_pipe_->setShaderResourceBindings(unpack_layout_srb_.data());
        return pack_pipe_->create() && unpack_pipe_->create();
    }

    // The shear parameter block: in the mixed representation (FFT along
    // freq_axis), shift each line by coeff * (its coord_axis coordinate).
    ShearParams make_shear_params(int freq_axis, int coord_axis, double coeff) const {
        const ses::Grid1D& fa = axis_grid(freq_axis);
        const ses::Grid1D& ca = axis_grid(coord_axis);
        const double two_pi = 6.283185307179586;
        ShearParams sp{};
        sp.n = static_cast<quint32>(cells_);
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

    // Stage everything one (or many) three-shear rotation(s) about the axis
    // perpendicular to (b, c) need in this frame: the two shear parameter sets
    // (set0 = (b, c, -tan(theta/2)) used twice, set1 = (c, b, sin(theta))),
    // the in-plane FFT axis params, and the conj scales.
    void stage_rotation_ubos(QRhiResourceUpdateBatch* u, int b, int c, double theta) {
        const double t = std::tan(0.5 * theta);
        const double sn = std::sin(theta);
        const ShearParams s0 = make_shear_params(b, c, -t);
        const ShearParams s1 = make_shear_params(c, b, sn);
        u->updateDynamicBuffer(shear_ubo_[0].data(), 0, sizeof(ShearParams), &s0);
        u->updateDynamicBuffer(shear_ubo_[1].data(), 0, sizeof(ShearParams), &s1);
        u->updateDynamicBuffer(fft_ubo_[b].data(), 0, sizeof(FftParams), &fftp_[b]);
        u->updateDynamicBuffer(fft_ubo_[c].data(), 0, sizeof(FftParams), &fftp_[c]);
        u->updateDynamicBuffer(conj1_ubo_.data(), 0, sizeof(ConjParams), &conj1_);
        u->updateDynamicBuffer(conjA_ubo_.data(), 0, sizeof(ConjParams), &conjA_);
    }

    // Record one staged shear in the current pass: FFT along freq_axis,
    // phase-shift (shear_srb_[which]), then the inverse FFT along that axis
    // (conj -> FFT -> conj/n).
    void record_shear(QRhiCommandBuffer* cb, int mul_groups, int which, int freq_axis) {
        fft_axis(cb, freq_axis);
        cb->setComputePipeline(shear_pipe_.data());
        cb->setShaderResources(shear_srb_[which].data());
        cb->dispatch(mul_groups, 1, 1);
        conjscale(cb, conj1_srb_.data(), mul_groups);  // conj (scale 1)
        fft_axis(cb, freq_axis);
        conjscale(cb, conjA_srb_.data(), mul_groups);  // conj + scale 1/n
    }

    // Record one full three-shear rotation (set0 on b, set1 on c, set0 on b).
    void record_rotation(QRhiCommandBuffer* cb, int mul_groups, int b, int c) {
        record_shear(cb, mul_groups, 0, b);
        record_shear(cb, mul_groups, 1, c);
        record_shear(cb, mul_groups, 0, b);
    }

    const ses::Grid1D& axis_grid(int a) const {
        return a == 0 ? grid_.x : (a == 1 ? grid_.y : grid_.z);
    }

    QRhi* rhi_ = nullptr;
    ses::Grid3D grid_{};
    int n_ = 0;
    std::size_t cells_ = 0;
    double dtau_ = 0.0;
    double cell_volume_ = 0.0;
    MulParams muln_{};
    ConjParams conj1_{};
    ConjParams conjN_{};
    FftParams fftp_[3]{};

    QScopedPointer<QRhiBuffer> psi_, half_, kin_;
    QScopedPointer<QRhiBuffer> muln_ubo_, conj1_ubo_, conjN_ubo_, fft_ubo_[3];
    QScopedPointer<QRhiShaderResourceBindings> mul_half_srb_, mul_kin_srb_, conj1_srb_,
        conjN_srb_, fft_srb_[3];
    QScopedPointer<QRhiComputePipeline> mul_pipe_, conj_pipe_, fft_pipe_;

    // Dipole kick (driven_step): exp(-i theta axis.r) applied in place; the
    // params UBO holds one dynamic-offset slot per half-kick of a batch.
    quint32 kick_stride_ = 256;
    QScopedPointer<QRhiBuffer> kick_ubo_;
    QScopedPointer<QRhiShaderResourceBindings> kick_srb_;
    QScopedPointer<QRhiComputePipeline> kick_pipe_;

    // Imaginary-time relaxation (set_relax_tables / relax_step).
    QScopedPointer<QRhiBuffer> relax_half_, relax_kin_, partials_, scale_ubo_;
    QScopedPointer<QRhiShaderResourceBindings> relax_half_srb_, relax_kin_srb_, norm_srb_,
        scale_srb_;
    QScopedPointer<QRhiComputePipeline> norm_pipe_, scale_pipe_;

    // Per-state ops (deflation / collapse / mask): pipelines built once from the
    // first state's SRBs (shared layout); axpy_ubo_ carries c = <phi|psi>.
    QScopedPointer<QRhiBuffer> axpy_ubo_;
    QScopedPointer<QRhiComputePipeline> inner_pipe_, axpy_pipe_, copy_pipe_;

    // Semiclassical radiation (set_potential_gradient / mean_force): grad V
    // field (vec4 per cell) + vec4 partials, reduced by the mean-force kernel.
    QScopedPointer<QRhiBuffer> grad_v_buf_, force_partials_;
    QScopedPointer<QRhiShaderResourceBindings> force_srb_;
    QScopedPointer<QRhiComputePipeline> force_pipe_;

    // Magnetic (ensure_magnetic / rotate_axis_shear / magnetic_step): the shear
    // kernel (two staged parameter sets) plus a per-axis inverse-FFT
    // conj-scale (scale 1/n, not 1/cells).
    ConjParams conjA_{};
    QScopedPointer<QRhiBuffer> shear_ubo_[2], conjA_ubo_;
    QScopedPointer<QRhiShaderResourceBindings> shear_srb_[2], conjA_srb_;
    QScopedPointer<QRhiComputePipeline> shear_pipe_;

    // Orbital synthesis (synthesize_into_psi): the radial u_nl(r) table plus the
    // synth kernel writing (u/r)Ylm into psi.
    quint32 radial_bytes_ = 0;
    QScopedPointer<QRhiBuffer> radial_buf_, synth_ubo_;
    QScopedPointer<QRhiShaderResourceBindings> synth_srb_;
    QScopedPointer<QRhiComputePipeline> synth_pipe_;

    // SSBO -> 3D volume texture bridge (write_psi_to_volume / bridge_roundtrip):
    // the RGBA32F volume the renderer samples, plus a scratch buffer + load
    // pipeline that reads it back for verification.
    QScopedPointer<QRhiTexture> volume_tex_;
    QScopedPointer<QRhiBuffer> scratch_bridge_buf_, bridge_ubo_;
    QScopedPointer<QRhiShaderResourceBindings> store_srb_, load_srb_;
    QScopedPointer<QRhiComputePipeline> store_pipe_, load_pipe_;

    // Atlas: resident fp32/fp16 eigenstates + the fp16 codec and the fp32
    // consumers (inner product, dipole) that read them (fp16 via decode-on-use).
    std::vector<std::unique_ptr<AtlasState>> atlas_;
    QScopedPointer<QRhiBuffer> decode_scratch_[2];
    QScopedPointer<QRhiShaderResourceBindings> pack_layout_srb_, unpack_layout_srb_;
    QScopedPointer<QRhiComputePipeline> pack_pipe_, unpack_pipe_;
    QScopedPointer<QRhiShaderResourceBindings> inner_layout_srb_;
    QScopedPointer<QRhiBuffer> dipole_ubo_, dipole_partials_;
    QScopedPointer<QRhiShaderResourceBindings> dipole_layout_srb_;
    QScopedPointer<QRhiComputePipeline> dipole_pipe_;

    // Orbital-free angular projection (set_projection_index / project_psi /
    // project_amplitude): static counting-sort index + g_lm deposit output.
    int proj_nr_ = 0;
    int proj_ncomp_ = 0;
    double proj_h_radial_ = 0.0;
    QScopedPointer<QRhiBuffer> proj_sorted_buf_, proj_binoff_buf_, glm_buf_, proj_ubo_;
    QScopedPointer<QRhiShaderResourceBindings> proj_srb_;
    QScopedPointer<QRhiComputePipeline> proj_pipe_;
    std::vector<std::vector<ses::Complex<double>>> glm_host_;

#ifdef SES_HAVE_VKFFT
    // VkFFT (M4): the 3D-transform plan on psi's native VkBuffer. The Vk
    // handles are members because VkFFTConfiguration stores POINTERS to them.
    bool use_vkfft_ = true;
    bool vkfft_ready_ = false;
    bool vkfft_failed_ = false;
    VkFFTApplication vkfft_app_{};
    VkBuffer vkfft_psi_buf_ = VK_NULL_HANDLE;
    quint64 vkfft_buf_size_ = 0;
    VkPhysicalDevice vkfft_phys_dev_ = VK_NULL_HANDLE;
    VkDevice vkfft_dev_ = VK_NULL_HANDLE;
    VkQueue vkfft_queue_ = VK_NULL_HANDLE;
    VkCommandPool vkfft_pool_ = VK_NULL_HANDLE;
    VkFence vkfft_fence_ = VK_NULL_HANDLE;
    QVulkanDeviceFunctions* vkfft_df_ = nullptr;
#endif
};

}  // namespace ses_qrhi
