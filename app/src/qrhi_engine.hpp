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

#include <core/complex.hpp>
#include <core/drive.hpp>
#include <core/grid.hpp>
#include <core/vec.hpp>

#include <rhi/qrhi.h>

#include <QFile>
#include <QString>

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
        if (grid.y.n != n_ || grid.z.n != n_) {
            std::fprintf(stderr, "QrhiEngine: only cubic grids supported\n");
            return false;
        }

        const QShader mulcs = load_qsb(QStringLiteral(":/shaders/phase_multiply.comp.qsb"));
        const QShader conjcs = load_qsb(QStringLiteral(":/shaders/conj_scale.comp.qsb"));
        const QShader fftcs =
            load_qsb(QStringLiteral(":/shaders/fft_line%1.comp.qsb").arg(n_));
        const QShader kickcs = load_qsb(QStringLiteral(":/shaders/dipole_kick.comp.qsb"));
        if (!mulcs.isValid() || !conjcs.isValid() || !fftcs.isValid() || !kickcs.isValid()) {
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
        kick_ubo_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                        sizeof(KickParams)));
        if (!kick_ubo_->create()) { return false; }

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
        kick_srb_.reset(rhi_->newShaderResourceBindings());
        kick_srb_->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                 B::uniformBuffer(1, cs, kick_ubo_.data()) });
        if (!mul_half_srb_->create() || !mul_kin_srb_->create() || !conj1_srb_->create() ||
            !conjN_srb_->create() || !fft_srb_[0]->create() || !fft_srb_[1]->create() ||
            !fft_srb_[2]->create() || !kick_srb_->create()) {
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

        // Upload the static buffers (psi/half/kin) once.
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return false; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->uploadStaticBuffer(psi_.data(), psi_f.data());
        u->uploadStaticBuffer(half_.data(), half_f.data());
        u->uploadStaticBuffer(kin_.data(), kin_f.data());
        cb->resourceUpdate(u);
        rhi_->endOffscreenFrame();
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
        cb->beginComputePass(u);
        for (int s = 0; s < nsteps; ++s) {
            run_step_body(cb, mul_groups, mul_half_srb_.data(), mul_kin_srb_.data());
        }
        cb->endComputePass();
        rhi_->endOffscreenFrame();
    }

    // Driven Strang steps (T3): kick(t) . step . kick(t+dt), the tested
    // core/drive.hpp composition. theta = amplitude cos(omega t) dt/2. Each
    // half-kick and each step body run in their own frame (a kick's theta
    // differs each time, and a Dynamic UBO cannot change mid-pass).
    void driven_step(const ses::DipoleDrive& d, double t0, double dt, int nsteps) {
        const int mul_groups = static_cast<int>((cells_ + 255) / 256);
        for (int s = 0; s < nsteps; ++s) {
            const double t = t0 + s * dt;
            kick(d.axis, d.amplitude * std::cos(d.omega * t) * 0.5 * dt);
            QRhiCommandBuffer* cb = nullptr;
            if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return; }
            QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
            update_step_ubos(u);
            cb->beginComputePass(u);
            run_step_body(cb, mul_groups, mul_half_srb_.data(), mul_kin_srb_.data());
            cb->endComputePass();
            rhi_->endOffscreenFrame();
            kick(d.axis, d.amplitude * std::cos(d.omega * (t + dt)) * 0.5 * dt);
        }
    }

    struct RelaxStats { double energy = 0.0; };

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
        const QShader normcs = load_qsb(QStringLiteral(":/shaders/norm_peak.comp.qsb"));
        const QShader scalecs = load_qsb(QStringLiteral(":/shaders/scale.comp.qsb"));
        if (!normcs.isValid() || !scalecs.isValid()) { return false; }

        relax_half_.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, bytes));
        relax_kin_.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, bytes));
        partials_.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer,
                                        static_cast<quint32>(2 * kGroups * sizeof(float))));
        scale_ubo_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                         sizeof(ConjParams)));
        if (!relax_half_->create() || !relax_kin_->create() || !partials_->create() ||
            !scale_ubo_->create()) { return false; }

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
        norm_srb_.reset(rhi_->newShaderResourceBindings());
        norm_srb_->setBindings({ B::bufferLoad(0, cs, psi_.data()),
                                 B::uniformBuffer(1, cs, muln_ubo_.data()),
                                 B::bufferLoadStore(2, cs, partials_.data()) });
        scale_srb_.reset(rhi_->newShaderResourceBindings());
        scale_srb_->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                  B::uniformBuffer(1, cs, scale_ubo_.data()) });
        if (!relax_half_srb_->create() || !relax_kin_srb_->create() || !norm_srb_->create() ||
            !scale_srb_->create()) { return false; }

        norm_pipe_.reset(rhi_->newComputePipeline());
        norm_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, normcs));
        norm_pipe_->setShaderResourceBindings(norm_srb_.data());
        scale_pipe_.reset(rhi_->newComputePipeline());
        scale_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, scalecs));
        scale_pipe_->setShaderResourceBindings(scale_srb_.data());
        if (!norm_pipe_->create() || !scale_pipe_->create()) { return false; }

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
            cb->beginComputePass(u);
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

    // ---- deflation (deflated imaginary-time relax) ---------------------
    // Upload an auxiliary state (a lower eigenstate to project out) into its own
    // Static buffer and build its inner/axpy/copy bindings. Returns a handle
    // (index) for relax_deflated_step / inner_with_psi / copy_into_psi, or -1 on
    // failure. Call after set_relax_tables (reuses the norm partials buffer).
    int create_state_buffer(const std::vector<ses::Complex<double>>& state) {
        if (partials_.isNull()) {
            std::fprintf(stderr, "QrhiEngine: create_state_buffer needs set_relax_tables\n");
            return -1;
        }
        // The projection-coefficient UBO is shared across aux states; create it
        // once, BEFORE the axpy SRB names it (a null binding would crash create).
        if (axpy_ubo_.isNull()) {
            axpy_ubo_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                            sizeof(AxpyParams)));
            if (!axpy_ubo_->create()) { return -1; }
        }
        const std::vector<float> sf = to_rg32f(state);
        const quint32 bytes = static_cast<quint32>(sf.size() * sizeof(float));
        auto a = std::make_unique<AuxState>();
        a->buf.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, bytes));
        if (!a->buf->create()) { return -1; }

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
        if (!a->inner_srb->create() || !a->axpy_srb->create() || !a->copy_srb->create()) {
            return -1;
        }
        if (!ensure_deflation_pipelines(a.get())) { return -1; }

        // Upload the (write-once) state.
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return -1; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->uploadStaticBuffer(a->buf.data(), sf.data());
        cb->resourceUpdate(u);
        rhi_->endOffscreenFrame();

        aux_.push_back(std::move(a));
        return static_cast<int>(aux_.size()) - 1;
    }

    // <phi|psi> = sum conj(phi)*psi * dV, the physical inner product of the live
    // psi with aux state `handle` (the QRhi analog of GpuEngine::inner_with_psi).
    ses::Complex<double> inner_with_psi(int handle) {
        AuxState* a = aux_.at(static_cast<std::size_t>(handle)).get();
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return {}; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->updateDynamicBuffer(muln_ubo_.data(), 0, sizeof(MulParams), &muln_);
        cb->beginComputePass(u);
        cb->setComputePipeline(inner_pipe_.data());
        cb->setShaderResources(a->inner_srb.data());
        cb->dispatch(kGroups, 1, 1);
        QRhiReadbackResult rb;
        QRhiResourceUpdateBatch* d = rhi_->nextResourceUpdateBatch();
        d->readBackBuffer(partials_.data(), 0,
                          static_cast<quint32>(2 * kGroups * sizeof(float)), &rb);
        cb->endComputePass(d);
        rhi_->endOffscreenFrame();
        const float* p = reinterpret_cast<const float*>(rb.data.constData());
        double re = 0.0;
        double im = 0.0;
        for (int gi = 0; gi < kGroups; ++gi) { re += p[2 * gi]; im += p[2 * gi + 1]; }
        return ses::Complex<double>{ re * cell_volume_, im * cell_volume_ };
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
                cb->beginComputePass(u);
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

    // psi <- src, the auxiliary state `handle` copied bitwise into psi (the
    // quantum-jump collapse path; GpuEngine::copy_into_psi via glCopyBufferSubData).
    void copy_into_psi(int handle) {
        const int groups = static_cast<int>((cells_ + 255) / 256);
        AuxState* a = aux_.at(static_cast<std::size_t>(handle)).get();
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
    // GPU transcription of core/rotation.hpp rotate_axis. In place.
    void rotate_axis_shear(int axis, double theta) {
        if (!ensure_magnetic()) { return; }
        const int b = (axis + 1) % 3;  // in-plane axes (b x c = axis)
        const int c = (axis + 2) % 3;
        const double t = std::tan(0.5 * theta);
        const double s = std::sin(theta);
        apply_shear(b, c, -t);
        apply_shear(c, b, s);
        apply_shear(b, c, -t);
    }
    void rotate_z_shear(double theta) { rotate_axis_shear(2, theta); }

    // Magnetic Strang step: R(a) . real-step . R(a), a = (B/2)(dt/2), rotating
    // about the field `axis`. half_ must hold the diamagnetic-augmented table
    // (set_half_potential). Mirrors GpuEngine::magnetic_step.
    void magnetic_step(int axis, double half_angle, int nsteps) {
        if (!ensure_magnetic()) { return; }
        for (int s = 0; s < nsteps; ++s) {
            rotate_axis_shear(axis, half_angle);
            step(1);
            rotate_axis_shear(axis, half_angle);
        }
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

    // An auxiliary (lower) eigenstate resident on the GPU plus its bindings for
    // the deflation kernels. Heap-owned (unique_ptr) so the QScopedPointer members
    // never move; the SRBs are per-state (they name this buffer as phi/src).
    struct AuxState {
        QScopedPointer<QRhiBuffer> buf;
        QScopedPointer<QRhiShaderResourceBindings> inner_srb, axpy_srb, copy_srb;
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
    void run_step_body(QRhiCommandBuffer* cb, int mul_groups,
                       QRhiShaderResourceBindings* half_srb,
                       QRhiShaderResourceBindings* kin_srb) {
        multiply(cb, half_srb, mul_groups);
        fft3(cb);
        multiply(cb, kin_srb, mul_groups);
        conjscale(cb, conj1_srb_.data(), mul_groups);
        fft3(cb);
        conjscale(cb, conjN_srb_.data(), mul_groups);
        multiply(cb, half_srb, mul_groups);
    }

    // psi <- exp(-i theta axis.r) psi, one dipole half-kick (own frame).
    void kick(const ses::Vec3d& axis, double theta) {
        const int groups = static_cast<int>((cells_ + 255) / 256);
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
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->updateDynamicBuffer(kick_ubo_.data(), 0, sizeof(KickParams), &kp);
        cb->beginComputePass(u);
        cb->setComputePipeline(kick_pipe_.data());
        cb->setShaderResources(kick_srb_.data());
        cb->dispatch(groups, 1, 1);
        cb->endComputePass();
        rhi_->endOffscreenFrame();
    }

    // Frame N: norm reduction (sum |psi|^2 dV) + partials readback -> host norm,
    // then Frame S: psi <- psi / sqrt(norm). Returns the free-energy estimate
    // from the pre-renorm norm (e^{-2 E dtau}). Shared by relax / relax_deflated.
    RelaxStats renormalize_and_estimate(int mul_groups) {
        RelaxStats stats;
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return stats; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->updateDynamicBuffer(muln_ubo_.data(), 0, sizeof(MulParams), &muln_);
        cb->beginComputePass(u);
        cb->setComputePipeline(norm_pipe_.data());
        cb->setShaderResources(norm_srb_.data());
        cb->dispatch(kGroups, 1, 1);
        QRhiReadbackResult rb;
        QRhiResourceUpdateBatch* d = rhi_->nextResourceUpdateBatch();
        d->readBackBuffer(partials_.data(), 0,
                          static_cast<quint32>(2 * kGroups * sizeof(float)), &rb);
        cb->endComputePass(d);
        rhi_->endOffscreenFrame();

        const float* p = reinterpret_cast<const float*>(rb.data.constData());
        double sum = 0.0;
        for (int gi = 0; gi < kGroups; ++gi) { sum += p[2 * gi]; }
        const double norm_sq = sum * cell_volume_;
        const double inv = (norm_sq > 0.0) ? 1.0 / std::sqrt(norm_sq) : 0.0;
        stats.energy = (norm_sq > 0.0) ? -std::log(norm_sq) / (2.0 * dtau_) : 0.0;

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

    // psi <- psi - c*phi (aux `handle`), the Gram-Schmidt projection (own frame).
    void subtract_projection(int handle, double cre, double cim, int groups) {
        AuxState* a = aux_.at(static_cast<std::size_t>(handle)).get();
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

    // Build the inner/axpy/copy pipelines once, on the first create_state_buffer.
    // The pipelines take `first`'s SRBs as their layout template; every later aux
    // state's SRBs share that layout (same bindings, different buffer), so they
    // dispatch against the same pipeline.
    bool ensure_deflation_pipelines(AuxState* first) {
        if (!axpy_pipe_.isNull()) { return true; }
        const QShader innercs = load_qsb(QStringLiteral(":/shaders/inner_product.comp.qsb"));
        const QShader axpycs = load_qsb(QStringLiteral(":/shaders/axpy.comp.qsb"));
        const QShader copycs = load_qsb(QStringLiteral(":/shaders/copy_state.comp.qsb"));
        if (!innercs.isValid() || !axpycs.isValid() || !copycs.isValid()) { return false; }
        inner_pipe_.reset(rhi_->newComputePipeline());
        inner_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, innercs));
        inner_pipe_->setShaderResourceBindings(first->inner_srb.data());
        axpy_pipe_.reset(rhi_->newComputePipeline());
        axpy_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, axpycs));
        axpy_pipe_->setShaderResourceBindings(first->axpy_srb.data());
        copy_pipe_.reset(rhi_->newComputePipeline());
        copy_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, copycs));
        copy_pipe_->setShaderResourceBindings(first->copy_srb.data());
        return inner_pipe_->create() && axpy_pipe_->create() && copy_pipe_->create();
    }

    // Lazily build the shear pipeline + its UBO/SRB and the per-axis inverse-FFT
    // conj-scale (scale 1/n, not 1/cells) on the first magnetic call.
    bool ensure_magnetic() {
        if (!shear_pipe_.isNull()) { return true; }
        const QShader shearcs = load_qsb(QStringLiteral(":/shaders/shear.comp.qsb"));
        if (!shearcs.isValid()) { return false; }
        using B = QRhiShaderResourceBinding;
        const auto cs = B::ComputeStage;
        shear_ubo_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                         sizeof(ShearParams)));
        conjA_ubo_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                         sizeof(ConjParams)));
        if (!shear_ubo_->create() || !conjA_ubo_->create()) { return false; }
        shear_srb_.reset(rhi_->newShaderResourceBindings());
        shear_srb_->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                  B::uniformBuffer(1, cs, shear_ubo_.data()) });
        conjA_srb_.reset(rhi_->newShaderResourceBindings());
        conjA_srb_->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                  B::uniformBuffer(1, cs, conjA_ubo_.data()) });
        if (!shear_srb_->create() || !conjA_srb_->create()) { return false; }
        shear_pipe_.reset(rhi_->newComputePipeline());
        shear_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, shearcs));
        shear_pipe_->setShaderResourceBindings(shear_srb_.data());
        // Per-axis inverse FFT scales by 1/n (the single transformed axis).
        conjA_ = ConjParams{ static_cast<quint32>(cells_), 1.0f / static_cast<float>(n_),
                             0.0f, 0.0f };
        return shear_pipe_->create();
    }

    // One shear (own frame): FFT along freq_axis, phase-shift by coeff*coord,
    // then inverse FFT along that axis (conj -> FFT -> conj/n). The freq/conj
    // UBOs are constant per axis; only shear_ubo_ carries this shear's geometry.
    void apply_shear(int freq_axis, int coord_axis, double coeff) {
        const int mul_groups = static_cast<int>((cells_ + 255) / 256);
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

        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->updateDynamicBuffer(shear_ubo_.data(), 0, sizeof(ShearParams), &sp);
        u->updateDynamicBuffer(conj1_ubo_.data(), 0, sizeof(ConjParams), &conj1_);
        u->updateDynamicBuffer(conjA_ubo_.data(), 0, sizeof(ConjParams), &conjA_);
        u->updateDynamicBuffer(fft_ubo_[freq_axis].data(), 0, sizeof(FftParams),
                               &fftp_[freq_axis]);
        cb->beginComputePass(u);
        fft_axis(cb, freq_axis);
        cb->setComputePipeline(shear_pipe_.data());
        cb->setShaderResources(shear_srb_.data());
        cb->dispatch(mul_groups, 1, 1);
        conjscale(cb, conj1_srb_.data(), mul_groups);  // conj (scale 1)
        fft_axis(cb, freq_axis);
        conjscale(cb, conjA_srb_.data(), mul_groups);  // conj + scale 1/n
        cb->endComputePass();
        rhi_->endOffscreenFrame();
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

    // Dipole kick (driven_step): exp(-i theta axis.r) applied in place.
    QScopedPointer<QRhiBuffer> kick_ubo_;
    QScopedPointer<QRhiShaderResourceBindings> kick_srb_;
    QScopedPointer<QRhiComputePipeline> kick_pipe_;

    // Imaginary-time relaxation (set_relax_tables / relax_step).
    QScopedPointer<QRhiBuffer> relax_half_, relax_kin_, partials_, scale_ubo_;
    QScopedPointer<QRhiShaderResourceBindings> relax_half_srb_, relax_kin_srb_, norm_srb_,
        scale_srb_;
    QScopedPointer<QRhiComputePipeline> norm_pipe_, scale_pipe_;

    // Deflation (create_state_buffer / relax_deflated_step / copy_into_psi).
    // The pipelines are built once from the first aux state's SRBs (shared
    // layout); axpy_ubo_ carries the projection coefficient c = <phi|psi>.
    std::vector<std::unique_ptr<AuxState>> aux_;
    QScopedPointer<QRhiBuffer> axpy_ubo_;
    QScopedPointer<QRhiComputePipeline> inner_pipe_, axpy_pipe_, copy_pipe_;

    // Magnetic (ensure_magnetic / rotate_axis_shear / magnetic_step): the shear
    // kernel plus a per-axis inverse-FFT conj-scale (scale 1/n, not 1/cells).
    ConjParams conjA_{};
    QScopedPointer<QRhiBuffer> shear_ubo_, conjA_ubo_;
    QScopedPointer<QRhiShaderResourceBindings> shear_srb_, conjA_srb_;
    QScopedPointer<QRhiComputePipeline> shear_pipe_;
};

}  // namespace ses_qrhi
