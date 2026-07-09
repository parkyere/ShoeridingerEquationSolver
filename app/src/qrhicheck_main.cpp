// sesolver_qrhicheck: QRhi (Vulkan backend) verification harness -- the first
// seed of the GL->QRhi migration (M1). It is the Vulkan-backend analog of
// sesolver_gpucheck: each decorated Vulkan-GLSL kernel is baked to SPIR-V by
// qsb (embedded via qt_add_shaders), run through QRhi's Vulkan backend, and
// compared against the SAME host reference the GL harness uses -- identical
// input data and tolerance -- so a green result proves the QRhi/Vulkan path
// reproduces the unit-tested kernel bit-for-bit-enough.
//
// Exit 77 = SKIP (ctest SKIP_RETURN_CODE) when no Vulkan device is available,
// mirroring sesolver_gpucheck's no-context convention.

#include "qrhi_engine.hpp"

#include <core/complex.hpp>
#include <core/drive.hpp>
#include <core/fft.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/harmonics.hpp>
#include <core/imaginary_time.hpp>
#include <core/magnetic.hpp>
#include <core/radial.hpp>
#include <core/rotation.hpp>
#include <core/potential.hpp>
#include <core/propagator.hpp>
#include <core/vec.hpp>
#include <core/wavepacket.hpp>

#include <rhi/qrhi.h>

#include <QGuiApplication>
#include <QVulkanInstance>
#include <QFile>
#include <QScopedPointer>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

// Complex<double> -> interleaved rg32f, matching ses_gpu::to_rg32f (the GL
// harness's upload format), so both backends see identical fp32 inputs.
std::vector<float> to_rg32f(const std::vector<ses::Complex<double>>& src) {
    std::vector<float> out(2 * src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        out[2 * i] = static_cast<float>(src[i].real());
        out[2 * i + 1] = static_cast<float>(src[i].imag());
    }
    return out;
}

QShader load_qsb(const QString& path) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) {
        return {};
    }
    return QShader::fromSerialized(f.readAll());
}

// psi <- psi (complex-*) phase, exactly kPhaseMultiplySrc -- same data and
// tolerance (1e-5) as sesolver_gpucheck's check_phase_multiply.
bool check_phase_multiply(QRhi* rhi) {
    const std::size_t n = 4096;
    std::vector<ses::Complex<double>> psi_d(n);
    std::vector<ses::Complex<double>> phase_d(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        psi_d[i] = ses::Complex<double>{std::sin(0.37 * x) + 0.2, std::cos(1.13 * x) - 0.1};
        phase_d[i] = ses::Complex<double>{std::cos(2.9 * x), std::sin(2.9 * x)};
    }
    const std::vector<float> psi_f = to_rg32f(psi_d);
    const std::vector<float> phase_f = to_rg32f(phase_d);
    const quint32 bytes = static_cast<quint32>(psi_f.size() * sizeof(float));

    QShader cs = load_qsb(QStringLiteral(":/shaders/phase_multiply.comp.qsb"));
    if (!cs.isValid()) {
        std::fprintf(stderr, "phase_multiply.comp.qsb missing/invalid\n");
        return false;
    }

    QScopedPointer<QRhiBuffer> psi(
        rhi->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, bytes));
    QScopedPointer<QRhiBuffer> phase(
        rhi->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, bytes));
    struct alignas(16) Params { quint32 n; quint32 pad0, pad1, pad2; };
    Params params{ static_cast<quint32>(n), 0, 0, 0 };
    QScopedPointer<QRhiBuffer> ubo(
        rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(Params)));
    if (!psi->create() || !phase->create() || !ubo->create()) {
        std::fprintf(stderr, "buffer create failed\n");
        return false;
    }

    QScopedPointer<QRhiShaderResourceBindings> srb(rhi->newShaderResourceBindings());
    srb->setBindings({
        QRhiShaderResourceBinding::bufferLoadStore(0, QRhiShaderResourceBinding::ComputeStage,
                                                   psi.data()),
        QRhiShaderResourceBinding::bufferLoad(1, QRhiShaderResourceBinding::ComputeStage,
                                              phase.data()),
        QRhiShaderResourceBinding::uniformBuffer(2, QRhiShaderResourceBinding::ComputeStage,
                                                 ubo.data()),
    });
    if (!srb->create()) {
        std::fprintf(stderr, "srb create failed\n");
        return false;
    }

    QScopedPointer<QRhiComputePipeline> pipe(rhi->newComputePipeline());
    pipe->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, cs));
    pipe->setShaderResourceBindings(srb.data());
    if (!pipe->create()) {
        std::fprintf(stderr, "compute pipeline create failed\n");
        return false;
    }

    QRhiCommandBuffer* cb = nullptr;
    if (rhi->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) {
        std::fprintf(stderr, "beginOffscreenFrame failed\n");
        return false;
    }
    QRhiResourceUpdateBatch* up = rhi->nextResourceUpdateBatch();
    up->uploadStaticBuffer(psi.data(), psi_f.data());
    up->uploadStaticBuffer(phase.data(), phase_f.data());
    up->updateDynamicBuffer(ubo.data(), 0, sizeof(Params), &params);

    cb->beginComputePass(up);
    cb->setComputePipeline(pipe.data());
    cb->setShaderResources(srb.data());
    cb->dispatch(static_cast<int>((n + 255) / 256), 1, 1);

    QRhiReadbackResult rb;
    QRhiResourceUpdateBatch* down = rhi->nextResourceUpdateBatch();
    down->readBackBuffer(psi.data(), 0, bytes, &rb);
    cb->endComputePass(down);
    rhi->endOffscreenFrame();

    if (rb.data.size() != static_cast<int>(bytes)) {
        std::fprintf(stderr, "readback size %lld != %u\n",
                     static_cast<long long>(rb.data.size()), bytes);
        return false;
    }
    const float* out = reinterpret_cast<const float*>(rb.data.constData());
    double max_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const ses::Complex<double> expected = psi_d[i] * phase_d[i];
        max_err = std::max(max_err, std::abs(out[2 * i] - expected.real()));
        max_err = std::max(max_err, std::abs(out[2 * i + 1] - expected.imag()));
    }
    const bool pass = max_err < 1e-5;
    std::printf("phase-multiply kernel (QRhi/Vulkan): max |gpu - cpu| = %.3e  [%s]\n",
                max_err, pass ? "PASS" : "FAIL");
    return pass;
}

// data <- s * data, exactly kScaleSrc (fp32 drift renormalization).
bool check_scale(QRhi* rhi) {
    const std::size_t n = 4096;
    const float s = 0.5f;
    std::vector<ses::Complex<double>> d0(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        d0[i] = ses::Complex<double>{std::sin(0.7 * x) - 0.3, std::cos(0.21 * x) + 0.4};
    }
    const std::vector<float> in = to_rg32f(d0);
    const quint32 bytes = static_cast<quint32>(in.size() * sizeof(float));

    QShader cs = load_qsb(QStringLiteral(":/shaders/scale.comp.qsb"));
    if (!cs.isValid()) { std::fprintf(stderr, "scale.comp.qsb missing\n"); return false; }

    QScopedPointer<QRhiBuffer> data(
        rhi->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, bytes));
    struct alignas(16) Params { quint32 n; float scale; float pad0, pad1; };
    Params params{ static_cast<quint32>(n), s, 0.0f, 0.0f };
    QScopedPointer<QRhiBuffer> ubo(
        rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(Params)));
    if (!data->create() || !ubo->create()) { return false; }

    QScopedPointer<QRhiShaderResourceBindings> srb(rhi->newShaderResourceBindings());
    srb->setBindings({
        QRhiShaderResourceBinding::bufferLoadStore(0, QRhiShaderResourceBinding::ComputeStage,
                                                   data.data()),
        QRhiShaderResourceBinding::uniformBuffer(1, QRhiShaderResourceBinding::ComputeStage,
                                                 ubo.data()),
    });
    if (!srb->create()) { return false; }

    QScopedPointer<QRhiComputePipeline> pipe(rhi->newComputePipeline());
    pipe->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, cs));
    pipe->setShaderResourceBindings(srb.data());
    if (!pipe->create()) { return false; }

    QRhiCommandBuffer* cb = nullptr;
    if (rhi->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return false; }
    QRhiResourceUpdateBatch* up = rhi->nextResourceUpdateBatch();
    up->uploadStaticBuffer(data.data(), in.data());
    up->updateDynamicBuffer(ubo.data(), 0, sizeof(Params), &params);
    cb->beginComputePass(up);
    cb->setComputePipeline(pipe.data());
    cb->setShaderResources(srb.data());
    cb->dispatch(static_cast<int>((n + 255) / 256), 1, 1);
    QRhiReadbackResult rb;
    QRhiResourceUpdateBatch* down = rhi->nextResourceUpdateBatch();
    down->readBackBuffer(data.data(), 0, bytes, &rb);
    cb->endComputePass(down);
    rhi->endOffscreenFrame();

    const float* out = reinterpret_cast<const float*>(rb.data.constData());
    double max_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        max_err = std::max(max_err, std::abs(out[2 * i] - static_cast<double>(s) * d0[i].real()));
        max_err = std::max(max_err,
                           std::abs(out[2 * i + 1] - static_cast<double>(s) * d0[i].imag()));
    }
    const bool pass = max_err < 1e-5;
    std::printf("scale kernel (QRhi/Vulkan): max |gpu - cpu| = %.3e  [%s]\n",
                max_err, pass ? "PASS" : "FAIL");
    return pass;
}

// Fixed-order tree reduction of |psi|^2 into per-workgroup (sum, max) partials,
// finished on the host -- exactly kNormPeakSrc + run_norm_peak. The first
// REDUCTION kernel verified on the Vulkan backend. n is deliberately not a
// multiple of 256, exercising the strided grid-stride loop.
bool check_norm_peak(QRhi* rhi) {
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
    const quint32 psi_bytes = static_cast<quint32>(in.size() * sizeof(float));

    const int groups = 256;  // matches ses_gpu::kNormPeakGroups
    const quint32 part_bytes = static_cast<quint32>(2 * groups * sizeof(float));

    QShader cs = load_qsb(QStringLiteral(":/shaders/norm_peak.comp.qsb"));
    if (!cs.isValid()) { std::fprintf(stderr, "norm_peak.comp.qsb missing\n"); return false; }

    QScopedPointer<QRhiBuffer> psi(
        rhi->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, psi_bytes));
    QScopedPointer<QRhiBuffer> partials(
        rhi->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, part_bytes));
    struct alignas(16) Params { quint32 n; quint32 pad0, pad1, pad2; };
    Params params{ static_cast<quint32>(n), 0, 0, 0 };
    QScopedPointer<QRhiBuffer> ubo(
        rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(Params)));
    if (!psi->create() || !partials->create() || !ubo->create()) { return false; }

    QScopedPointer<QRhiShaderResourceBindings> srb(rhi->newShaderResourceBindings());
    srb->setBindings({
        QRhiShaderResourceBinding::bufferLoad(0, QRhiShaderResourceBinding::ComputeStage,
                                              psi.data()),
        QRhiShaderResourceBinding::uniformBuffer(1, QRhiShaderResourceBinding::ComputeStage,
                                                 ubo.data()),
        QRhiShaderResourceBinding::bufferLoadStore(2, QRhiShaderResourceBinding::ComputeStage,
                                                   partials.data()),
    });
    if (!srb->create()) { return false; }

    QScopedPointer<QRhiComputePipeline> pipe(rhi->newComputePipeline());
    pipe->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, cs));
    pipe->setShaderResourceBindings(srb.data());
    if (!pipe->create()) { return false; }

    QRhiCommandBuffer* cb = nullptr;
    if (rhi->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return false; }
    QRhiResourceUpdateBatch* up = rhi->nextResourceUpdateBatch();
    up->uploadStaticBuffer(psi.data(), in.data());
    up->updateDynamicBuffer(ubo.data(), 0, sizeof(Params), &params);
    cb->beginComputePass(up);
    cb->setComputePipeline(pipe.data());
    cb->setShaderResources(srb.data());
    cb->dispatch(groups, 1, 1);
    QRhiReadbackResult rb;
    QRhiResourceUpdateBatch* down = rhi->nextResourceUpdateBatch();
    down->readBackBuffer(partials.data(), 0, part_bytes, &rb);
    cb->endComputePass(down);
    rhi->endOffscreenFrame();

    const float* p = reinterpret_cast<const float*>(rb.data.constData());
    double gpu_sum = 0.0;
    double gpu_peak = 0.0;
    for (int g = 0; g < groups; ++g) {
        gpu_sum += p[2 * g];
        gpu_peak = std::max(gpu_peak, static_cast<double>(p[2 * g + 1]));
    }
    const double sum_rel = std::abs(gpu_sum - cpu_sum) / cpu_sum;
    const double peak_rel = std::abs(gpu_peak - cpu_peak) / cpu_peak;
    const bool pass = sum_rel < 1e-5 && peak_rel < 1e-6;
    std::printf("norm/peak reduce (QRhi/Vulkan): rel err sum %.3e, peak %.3e  [%s]\n",
                sum_rel, peak_rel, pass ? "PASS" : "FAIL");
    return pass;
}

// <phi|psi> = sum conj(phi)*psi, exactly kInnerProductSrc -- a complex-valued
// two-input reduction. psi(0)/phi(3) readonly, partials(2) read-write, n(1).
bool check_inner_product(QRhi* rhi) {
    const std::size_t n = 20000;
    std::vector<ses::Complex<double>> psi_d(n);
    std::vector<ses::Complex<double>> phi_d(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        psi_d[i] = ses::Complex<double>{std::sin(0.019 * x) + 0.1, std::cos(0.023 * x)};
        phi_d[i] = ses::Complex<double>{std::cos(0.011 * x), std::sin(0.029 * x) - 0.2};
    }
    double cpu_re = 0.0;
    double cpu_im = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double ar = phi_d[i].real();
        const double ai = phi_d[i].imag();
        const double br = psi_d[i].real();
        const double bi = psi_d[i].imag();
        cpu_re += ar * br + ai * bi;   // conj(phi)*psi, real
        cpu_im += ar * bi - ai * br;   // conj(phi)*psi, imag
    }
    const std::vector<float> psi_f = to_rg32f(psi_d);
    const std::vector<float> phi_f = to_rg32f(phi_d);
    const quint32 bytes = static_cast<quint32>(psi_f.size() * sizeof(float));
    const int groups = 256;
    const quint32 part_bytes = static_cast<quint32>(2 * groups * sizeof(float));

    QShader cs = load_qsb(QStringLiteral(":/shaders/inner_product.comp.qsb"));
    if (!cs.isValid()) { std::fprintf(stderr, "inner_product.comp.qsb missing\n"); return false; }

    QScopedPointer<QRhiBuffer> psi(
        rhi->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, bytes));
    QScopedPointer<QRhiBuffer> phi(
        rhi->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, bytes));
    QScopedPointer<QRhiBuffer> partials(
        rhi->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, part_bytes));
    struct alignas(16) Params { quint32 n; quint32 pad0, pad1, pad2; };
    Params params{ static_cast<quint32>(n), 0, 0, 0 };
    QScopedPointer<QRhiBuffer> ubo(
        rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(Params)));
    if (!psi->create() || !phi->create() || !partials->create() || !ubo->create()) { return false; }

    QScopedPointer<QRhiShaderResourceBindings> srb(rhi->newShaderResourceBindings());
    srb->setBindings({
        QRhiShaderResourceBinding::bufferLoad(0, QRhiShaderResourceBinding::ComputeStage,
                                              psi.data()),
        QRhiShaderResourceBinding::uniformBuffer(1, QRhiShaderResourceBinding::ComputeStage,
                                                 ubo.data()),
        QRhiShaderResourceBinding::bufferLoadStore(2, QRhiShaderResourceBinding::ComputeStage,
                                                   partials.data()),
        QRhiShaderResourceBinding::bufferLoad(3, QRhiShaderResourceBinding::ComputeStage,
                                              phi.data()),
    });
    if (!srb->create()) { return false; }

    QScopedPointer<QRhiComputePipeline> pipe(rhi->newComputePipeline());
    pipe->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, cs));
    pipe->setShaderResourceBindings(srb.data());
    if (!pipe->create()) { return false; }

    QRhiCommandBuffer* cb = nullptr;
    if (rhi->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return false; }
    QRhiResourceUpdateBatch* up = rhi->nextResourceUpdateBatch();
    up->uploadStaticBuffer(psi.data(), psi_f.data());
    up->uploadStaticBuffer(phi.data(), phi_f.data());
    up->updateDynamicBuffer(ubo.data(), 0, sizeof(Params), &params);
    cb->beginComputePass(up);
    cb->setComputePipeline(pipe.data());
    cb->setShaderResources(srb.data());
    cb->dispatch(groups, 1, 1);
    QRhiReadbackResult rb;
    QRhiResourceUpdateBatch* down = rhi->nextResourceUpdateBatch();
    down->readBackBuffer(partials.data(), 0, part_bytes, &rb);
    cb->endComputePass(down);
    rhi->endOffscreenFrame();

    const float* p = reinterpret_cast<const float*>(rb.data.constData());
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
    std::printf("inner-product kernel (QRhi/Vulkan): rel err = %.3e  [%s]\n",
                rel, pass ? "PASS" : "FAIL");
    return pass;
}

// One dipole half-kick psi <- exp(-i theta axis.r) psi per grid cell, exactly
// kDipoleKickSrc. Exercises grid-coordinate math + a std140 UBO with vec4-padded
// vec3 geometry, against a CPU reference applying the same per-cell rotation.
bool check_dipole_kick(QRhi* rhi) {
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
        psi_d[idx] = ses::Complex<double>{std::sin(0.13 * x) + 0.2, std::cos(0.09 * x) - 0.1};
        const int i = static_cast<int>(idx) % nx;
        const int j = (static_cast<int>(idx) / nx) % ny;
        const int k = static_cast<int>(idx) / (nx * ny);
        const double rx = box_min[0] + i * cell_h[0];
        const double ry = box_min[1] + j * cell_h[1];
        const double rz = box_min[2] + k * cell_h[2];
        const double ang = -theta * (axis[0] * rx + axis[1] * ry + axis[2] * rz);
        const double wr = std::cos(ang);
        const double wi = std::sin(ang);
        const double ar = psi_d[idx].real();
        const double ai = psi_d[idx].imag();
        cpu[idx] = ses::Complex<double>{ar * wr - ai * wi, ar * wi + ai * wr};
    }
    const std::vector<float> in = to_rg32f(psi_d);
    const quint32 bytes = static_cast<quint32>(in.size() * sizeof(float));

    QShader cs = load_qsb(QStringLiteral(":/shaders/dipole_kick.comp.qsb"));
    if (!cs.isValid()) { std::fprintf(stderr, "dipole_kick.comp.qsb missing\n"); return false; }

    QScopedPointer<QRhiBuffer> psi(
        rhi->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, bytes));
    struct alignas(16) Params {
        quint32 n;
        qint32 nx;
        qint32 ny;
        float theta;
        float box_min[4];
        float cell_h[4];
        float axis[4];
    };
    Params params{};
    params.n = static_cast<quint32>(n);
    params.nx = nx;
    params.ny = ny;
    params.theta = static_cast<float>(theta);
    for (int c = 0; c < 3; ++c) {
        params.box_min[c] = static_cast<float>(box_min[c]);
        params.cell_h[c] = static_cast<float>(cell_h[c]);
        params.axis[c] = static_cast<float>(axis[c]);
    }
    QScopedPointer<QRhiBuffer> ubo(
        rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(Params)));
    if (!psi->create() || !ubo->create()) { return false; }

    QScopedPointer<QRhiShaderResourceBindings> srb(rhi->newShaderResourceBindings());
    srb->setBindings({
        QRhiShaderResourceBinding::bufferLoadStore(0, QRhiShaderResourceBinding::ComputeStage,
                                                   psi.data()),
        QRhiShaderResourceBinding::uniformBuffer(1, QRhiShaderResourceBinding::ComputeStage,
                                                 ubo.data()),
    });
    if (!srb->create()) { return false; }

    QScopedPointer<QRhiComputePipeline> pipe(rhi->newComputePipeline());
    pipe->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, cs));
    pipe->setShaderResourceBindings(srb.data());
    if (!pipe->create()) { return false; }

    QRhiCommandBuffer* cb = nullptr;
    if (rhi->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return false; }
    QRhiResourceUpdateBatch* up = rhi->nextResourceUpdateBatch();
    up->uploadStaticBuffer(psi.data(), in.data());
    up->updateDynamicBuffer(ubo.data(), 0, sizeof(Params), &params);
    cb->beginComputePass(up);
    cb->setComputePipeline(pipe.data());
    cb->setShaderResources(srb.data());
    cb->dispatch(static_cast<int>((n + 255) / 256), 1, 1);
    QRhiReadbackResult rb;
    QRhiResourceUpdateBatch* down = rhi->nextResourceUpdateBatch();
    down->readBackBuffer(psi.data(), 0, bytes, &rb);
    cb->endComputePass(down);
    rhi->endOffscreenFrame();

    const float* out = reinterpret_cast<const float*>(rb.data.constData());
    double max_err = 0.0;
    for (std::size_t idx = 0; idx < n; ++idx) {
        max_err = std::max(max_err, std::abs(out[2 * idx] - cpu[idx].real()));
        max_err = std::max(max_err, std::abs(out[2 * idx + 1] - cpu[idx].imag()));
    }
    const bool pass = max_err < 1e-4;
    std::printf("dipole-kick kernel (QRhi/Vulkan): max |gpu - cpu| = %.3e  [%s]\n",
                max_err, pass ? "PASS" : "FAIL");
    return pass;
}

// <grad V> = sum |psi|^2 grad V, exactly kMeanForceSrc -- a vec4 reduction that
// folds psi (0) against a per-cell grad V field (4, vec4) into vec4 partials
// (2). Verifies the field-input reduction pattern on QRhi.
bool check_mean_force(QRhi* rhi) {
    const std::size_t n = 20000;
    std::vector<ses::Complex<double>> psi_d(n);
    std::vector<float> grad_f(4 * n);
    double cpu[3] = {0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        psi_d[i] = ses::Complex<double>{0.4 * std::sin(0.011 * x), 0.3 * std::cos(0.013 * x) + 0.05};
        const double gx = std::sin(0.007 * x);
        const double gy = std::cos(0.005 * x) - 0.3;
        const double gz = 0.2 * std::sin(0.017 * x);
        grad_f[4 * i + 0] = static_cast<float>(gx);
        grad_f[4 * i + 1] = static_cast<float>(gy);
        grad_f[4 * i + 2] = static_cast<float>(gz);
        grad_f[4 * i + 3] = 0.0f;
        const double d = psi_d[i].real() * psi_d[i].real() + psi_d[i].imag() * psi_d[i].imag();
        cpu[0] += d * gx;
        cpu[1] += d * gy;
        cpu[2] += d * gz;
    }
    const std::vector<float> psi_f = to_rg32f(psi_d);
    const quint32 psi_bytes = static_cast<quint32>(psi_f.size() * sizeof(float));
    const quint32 grad_bytes = static_cast<quint32>(grad_f.size() * sizeof(float));
    const int groups = 256;
    const quint32 part_bytes = static_cast<quint32>(4 * groups * sizeof(float));

    QShader cs = load_qsb(QStringLiteral(":/shaders/mean_force.comp.qsb"));
    if (!cs.isValid()) { std::fprintf(stderr, "mean_force.comp.qsb missing\n"); return false; }

    QScopedPointer<QRhiBuffer> psi(
        rhi->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, psi_bytes));
    QScopedPointer<QRhiBuffer> grad(
        rhi->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, grad_bytes));
    QScopedPointer<QRhiBuffer> partials(
        rhi->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, part_bytes));
    struct alignas(16) Params { quint32 n; quint32 pad0, pad1, pad2; };
    Params params{ static_cast<quint32>(n), 0, 0, 0 };
    QScopedPointer<QRhiBuffer> ubo(
        rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(Params)));
    if (!psi->create() || !grad->create() || !partials->create() || !ubo->create()) { return false; }

    QScopedPointer<QRhiShaderResourceBindings> srb(rhi->newShaderResourceBindings());
    srb->setBindings({
        QRhiShaderResourceBinding::bufferLoad(0, QRhiShaderResourceBinding::ComputeStage,
                                              psi.data()),
        QRhiShaderResourceBinding::uniformBuffer(1, QRhiShaderResourceBinding::ComputeStage,
                                                 ubo.data()),
        QRhiShaderResourceBinding::bufferLoadStore(2, QRhiShaderResourceBinding::ComputeStage,
                                                   partials.data()),
        QRhiShaderResourceBinding::bufferLoad(4, QRhiShaderResourceBinding::ComputeStage,
                                              grad.data()),
    });
    if (!srb->create()) { return false; }

    QScopedPointer<QRhiComputePipeline> pipe(rhi->newComputePipeline());
    pipe->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, cs));
    pipe->setShaderResourceBindings(srb.data());
    if (!pipe->create()) { return false; }

    QRhiCommandBuffer* cb = nullptr;
    if (rhi->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return false; }
    QRhiResourceUpdateBatch* up = rhi->nextResourceUpdateBatch();
    up->uploadStaticBuffer(psi.data(), psi_f.data());
    up->uploadStaticBuffer(grad.data(), grad_f.data());
    up->updateDynamicBuffer(ubo.data(), 0, sizeof(Params), &params);
    cb->beginComputePass(up);
    cb->setComputePipeline(pipe.data());
    cb->setShaderResources(srb.data());
    cb->dispatch(groups, 1, 1);
    QRhiReadbackResult rb;
    QRhiResourceUpdateBatch* down = rhi->nextResourceUpdateBatch();
    down->readBackBuffer(partials.data(), 0, part_bytes, &rb);
    cb->endComputePass(down);
    rhi->endOffscreenFrame();

    const float* p = reinterpret_cast<const float*>(rb.data.constData());
    double gpu[3] = {0.0, 0.0, 0.0};
    for (int g = 0; g < groups; ++g) {
        gpu[0] += p[4 * g + 0];
        gpu[1] += p[4 * g + 1];
        gpu[2] += p[4 * g + 2];
    }
    const double mag = std::sqrt(cpu[0] * cpu[0] + cpu[1] * cpu[1] + cpu[2] * cpu[2]);
    const double err = std::sqrt((gpu[0] - cpu[0]) * (gpu[0] - cpu[0]) +
                                 (gpu[1] - cpu[1]) * (gpu[1] - cpu[1]) +
                                 (gpu[2] - cpu[2]) * (gpu[2] - cpu[2]));
    const double rel = (mag > 0.0) ? err / mag : err;
    const bool pass = rel < 1e-4;
    std::printf("mean-force <grad V> (QRhi/Vulkan): rel err = %.3e  [%s]\n",
                rel, pass ? "PASS" : "FAIL");
    return pass;
}

// <to| r |from> = sum conj(to)*from * (x,y,z), exactly kDipoleSrc -- three
// complex reductions with grid coordinates, written as 6 floats/workgroup.
// to(0)/from(3) readonly, partials(2) read-write, vec4-padded geometry UBO(1).
bool check_dipole(QRhi* rhi) {
    const int nx = 8;
    const int ny = 8;
    const int nz = 8;
    const std::size_t n = static_cast<std::size_t>(nx * ny * nz);
    const double box_min[3] = {-4.0, -4.0, -4.0};
    const double cell_h[3] = {1.0, 1.1, 0.9};

    std::vector<ses::Complex<double>> to_d(n);
    std::vector<ses::Complex<double>> from_d(n);
    double cpu[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // Dx re,im, Dy re,im, Dz re,im
    for (std::size_t idx = 0; idx < n; ++idx) {
        const double x = static_cast<double>(idx);
        to_d[idx] = ses::Complex<double>{std::sin(0.11 * x) + 0.1, std::cos(0.07 * x)};
        from_d[idx] = ses::Complex<double>{std::cos(0.05 * x), std::sin(0.13 * x) - 0.2};
        const int i = static_cast<int>(idx) % nx;
        const int j = (static_cast<int>(idx) / nx) % ny;
        const int k = static_cast<int>(idx) / (nx * ny);
        const double r[3] = {box_min[0] + i * cell_h[0], box_min[1] + j * cell_h[1],
                             box_min[2] + k * cell_h[2]};
        const double ar = to_d[idx].real();
        const double ai = to_d[idx].imag();
        const double br = from_d[idx].real();
        const double bi = from_d[idx].imag();
        const double c_re = ar * br + ai * bi;  // conj(to)*from
        const double c_im = ar * bi - ai * br;
        for (int a = 0; a < 3; ++a) {
            cpu[2 * a + 0] += c_re * r[a];
            cpu[2 * a + 1] += c_im * r[a];
        }
    }
    const std::vector<float> to_f = to_rg32f(to_d);
    const std::vector<float> from_f = to_rg32f(from_d);
    const quint32 bytes = static_cast<quint32>(to_f.size() * sizeof(float));
    const int groups = 256;
    const quint32 part_bytes = static_cast<quint32>(6 * groups * sizeof(float));

    QShader cs = load_qsb(QStringLiteral(":/shaders/dipole.comp.qsb"));
    if (!cs.isValid()) { std::fprintf(stderr, "dipole.comp.qsb missing\n"); return false; }

    QScopedPointer<QRhiBuffer> fto(
        rhi->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, bytes));
    QScopedPointer<QRhiBuffer> ffrom(
        rhi->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, bytes));
    QScopedPointer<QRhiBuffer> partials(
        rhi->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, part_bytes));
    struct alignas(16) Params {
        quint32 n;
        qint32 nx;
        qint32 ny;
        qint32 pad0;
        float box_min[4];
        float cell_h[4];
    };
    Params params{};
    params.n = static_cast<quint32>(n);
    params.nx = nx;
    params.ny = ny;
    for (int c = 0; c < 3; ++c) {
        params.box_min[c] = static_cast<float>(box_min[c]);
        params.cell_h[c] = static_cast<float>(cell_h[c]);
    }
    QScopedPointer<QRhiBuffer> ubo(
        rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(Params)));
    if (!fto->create() || !ffrom->create() || !partials->create() || !ubo->create()) { return false; }

    QScopedPointer<QRhiShaderResourceBindings> srb(rhi->newShaderResourceBindings());
    srb->setBindings({
        QRhiShaderResourceBinding::bufferLoad(0, QRhiShaderResourceBinding::ComputeStage,
                                              fto.data()),
        QRhiShaderResourceBinding::uniformBuffer(1, QRhiShaderResourceBinding::ComputeStage,
                                                 ubo.data()),
        QRhiShaderResourceBinding::bufferLoadStore(2, QRhiShaderResourceBinding::ComputeStage,
                                                   partials.data()),
        QRhiShaderResourceBinding::bufferLoad(3, QRhiShaderResourceBinding::ComputeStage,
                                              ffrom.data()),
    });
    if (!srb->create()) { return false; }

    QScopedPointer<QRhiComputePipeline> pipe(rhi->newComputePipeline());
    pipe->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, cs));
    pipe->setShaderResourceBindings(srb.data());
    if (!pipe->create()) { return false; }

    QRhiCommandBuffer* cb = nullptr;
    if (rhi->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return false; }
    QRhiResourceUpdateBatch* up = rhi->nextResourceUpdateBatch();
    up->uploadStaticBuffer(fto.data(), to_f.data());
    up->uploadStaticBuffer(ffrom.data(), from_f.data());
    up->updateDynamicBuffer(ubo.data(), 0, sizeof(Params), &params);
    cb->beginComputePass(up);
    cb->setComputePipeline(pipe.data());
    cb->setShaderResources(srb.data());
    cb->dispatch(groups, 1, 1);
    QRhiReadbackResult rb;
    QRhiResourceUpdateBatch* down = rhi->nextResourceUpdateBatch();
    down->readBackBuffer(partials.data(), 0, part_bytes, &rb);
    cb->endComputePass(down);
    rhi->endOffscreenFrame();

    const float* p = reinterpret_cast<const float*>(rb.data.constData());
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
    std::printf("dipole <to|r|from> (QRhi/Vulkan): rel err = %.3e  [%s]\n",
                rel, pass ? "PASS" : "FAIL");
    return pass;
}

// Radix-2 shared-memory line FFT at N=64, exactly kLineFftTemplate: a forward
// unnormalized DFT of one contiguous line (n_lines=1, stride=1, base=0),
// compared against ses::fft. The FIRST FFT kernel on the Vulkan backend --
// bit-reversed load, log2(N)=6 barrier-separated butterfly stages.
bool check_fft(QRhi* rhi) {
    const int N = 64;
    std::vector<ses::Complex<double>> line(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        line[static_cast<std::size_t>(i)] =
            ses::Complex<double>{std::sin(0.3 * i) + 0.1 * i, std::cos(0.7 * i) - 0.2};
    }
    std::vector<ses::Complex<double>> cpu = line;
    ses::fft(cpu);  // forward, unnormalized -- same convention as the kernel

    const std::vector<float> in = to_rg32f(line);
    const quint32 bytes = static_cast<quint32>(in.size() * sizeof(float));

    QShader cs = load_qsb(QStringLiteral(":/shaders/fft_line64.comp.qsb"));
    if (!cs.isValid()) { std::fprintf(stderr, "fft_line64.comp.qsb missing\n"); return false; }

    QScopedPointer<QRhiBuffer> data(
        rhi->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, bytes));
    struct alignas(16) Params {
        qint32 mod_a, mul_b, mul_c, stride, n_lines, pad0, pad1, pad2;
    };
    Params params{};
    params.mod_a = N;      // l=0 -> base = (0 % N)*1 + (0 / N)*0 = 0
    params.mul_b = 1;
    params.mul_c = 0;
    params.stride = 1;
    params.n_lines = 1;
    QScopedPointer<QRhiBuffer> ubo(
        rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(Params)));
    if (!data->create() || !ubo->create()) { return false; }

    QScopedPointer<QRhiShaderResourceBindings> srb(rhi->newShaderResourceBindings());
    srb->setBindings({
        QRhiShaderResourceBinding::bufferLoadStore(0, QRhiShaderResourceBinding::ComputeStage,
                                                   data.data()),
        QRhiShaderResourceBinding::uniformBuffer(1, QRhiShaderResourceBinding::ComputeStage,
                                                 ubo.data()),
    });
    if (!srb->create()) { return false; }

    QScopedPointer<QRhiComputePipeline> pipe(rhi->newComputePipeline());
    pipe->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, cs));
    pipe->setShaderResourceBindings(srb.data());
    if (!pipe->create()) { return false; }

    QRhiCommandBuffer* cb = nullptr;
    if (rhi->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return false; }
    QRhiResourceUpdateBatch* up = rhi->nextResourceUpdateBatch();
    up->uploadStaticBuffer(data.data(), in.data());
    up->updateDynamicBuffer(ubo.data(), 0, sizeof(Params), &params);
    cb->beginComputePass(up);
    cb->setComputePipeline(pipe.data());
    cb->setShaderResources(srb.data());
    cb->dispatch(1, 1, 1);  // n_lines = 1 workgroup
    QRhiReadbackResult rb;
    QRhiResourceUpdateBatch* down = rhi->nextResourceUpdateBatch();
    down->readBackBuffer(data.data(), 0, bytes, &rb);
    cb->endComputePass(down);
    rhi->endOffscreenFrame();

    const float* out = reinterpret_cast<const float*>(rb.data.constData());
    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        max_err = std::max(max_err, std::abs(out[2 * i] - cpu[static_cast<std::size_t>(i)].real()));
        max_err = std::max(max_err,
                           std::abs(out[2 * i + 1] - cpu[static_cast<std::size_t>(i)].imag()));
    }
    const bool pass = max_err < 1e-3;
    std::printf("line FFT N=64 (QRhi/Vulkan): max |gpu - cpu| = %.3e  [%s]\n",
                max_err, pass ? "PASS" : "FAIL");
    return pass;
}

// fp16 storage codec roundtrip: pack fp32 -> half (packHalf2x16), then unpack
// half -> fp32 (unpackHalf2x16), compare to the original. Two dispatches in one
// compute pass (QRhi orders the read-after-write on the half buffer). Validates
// kPackHalfSrc + kUnpackHalfSrc; error is fp16 precision (~5e-4), not fp32.
bool check_fp16_roundtrip(QRhi* rhi) {
    const std::size_t n = 4096;
    std::vector<ses::Complex<double>> src_d(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        src_d[i] = ses::Complex<double>{0.5 * std::sin(0.05 * x), 0.5 * std::cos(0.03 * x)};
    }
    const std::vector<float> src_f = to_rg32f(src_d);
    const quint32 fp32_bytes = static_cast<quint32>(src_f.size() * sizeof(float));
    const quint32 half_bytes = static_cast<quint32>(n * sizeof(quint32));

    QShader packcs = load_qsb(QStringLiteral(":/shaders/pack_half.comp.qsb"));
    QShader unpackcs = load_qsb(QStringLiteral(":/shaders/unpack_half.comp.qsb"));
    if (!packcs.isValid() || !unpackcs.isValid()) {
        std::fprintf(stderr, "pack/unpack_half.comp.qsb missing\n");
        return false;
    }

    QScopedPointer<QRhiBuffer> src(
        rhi->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, fp32_bytes));
    QScopedPointer<QRhiBuffer> half(
        rhi->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, half_bytes));
    QScopedPointer<QRhiBuffer> dst(
        rhi->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, fp32_bytes));
    struct alignas(16) Params { quint32 n; quint32 pad0, pad1, pad2; };
    Params params{ static_cast<quint32>(n), 0, 0, 0 };
    QScopedPointer<QRhiBuffer> ubo(
        rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(Params)));
    if (!src->create() || !half->create() || !dst->create() || !ubo->create()) { return false; }

    QScopedPointer<QRhiShaderResourceBindings> pack_srb(rhi->newShaderResourceBindings());
    pack_srb->setBindings({
        QRhiShaderResourceBinding::bufferLoad(0, QRhiShaderResourceBinding::ComputeStage,
                                              src.data()),
        QRhiShaderResourceBinding::uniformBuffer(1, QRhiShaderResourceBinding::ComputeStage,
                                                 ubo.data()),
        QRhiShaderResourceBinding::bufferLoadStore(6, QRhiShaderResourceBinding::ComputeStage,
                                                   half.data()),
    });
    if (!pack_srb->create()) { return false; }
    QScopedPointer<QRhiShaderResourceBindings> unpack_srb(rhi->newShaderResourceBindings());
    unpack_srb->setBindings({
        QRhiShaderResourceBinding::bufferLoadStore(0, QRhiShaderResourceBinding::ComputeStage,
                                                   dst.data()),
        QRhiShaderResourceBinding::uniformBuffer(1, QRhiShaderResourceBinding::ComputeStage,
                                                 ubo.data()),
        QRhiShaderResourceBinding::bufferLoad(6, QRhiShaderResourceBinding::ComputeStage,
                                              half.data()),
    });
    if (!unpack_srb->create()) { return false; }

    QScopedPointer<QRhiComputePipeline> pack_pipe(rhi->newComputePipeline());
    pack_pipe->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, packcs));
    pack_pipe->setShaderResourceBindings(pack_srb.data());
    if (!pack_pipe->create()) { return false; }
    QScopedPointer<QRhiComputePipeline> unpack_pipe(rhi->newComputePipeline());
    unpack_pipe->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, unpackcs));
    unpack_pipe->setShaderResourceBindings(unpack_srb.data());
    if (!unpack_pipe->create()) { return false; }

    QRhiCommandBuffer* cb = nullptr;
    if (rhi->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return false; }
    QRhiResourceUpdateBatch* up = rhi->nextResourceUpdateBatch();
    up->uploadStaticBuffer(src.data(), src_f.data());
    up->updateDynamicBuffer(ubo.data(), 0, sizeof(Params), &params);
    const int groups = static_cast<int>((n + 255) / 256);
    cb->beginComputePass(up);
    cb->setComputePipeline(pack_pipe.data());
    cb->setShaderResources(pack_srb.data());
    cb->dispatch(groups, 1, 1);
    cb->setComputePipeline(unpack_pipe.data());
    cb->setShaderResources(unpack_srb.data());
    cb->dispatch(groups, 1, 1);
    QRhiReadbackResult rb;
    QRhiResourceUpdateBatch* down = rhi->nextResourceUpdateBatch();
    down->readBackBuffer(dst.data(), 0, fp32_bytes, &rb);
    cb->endComputePass(down);
    rhi->endOffscreenFrame();

    const float* out = reinterpret_cast<const float*>(rb.data.constData());
    double max_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        max_err = std::max(max_err, std::abs(out[2 * i] - src_d[i].real()));
        max_err = std::max(max_err, std::abs(out[2 * i + 1] - src_d[i].imag()));
    }
    const bool pass = max_err < 5e-3;  // fp16, not fp32
    std::printf("fp16 pack/unpack roundtrip (QRhi/Vulkan): max |gpu - cpu| = %.3e  [%s]\n",
                max_err, pass ? "PASS" : "FAIL");
    return pass;
}

// 3-D forward FFT of an 8x8x8 cube = the line FFT run once per axis (x, y, z),
// three dispatches in ONE compute pass with QRhi ordering the read-after-write
// on the shared data buffer between axes. This is the multi-axis orchestration
// at the heart of the engine's Strang step. Per-axis line-enumeration uniforms
// come from ses_gpu::axis_passes; compared against ses::fft(Field3D).
bool check_fft3(QRhi* rhi) {
    const int nx = 8;
    const int ny = 8;
    const int nz = 8;
    const ses::Grid1D ax{-4.0, 4.0, 8};
    const ses::Grid3D g{ax, ax, ax};
    ses::Field3D original{g};
    for (int i = 0; i < original.size(); ++i) {
        const double x = static_cast<double>(i);
        original.data()[static_cast<std::size_t>(i)] =
            ses::Complex<double>{std::sin(0.61 * x) + 0.15, std::cos(1.27 * x) - 0.2};
    }
    ses::Field3D cpu = original;
    ses::fft(cpu);  // 3-D forward, x/y/z line FFTs (same convention as the kernel)

    const std::vector<float> in = to_rg32f(original.data());
    const quint32 bytes = static_cast<quint32>(in.size() * sizeof(float));

    QShader cs = load_qsb(QStringLiteral(":/shaders/fft_line8.comp.qsb"));
    if (!cs.isValid()) { std::fprintf(stderr, "fft_line8.comp.qsb missing\n"); return false; }

    QScopedPointer<QRhiBuffer> data(
        rhi->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, bytes));
    if (!data->create()) { return false; }

    struct alignas(16) AxisParams {
        qint32 mod_a, mul_b, mul_c, stride, n_lines, p0, p1, p2;
    };
    // {mod_a, mul_b, mul_c, stride, n_lines} per axis (ses_gpu::axis_passes).
    const AxisParams axp[3] = {
        { ny * nz, nx, 0, 1, ny * nz, 0, 0, 0 },       // x-lines (contiguous)
        { nx, 1, nx * ny, nx, nx * nz, 0, 0, 0 },      // y-lines
        { nx * ny, 1, 0, nx * ny, nx * ny, 0, 0, 0 },  // z-lines
    };

    QScopedPointer<QRhiBuffer> ubo[3];
    QScopedPointer<QRhiShaderResourceBindings> srb[3];
    for (int a = 0; a < 3; ++a) {
        ubo[a].reset(
            rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, sizeof(AxisParams)));
        if (!ubo[a]->create()) { return false; }
        srb[a].reset(rhi->newShaderResourceBindings());
        srb[a]->setBindings({
            QRhiShaderResourceBinding::bufferLoadStore(0, QRhiShaderResourceBinding::ComputeStage,
                                                       data.data()),
            QRhiShaderResourceBinding::uniformBuffer(1, QRhiShaderResourceBinding::ComputeStage,
                                                     ubo[a].data()),
        });
        if (!srb[a]->create()) { return false; }
    }

    QScopedPointer<QRhiComputePipeline> pipe(rhi->newComputePipeline());
    pipe->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, cs));
    pipe->setShaderResourceBindings(srb[0].data());  // layout template (all 3 share it)
    if (!pipe->create()) { return false; }

    QRhiCommandBuffer* cb = nullptr;
    if (rhi->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return false; }
    QRhiResourceUpdateBatch* up = rhi->nextResourceUpdateBatch();
    up->uploadStaticBuffer(data.data(), in.data());
    for (int a = 0; a < 3; ++a) {
        up->updateDynamicBuffer(ubo[a].data(), 0, sizeof(AxisParams), &axp[a]);
    }
    cb->beginComputePass(up);
    cb->setComputePipeline(pipe.data());
    for (int a = 0; a < 3; ++a) {
        cb->setShaderResources(srb[a].data());
        cb->dispatch(axp[a].n_lines, 1, 1);
    }
    QRhiReadbackResult rb;
    QRhiResourceUpdateBatch* down = rhi->nextResourceUpdateBatch();
    down->readBackBuffer(data.data(), 0, bytes, &rb);
    cb->endComputePass(down);
    rhi->endOffscreenFrame();

    const float* out = reinterpret_cast<const float*>(rb.data.constData());
    double max_err = 0.0;
    double max_mag = 0.0;
    for (std::size_t i = 0; i < cpu.data().size(); ++i) {
        max_err = std::max(max_err, std::abs(out[2 * i] - cpu.data()[i].real()));
        max_err = std::max(max_err, std::abs(out[2 * i + 1] - cpu.data()[i].imag()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].real()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].imag()));
    }
    const double tol = 1e-3 + 1e-5 * max_mag;
    const bool pass = max_err < tol;
    std::printf("fft3 8x8x8 (QRhi/Vulkan): max |gpu - cpu| = %.3e (tol %.3e)  [%s]\n",
                max_err, tol, pass ? "PASS" : "FAIL");
    return pass;
}

// The production Strang step through the QrhiEngine (qrhi_engine.hpp), 20 steps
// on an 8x8x8 soft-Coulomb grid vs SplitOperator3D::step -- the QRhi analog of
// sesolver_gpucheck's G4. This is the whole engine chain (halfV, fft3, kinetic,
// IFFT, halfV) orchestrated on QRhi, not a single kernel.
bool check_engine_step(QRhi* rhi) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const double dt = 0.02;
    const ses::SplitOperator3D cpu_prop{g, v, dt};
    ses::Field3D psi0 = ses::gaussian_wavepacket(g, ses::Vec3d{1.0, 0.0, 0.0},
                                                 ses::Vec3d{1.2, 1.2, 1.2},
                                                 ses::Vec3d{0.0, 0.5, 0.0});

    ses_qrhi::QrhiEngine engine;
    if (!engine.initialize(rhi, g, cpu_prop.half_potential_phase(), cpu_prop.kinetic_phase(),
                           psi0.data())) {
        std::printf("engine 20 steps (QRhi/Vulkan): engine init FAIL\n");
        return false;
    }
    engine.step(20);
    std::vector<float> gpu_out;
    engine.readback(gpu_out);

    ses::Field3D cpu = psi0;
    cpu_prop.step(cpu, 20);

    double max_err = 0.0;
    double max_mag = 0.0;
    for (std::size_t i = 0; i < cpu.data().size(); ++i) {
        max_err = std::max(max_err, std::abs(gpu_out[2 * i] - cpu.data()[i].real()));
        max_err = std::max(max_err, std::abs(gpu_out[2 * i + 1] - cpu.data()[i].imag()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].real()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].imag()));
    }
    const double tol = 1e-4 + 1e-5 * max_mag;
    const bool pass = max_err < tol;
    std::printf("engine 20 steps (QRhi/Vulkan): max |gpu - cpu| = %.3e (tol %.3e)  [%s]\n",
                max_err, tol, pass ? "PASS" : "FAIL");
    return pass;
}

// Imaginary-time relaxation through the QrhiEngine, 50 steps on an 8x8x8
// harmonic grid vs ImaginaryTimePropagator3D::relax (the QRhi analog of
// gpucheck's G7 numeric match). Exercises the per-step renormalize path (norm/
// peak reduction -> host sum -> scale). The free energy estimator needs a finer
// grid to converge to 3w/2, so it is reported for info, not asserted here.
bool check_relax(QRhi* rhi) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::harmonic_potential(g, 1.0, ses::Vec3d{});
    const double dtau = 0.05;
    const ses::SplitOperator3D real_prop{g, v, 0.02};  // phase tables for init
    const ses::ImaginaryTimePropagator3D cpu_relaxer{g, v, dtau};
    ses::Field3D psi0 = ses::gaussian_wavepacket(g, ses::Vec3d{1.0, 0.0, 0.0},
                                                 ses::Vec3d{1.5, 1.5, 1.5}, ses::Vec3d{});

    ses_qrhi::QrhiEngine engine;
    if (!engine.initialize(rhi, g, real_prop.half_potential_phase(), real_prop.kinetic_phase(),
                           psi0.data())) {
        std::printf("relax 50 steps (QRhi/Vulkan): engine init FAIL\n");
        return false;
    }
    if (!engine.set_relax_tables(cpu_relaxer.half_potential_weight(),
                                 cpu_relaxer.kinetic_weight(), dtau, g.cell_volume())) {
        std::printf("relax 50 steps (QRhi/Vulkan): set_relax_tables FAIL\n");
        return false;
    }
    const ses_qrhi::QrhiEngine::RelaxStats stats = engine.relax_step(50);
    std::vector<float> gpu_out;
    engine.readback(gpu_out);

    ses::Field3D cpu = psi0;
    cpu_relaxer.relax(cpu, 50);

    double max_err = 0.0;
    double max_mag = 0.0;
    for (std::size_t i = 0; i < cpu.data().size(); ++i) {
        max_err = std::max(max_err, std::abs(gpu_out[2 * i] - cpu.data()[i].real()));
        max_err = std::max(max_err, std::abs(gpu_out[2 * i + 1] - cpu.data()[i].imag()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].real()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].imag()));
    }
    const double tol = 1e-4 + 1e-5 * max_mag;
    const bool pass = max_err < tol;
    std::printf("relax 50 steps (QRhi/Vulkan): max |gpu - cpu| = %.3e (tol %.3e), E~%.3f  [%s]\n",
                max_err, tol, stats.energy, pass ? "PASS" : "FAIL");
    return pass;
}

// T3 (QRhi): the driven Strang step (dipole half-kicks around the static tables)
// vs core/drive.hpp driven_step. The drive is the adversarial gpucheck one --
// skew (non-unit) polarization axis, nonzero omega, nonzero start time -- so the
// kick uniforms (axis/box_min/cell_h/theta) all get exercised. An 8x8x8 grid is
// used to match the baked fft_line8 (32^3 would need an unbaked fft_line32).
bool check_driven(QRhi* rhi) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const double dt = 0.02;
    const ses::SplitOperator3D cpu_prop{g, v, dt};
    const ses::DipoleDrive drive{ses::Vec3d{0.3, -0.2, 1.0}, 0.5, 0.6};
    ses::Field3D psi0 = ses::gaussian_wavepacket(g, ses::Vec3d{1.0, 0.0, 0.0},
                                                 ses::Vec3d{1.2, 1.2, 1.2},
                                                 ses::Vec3d{0.0, 0.5, 0.0});

    ses_qrhi::QrhiEngine engine;
    if (!engine.initialize(rhi, g, cpu_prop.half_potential_phase(), cpu_prop.kinetic_phase(),
                           psi0.data())) {
        std::printf("driven 20 steps (QRhi/Vulkan): engine init FAIL\n");
        return false;
    }
    engine.driven_step(drive, 1.3, dt, 20);
    std::vector<float> gpu_out;
    engine.readback(gpu_out);

    ses::Field3D cpu = psi0;
    ses::driven_step(cpu, cpu_prop, drive, 1.3, 20);

    double max_err = 0.0;
    double max_mag = 0.0;
    for (std::size_t i = 0; i < cpu.data().size(); ++i) {
        max_err = std::max(max_err, std::abs(gpu_out[2 * i] - cpu.data()[i].real()));
        max_err = std::max(max_err, std::abs(gpu_out[2 * i + 1] - cpu.data()[i].imag()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].real()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].imag()));
    }
    const double tol = 1e-4 + 1e-5 * max_mag;
    const bool pass = max_err < tol;
    std::printf("driven 20 steps (QRhi/Vulkan): max |gpu - cpu| = %.3e (tol %.3e)  [%s]\n",
                max_err, tol, pass ? "PASS" : "FAIL");
    return pass;
}

// T1 (QRhi): the deflated imaginary-time relax vs ImaginaryTimePropagator3D::
// relax_deflated -- Gram-Schmidt projecting the ground state out each step so
// the flow climbs to the first excited level. Exercises the inner-product
// reduction, the axpy projection, and copy_into_psi (the collapse path). An
// 8^3 grid matches the baked fft_line8; the excited energy estimate is reported
// for info (the 8^3 grid is too coarse to pin it to 5w/2, like check_relax).
bool check_deflation(QRhi* rhi) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::harmonic_potential(g, 1.0, ses::Vec3d{});
    const double dtau = 0.05;
    const ses::SplitOperator3D real_prop{g, v, 0.02};  // phase tables for init
    const ses::ImaginaryTimePropagator3D cpu_relaxer{g, v, dtau};

    // CPU ground state (the deflation target) + a mixed-parity guess.
    ses::Field3D ground = ses::gaussian_wavepacket(g, ses::Vec3d{},
                                                   ses::Vec3d{1.2, 1.2, 1.2}, ses::Vec3d{});
    cpu_relaxer.relax(ground, 600);
    ses::Field3D guess = ses::gaussian_wavepacket(g, ses::Vec3d{1.0, 0.5, 0.0},
                                                  ses::Vec3d{1.2, 1.2, 1.2}, ses::Vec3d{});

    ses_qrhi::QrhiEngine engine;
    if (!engine.initialize(rhi, g, real_prop.half_potential_phase(), real_prop.kinetic_phase(),
                           guess.data())) {
        std::printf("deflation (QRhi/Vulkan): engine init FAIL\n");
        return false;
    }
    if (!engine.set_relax_tables(cpu_relaxer.half_potential_weight(),
                                 cpu_relaxer.kinetic_weight(), dtau, g.cell_volume())) {
        std::printf("deflation (QRhi/Vulkan): set_relax_tables FAIL\n");
        return false;
    }
    const int ground_h = engine.create_state_buffer(ground.data());
    if (ground_h < 0) {
        std::printf("deflation (QRhi/Vulkan): create_state_buffer FAIL\n");
        return false;
    }

    // Inner-product kernel: <ground|guess> vs the CPU double reference.
    const ses::Complex<double> cpu_ip = ses::inner_product(ground, guess);
    const ses::Complex<double> gpu_ip = engine.inner_with_psi(ground_h);
    const double ip_err = std::max(std::abs(gpu_ip.real() - cpu_ip.real()),
                                   std::abs(gpu_ip.imag() - cpu_ip.imag()));
    const bool ip_ok = ip_err < 1e-6;
    std::printf("  inner-product <phi|psi> (QRhi/Vulkan): max err %.3e  [%s]\n", ip_err,
                ip_ok ? "PASS" : "FAIL");

    // Deflated relax: 50 GPU steps vs 50 CPU steps (numeric parity).
    engine.relax_deflated_step({ground_h}, 50);
    std::vector<float> gpu_out;
    engine.readback(gpu_out);
    ses::Field3D cpu = guess;
    cpu_relaxer.relax_deflated(cpu, {&ground}, 50);
    double max_err = 0.0;
    double max_mag = 0.0;
    for (std::size_t i = 0; i < cpu.data().size(); ++i) {
        max_err = std::max(max_err, std::abs(gpu_out[2 * i] - cpu.data()[i].real()));
        max_err = std::max(max_err, std::abs(gpu_out[2 * i + 1] - cpu.data()[i].imag()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].real()));
        max_mag = std::max(max_mag, std::abs(cpu.data()[i].imag()));
    }
    const double tol = 1e-4 + 1e-5 * max_mag;
    const bool relax_ok = max_err < tol;
    std::printf("  deflated relax 50 steps (QRhi/Vulkan): max |gpu - cpu| = %.3e (tol %.3e)  [%s]\n",
                max_err, tol, relax_ok ? "PASS" : "FAIL");

    // Excited energy estimator (reported; 8^3 is too coarse to assert 5w/2).
    const ses_qrhi::QrhiEngine::RelaxStats stats = engine.relax_deflated_step({ground_h}, 550);
    std::printf("  deflated energy estimator (QRhi/Vulkan): E = %.4f (5w/2 = 2.5, coarse grid)\n",
                stats.energy);

    // copy_into_psi (quantum-jump collapse): bitwise copy of the aux buffer.
    engine.copy_into_psi(ground_h);
    std::vector<float> copied;
    engine.readback(copied);
    const std::vector<float> ground_staged = ses_qrhi::to_rg32f(ground.data());
    double copy_err = 0.0;
    for (std::size_t i = 0; i < copied.size(); ++i) {
        copy_err = std::max(copy_err,
                            static_cast<double>(std::abs(copied[i] - ground_staged[i])));
    }
    const bool copy_ok = copy_err == 0.0;
    std::printf("  copy_into_psi (QRhi/Vulkan): max err %.3e  [%s]\n", copy_err,
                copy_ok ? "PASS" : "FAIL");

    const bool pass = ip_ok && relax_ok && copy_ok;
    std::printf("deflation (QRhi/Vulkan): [%s]\n", pass ? "PASS" : "FAIL");
    return pass;
}

// Magnetic minimal coupling (proper solve): the exact three-shear rotate_z on
// the QRhi psi buffer vs core ses::rotate_z, and the full magnetic Strang step
// (paramagnetic rotation + diamagnetic potential) vs MagneticPropagator3D for a
// field along z and along x. 8^3 grid (baked fft_line8); the shear needs single-
// axis FFTs and a per-axis inverse-FFT scale (1/n), exercised here.
bool check_magnetic(QRhi* rhi) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const double dt = 0.02;
    const double b = 0.5;
    ses::Field3D psi0 = ses::gaussian_wavepacket(
        g, ses::Vec3d{1.0, 0.0, 0.5}, ses::Vec3d{1.4, 1.4, 1.4}, ses::Vec3d{0.0, 0.4, 0.0});

    const ses::SplitOperator3D base{g, v, dt};
    ses_qrhi::QrhiEngine engine;
    if (!engine.initialize(rhi, g, base.half_potential_phase(), base.kinetic_phase(),
                           psi0.data())) {
        std::printf("magnetic (QRhi/Vulkan): engine init FAIL\n");
        return false;
    }

    // Exact three-shear rotation vs core rotate_z.
    engine.rotate_z_shear(0.6);
    std::vector<float> gpu_out;
    engine.readback(gpu_out);
    ses::Field3D cpu = psi0;
    ses::rotate_z(cpu, 0.6);
    double rot_err = 0.0;
    double rot_mag = 0.0;
    for (std::size_t i = 0; i < cpu.data().size(); ++i) {
        rot_err = std::max(rot_err, std::abs(gpu_out[2 * i] - cpu.data()[i].real()));
        rot_err = std::max(rot_err, std::abs(gpu_out[2 * i + 1] - cpu.data()[i].imag()));
        rot_mag = std::max(rot_mag, std::abs(cpu.data()[i].real()));
        rot_mag = std::max(rot_mag, std::abs(cpu.data()[i].imag()));
    }
    const double rot_tol = 1e-3 + 1e-4 * rot_mag;
    const bool rot_ok = rot_err < rot_tol;
    std::printf("  rotate_z (three-shear, QRhi/Vulkan): max |gpu - cpu| = %.3e (tol %.3e)  [%s]\n",
                rot_err, rot_tol, rot_ok ? "PASS" : "FAIL");

    // Full magnetic step vs MagneticPropagator3D, field along z then along x.
    bool step_ok = true;
    for (int fa = 2; fa >= 0; fa -= 2) {  // axis z then x
        const ses::MagneticPropagator3D mprop{g, v, dt, b, fa};
        const ses::SplitOperator3D core_diamag{g, mprop.effective_potential(), dt};
        engine.set_half_potential(core_diamag.half_potential_phase());
        engine.upload_state(psi0.data());
        engine.magnetic_step(fa, 0.5 * b * (0.5 * dt), 20);
        engine.readback(gpu_out);
        ses::Field3D cpu2 = psi0;
        mprop.step(cpu2, 20);
        double err = 0.0;
        double mag = 0.0;
        for (std::size_t i = 0; i < cpu2.data().size(); ++i) {
            err = std::max(err, std::abs(gpu_out[2 * i] - cpu2.data()[i].real()));
            err = std::max(err, std::abs(gpu_out[2 * i + 1] - cpu2.data()[i].imag()));
            mag = std::max(mag, std::abs(cpu2.data()[i].real()));
            mag = std::max(mag, std::abs(cpu2.data()[i].imag()));
        }
        const double tol = 2e-3 + 1e-4 * mag;
        const bool ok = err < tol;
        std::printf("  magnetic step %c (QRhi/Vulkan): max |gpu - cpu| = %.3e (tol %.3e)  [%s]\n",
                    fa == 2 ? 'z' : 'x', err, tol, ok ? "PASS" : "FAIL");
        step_ok = ok && step_ok;
    }

    const bool pass = rot_ok && step_ok;
    std::printf("magnetic (QRhi/Vulkan): [%s]\n", pass ? "PASS" : "FAIL");
    return pass;
}

// Orbital synthesis: psi = (u(|r|)/|r|) Y_lm built on the GPU from a radial
// table, vs core synthesize_orbital, across l = 0..5 and several m. Exercises
// the real-spherical-harmonic evaluation + radial interpolation + in-place
// normalization. Same grid on both sides, so this is a kernel-parity check.
bool check_synth(QRhi* rhi) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const ses::SplitOperator3D prop{g, v, 0.02};
    ses::Field3D seed = ses::gaussian_wavepacket(g, ses::Vec3d{}, ses::Vec3d{1.5, 1.5, 1.5},
                                                 ses::Vec3d{});
    ses_qrhi::QrhiEngine engine;
    if (!engine.initialize(rhi, g, prop.half_potential_phase(), prop.kinetic_phase(),
                           seed.data())) {
        std::printf("orbital synthesis (QRhi/Vulkan): engine init FAIL\n");
        return false;
    }

    const ses::RadialGrid rg{8.0, 1599};
    std::vector<double> vr(static_cast<std::size_t>(rg.n));
    for (int i = 0; i < rg.n; ++i) {
        vr[static_cast<std::size_t>(i)] = 0.5 * rg.r(i) * rg.r(i);
    }
    struct SynCase { int l, k, m; };
    const SynCase cases[] = {{0, 0, 0}, {1, 0, -1}, {2, 0, 1},
                             {3, 0, -2}, {4, 0, 3}, {5, 0, 5}, {5, 0, 0}};
    double worst = 0.0;
    for (const SynCase& c : cases) {
        const ses::RadialState st =
            ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, c.l), c.k);
        const ses::Field3D cpu = ses::synthesize_orbital(g, rg, st.u, c.l, c.m);
        if (!engine.synthesize_into_psi(st.u, c.l, c.m, rg.h(), rg.rmax, rg.n)) {
            std::printf("orbital synthesis (QRhi/Vulkan): synthesize FAIL\n");
            return false;
        }
        std::vector<float> gpu;
        engine.readback(gpu);
        for (std::size_t i = 0; i < cpu.data().size(); ++i) {
            worst = std::max(worst, std::abs(static_cast<double>(gpu[2 * i]) -
                                             cpu.data()[i].real()));
            worst = std::max(worst, std::abs(static_cast<double>(gpu[2 * i + 1]) -
                                             cpu.data()[i].imag()));
        }
    }
    const bool ok = worst < 1e-4;
    std::printf("orbital synthesis (u/r)Ylm (QRhi/Vulkan): max |gpu - cpu| = %.3e  [%s]\n",
                worst, ok ? "PASS" : "FAIL");
    return ok;
}

// SSBO -> 3D volume texture bridge (the renderer feed for M3): copy psi into an
// RGBA32F volume via imageStore, then read the texture back through an SSBO
// (imageLoad). A store->load round-trip that reproduces psi bit-for-bit proves
// imageStore wrote exactly the SSBO contents (the GL check's "pure copy: must
// be bitwise"). Exercises QRhi 3D storage-image handling, new for M3.
bool check_bridge(QRhi* rhi) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const ses::SplitOperator3D prop{g, v, 0.02};
    ses::Field3D psi0 = ses::gaussian_wavepacket(g, ses::Vec3d{2.0, 0.0, 0.0},
                                                 ses::Vec3d{1.2, 1.2, 1.2},
                                                 ses::Vec3d{0.0, 0.5, 0.0});
    ses_qrhi::QrhiEngine engine;
    if (!engine.initialize(rhi, g, prop.half_potential_phase(), prop.kinetic_phase(),
                           psi0.data())) {
        std::printf("texture bridge (QRhi/Vulkan): engine init FAIL\n");
        return false;
    }
    engine.step(20);  // a non-trivial psi to copy

    std::vector<float> ssbo;
    engine.readback(ssbo);
    std::vector<float> tex;
    if (!engine.bridge_roundtrip(tex)) {
        std::printf("texture bridge (QRhi/Vulkan): bridge FAIL\n");
        return false;
    }
    double err = 0.0;
    for (std::size_t i = 0; i < ssbo.size(); ++i) {
        err = std::max(err, static_cast<double>(std::abs(tex[i] - ssbo[i])));
    }
    const bool ok = err == 0.0;  // pure copy: must be bitwise
    std::printf("texture bridge (QRhi/Vulkan): max |tex - ssbo| = %.3e  [%s]\n", err,
                ok ? "PASS" : "FAIL");
    return ok;
}

// fp16 atlas consumers: an eigenstate stored fp32 and fp16 must give the SAME
// inner product and dipole matrix elements (to fp16 precision ~1e-3), proving
// the fp32 consumers read an fp16 state (unpacked on demand) correctly -- the
// small-VRAM storage fallback. Also checks a mixed fp32/fp16 dipole.
bool check_fp16_consumers(QRhi* rhi) {
    const ses::Grid1D axis{-4.0, 4.0, 8};
    const ses::Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const ses::SplitOperator3D prop{g, v, 0.02};
    ses::Field3D seed = ses::gaussian_wavepacket(g, ses::Vec3d{}, ses::Vec3d{1.5, 1.5, 1.5},
                                                 ses::Vec3d{});
    ses_qrhi::QrhiEngine engine;
    if (!engine.initialize(rhi, g, prop.half_potential_phase(), prop.kinetic_phase(),
                           seed.data())) {
        std::printf("fp16 consumers (QRhi/Vulkan): engine init FAIL\n");
        return false;
    }
    const ses::RadialGrid rg{8.0, 1599};
    std::vector<double> vr(static_cast<std::size_t>(rg.n));
    for (int i = 0; i < rg.n; ++i) {
        vr[static_cast<std::size_t>(i)] = 0.5 * rg.r(i) * rg.r(i);
    }
    const ses::RadialState st =
        ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, 1), 0);
    const int s32 = engine.synthesize_state(st.u, 1, 0, rg.h(), rg.rmax, rg.n);
    const int s16 = engine.synthesize_state_half(st.u, 1, 0, rg.h(), rg.rmax, rg.n);
    if (s32 < 0 || s16 < 0) {
        std::printf("fp16 consumers (QRhi/Vulkan): synthesize FAIL\n");
        return false;
    }

    ses::Field3D testpsi = ses::gaussian_wavepacket(g, ses::Vec3d{1.0, 0.5, -0.4},
                                                    ses::Vec3d{1.7, 1.7, 1.7}, ses::Vec3d{});
    engine.upload_state(testpsi.data());

    const ses::Complex<double> ip32 = engine.inner_state_with_psi(s32);
    const ses::Complex<double> ip16 = engine.inner_state_with_psi(s16);
    const double inner_err = std::max(std::abs(ip32.real() - ip16.real()),
                                      std::abs(ip32.imag() - ip16.imag()));

    const ses::DipoleMatrixElement d32 = engine.dipole_between(s32, s32);
    const ses::DipoleMatrixElement d16 = engine.dipole_between(s16, s16);
    const ses::DipoleMatrixElement dmix = engine.dipole_between(s32, s16);
    auto dip_err = [](const ses::DipoleMatrixElement& a, const ses::DipoleMatrixElement& b) {
        return std::max({std::abs(a.x.real() - b.x.real()), std::abs(a.y.real() - b.y.real()),
                         std::abs(a.z.real() - b.z.real()), std::abs(a.x.imag() - b.x.imag()),
                         std::abs(a.y.imag() - b.y.imag()), std::abs(a.z.imag() - b.z.imag())});
    };
    const double d16_err = dip_err(d32, d16);
    const double dmix_err = dip_err(d32, dmix);

    const double worst = std::max({inner_err, d16_err, dmix_err});
    const bool ok = worst < 3e-3;
    std::printf("fp16 consumers (inner/dipole/mixed vs fp32, QRhi/Vulkan): max = %.3e  [%s]\n",
                worst, ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    // Unbuffered stdout: when the harness output is piped, printf is block-
    // buffered, so a crash mid-run would discard every earlier check's line.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    QGuiApplication app(argc, argv);

    QVulkanInstance inst;
    inst.setExtensions(QRhiVulkanInitParams::preferredInstanceExtensions());
    if (!inst.create()) {
        std::printf("SKIP: QVulkanInstance::create failed (no Vulkan runtime, err %d)\n",
                    inst.errorCode());
        return 77;
    }

    QRhiVulkanInitParams params;
    params.inst = &inst;
    QScopedPointer<QRhi> rhi(QRhi::create(QRhi::Vulkan, &params));
    if (!rhi) {
        std::printf("SKIP: QRhi::create(Vulkan) failed (no Vulkan device)\n");
        return 77;
    }
    if (!rhi->isFeatureSupported(QRhi::Compute)) {
        std::printf("SKIP: QRhi Vulkan backend reports no Compute support\n");
        return 77;
    }
    std::printf("QRhi Vulkan: device='%s'\n", rhi->driverInfo().deviceName.constData());

    bool ok = true;
    ok = check_phase_multiply(rhi.data()) && ok;
    ok = check_scale(rhi.data()) && ok;
    ok = check_norm_peak(rhi.data()) && ok;
    ok = check_inner_product(rhi.data()) && ok;
    ok = check_dipole_kick(rhi.data()) && ok;
    ok = check_mean_force(rhi.data()) && ok;
    ok = check_dipole(rhi.data()) && ok;
    ok = check_fft(rhi.data()) && ok;
    ok = check_fp16_roundtrip(rhi.data()) && ok;
    ok = check_fft3(rhi.data()) && ok;
    ok = check_engine_step(rhi.data()) && ok;
    ok = check_relax(rhi.data()) && ok;
    ok = check_driven(rhi.data()) && ok;
    ok = check_deflation(rhi.data()) && ok;
    ok = check_magnetic(rhi.data()) && ok;
    ok = check_synth(rhi.data()) && ok;
    ok = check_bridge(rhi.data()) && ok;
    ok = check_fp16_consumers(rhi.data()) && ok;
    std::printf("%s\n", ok ? "QRhi kernel checks PASS" : "QRhi kernel checks FAILED");
    return ok ? 0 : 1;
}
