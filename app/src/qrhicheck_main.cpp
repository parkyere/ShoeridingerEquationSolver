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

#include <core/complex.hpp>
#include <core/fft.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>

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

}  // namespace

int main(int argc, char** argv) {
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
    std::printf("%s\n", ok ? "QRhi kernel checks PASS" : "QRhi kernel checks FAILED");
    return ok ? 0 : 1;
}
