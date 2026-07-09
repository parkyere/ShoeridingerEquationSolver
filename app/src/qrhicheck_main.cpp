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
    std::printf("%s\n", ok ? "QRhi kernel checks PASS" : "QRhi kernel checks FAILED");
    return ok ? 0 : 1;
}
