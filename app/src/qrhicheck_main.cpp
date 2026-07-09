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

    const bool ok = check_phase_multiply(rhi.data());
    std::printf("%s\n", ok ? "QRhi kernel checks PASS" : "QRhi kernel checks FAILED");
    return ok ? 0 : 1;
}
