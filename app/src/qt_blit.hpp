#pragma once

// Qt's ENTIRE render layer: import the ses_vk scene image (createFrom,
// non-owning) and draw one fullscreen triangle sampling it. The scene itself
// renders in ses_vk; when the offscreen target is recreated (resize), the
// import and the SRB are rebuilt against the new image.

// volk (inside vk_device.hpp) must own the vulkan.h inclusion before any Qt
// header pulls its own Vulkan integration.
#include "vk_device.hpp"

#include <QColor>
#include <QFile>
#include <QScopedPointer>
#include <QSize>
#include <QString>

#include <rhi/qrhi.h>

#include <cstdint>
#include <cstdio>

namespace ses_shell {

class BlitPresenter {
public:
    // Build the one persistent resource (the sampler); the SRB and pipeline
    // are built lazily in present() once a scene image exists to seed them.
    bool init(QRhi* rhi) {
        rhi_ = rhi;
        sampler_.reset(rhi_->newSampler(
            QRhiSampler::Nearest, QRhiSampler::Nearest, QRhiSampler::None,
            QRhiSampler::ClampToEdge, QRhiSampler::ClampToEdge,
            QRhiSampler::ClampToEdge));
        return sampler_->create();
    }

    void release() {
        pipe_.reset();
        srb_.reset();
        scene_wrap_.reset();
        wrapped_img_ = VK_NULL_HANDLE;
        sampler_.reset();
        rhi_ = nullptr;
    }

    // The whole Qt draw: clear-only until the scene image exists, otherwise
    // import-on-change + one 3-vertex pass.
    void present(QRhiCommandBuffer* cb, QRhiRenderTarget* rt, VkImage scene_img,
                 std::uint32_t scene_w, std::uint32_t scene_h) {
        if (scene_img == VK_NULL_HANDLE) {
            // Nothing rendered yet: clear only.
            cb->beginPass(rt, QColor(10, 13, 23), {1.0f, 0}, nullptr);
            cb->endPass();
            return;
        }
        if (scene_img != wrapped_img_) {
            scene_wrap_.reset(rhi_->newTexture(
                QRhiTexture::RGBA8, QSize(static_cast<int>(scene_w),
                                          static_cast<int>(scene_h))));
            QRhiTexture::NativeTexture src;
            src.object = quint64(reinterpret_cast<std::uintptr_t>(scene_img));
            src.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            if (!scene_wrap_->createFrom(src)) {
                std::fprintf(stderr, "blit: scene image import failed\n");
                scene_wrap_.reset();
                wrapped_img_ = VK_NULL_HANDLE;
                return;
            }
            wrapped_img_ = scene_img;
            srb_.reset(rhi_->newShaderResourceBindings());
            srb_->setBindings({QRhiShaderResourceBinding::sampledTexture(
                0, QRhiShaderResourceBinding::FragmentStage, scene_wrap_.data(),
                sampler_.data())});
            if (!srb_->create()) {
                std::fprintf(stderr, "blit: SRB create failed\n");
                srb_.reset();
                return;
            }
            if (pipe_.isNull()) {
                auto load_qsb = [](const char* path) {
                    QFile f{QLatin1String(path)};
                    if (!f.open(QIODevice::ReadOnly)) {
                        return QShader();
                    }
                    return QShader::fromSerialized(f.readAll());
                };
                pipe_.reset(rhi_->newGraphicsPipeline());
                pipe_->setShaderStages({
                    {QRhiShaderStage::Vertex,
                     load_qsb(":/shaders/blit.vert.qsb")},
                    {QRhiShaderStage::Fragment,
                     load_qsb(":/shaders/blit.frag.qsb")},
                });
                pipe_->setVertexInputLayout({});
                pipe_->setShaderResourceBindings(srb_.data());
                pipe_->setRenderPassDescriptor(rt->renderPassDescriptor());
                pipe_->setTopology(QRhiGraphicsPipeline::Triangles);
                pipe_->setDepthTest(false);
                pipe_->setDepthWrite(false);
                if (!pipe_->create()) {
                    std::fprintf(stderr, "blit: pipeline create failed\n");
                    pipe_.reset();
                    return;
                }
            }
        }
        if (pipe_.isNull() || srb_.isNull()) {
            return;
        }
        cb->beginPass(rt, QColor(0, 0, 0), {1.0f, 0}, nullptr);
        cb->setGraphicsPipeline(pipe_.data());
        const QSize px = rt->pixelSize();
        cb->setViewport(QRhiViewport(0, 0, static_cast<float>(px.width()),
                                     static_cast<float>(px.height())));
        cb->setShaderResources(srb_.data());
        cb->draw(3);
        cb->endPass();
    }

private:
    QRhi* rhi_ = nullptr;
    QScopedPointer<QRhiTexture> scene_wrap_;  // createFrom import (non-owning)
    VkImage wrapped_img_ = VK_NULL_HANDLE;
    QScopedPointer<QRhiSampler> sampler_;
    QScopedPointer<QRhiShaderResourceBindings> srb_;
    QScopedPointer<QRhiGraphicsPipeline> pipe_;
};

}  // namespace ses_shell
