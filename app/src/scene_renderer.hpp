#pragma once

// The QRhi scene renderer (Stage 4 extraction from main.cpp's Viewport): mesh
// (isosurface + proton + axes gizmo + billboarded z label) and volume (the
// front-to-back |psi|^2 raymarch) graphics pipelines, their resources, and the
// per-frame record. Pure rendering -- no simulation state, no engine calls.
// The shell computes per-frame inputs (camera, view mode, brightness, staged
// uploads) and hands them in as FrameInput; initialization failures return
// false with a stderr note and the shell decides how fatal that is.

#include <core/camera.hpp>
#include <core/colormap.hpp>
#include <core/grid.hpp>
#include <core/marching_cubes.hpp>
#include <core/sphere.hpp>
#include <core/vec.hpp>

#include <rhi/qrhi.h>

#include <QFile>
#include <QMatrix4x4>
#include <QString>
#include <QVarLengthArray>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace ses_shell {

constexpr int kPhaseLutSize = 256;
constexpr double kProtonMarkerRadius = 0.35;  // symbolic (a real proton is ~1e-5 Bohr)

class SceneRenderer {
public:
    // Per-frame inputs computed by the shell. Non-null mesh/volume_staging
    // pointers request a (re)upload staged into this frame's batch.
    struct FrameInput {
        bool cloud = true;  // Cloud (volume) vs Surface (isosurface) view
        double azimuth = 0.0;
        double elevation = 0.0;
        double distance = 0.0;
        double peak = 0.0;        // |psi|^2 brightness normalizer
        double absorbance = 0.0;  // Beer-Lambert opacity
        float flash = 0.0f;       // photon-flash background weight, 0..1
        QRhiTexture* psi_volume = nullptr;  // engine bridge texture; null -> CPU fallback
        const ses::Mesh* mesh = nullptr;                   // non-null: upload isosurface
        const std::vector<ses::Rgb>* mesh_colors = nullptr;
        const std::vector<float>* volume_staging = nullptr;  // non-null: upload RG staging
    };

    // Build every render resource: pipelines, samplers, UBOs, static vertex
    // buffers, and the phase LUT (uploads enqueue on `u`, committed by the
    // caller's frame). `rp` is the render target's pass descriptor.
    bool initialize(QRhi* rhi, QRhiRenderPassDescriptor* rp, const ses::Grid3D& grid,
                    QRhiResourceUpdateBatch* u) {
        rhi_ = rhi;
        grid_ = grid;

        // -- shared samplers + per-pass uniform buffers --
        sampler_3d_.reset(rhi_->newSampler(QRhiSampler::Linear, QRhiSampler::Linear,
                                           QRhiSampler::None, QRhiSampler::ClampToEdge,
                                           QRhiSampler::ClampToEdge, QRhiSampler::ClampToEdge));
        sampler_1d_.reset(rhi_->newSampler(QRhiSampler::Linear, QRhiSampler::Linear,
                                           QRhiSampler::None, QRhiSampler::Repeat,
                                           QRhiSampler::Repeat, QRhiSampler::Repeat));
        scene_ubuf_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                          sizeof(MeshUbo)));
        gizmo_ubuf_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                          sizeof(MeshUbo)));
        volume_ubuf_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                           sizeof(VolumeUbo)));
        if (!sampler_3d_->create() || !sampler_1d_->create() || !scene_ubuf_->create() ||
            !gizmo_ubuf_->create() || !volume_ubuf_->create()) {
            std::fprintf(stderr, "SceneRenderer: sampler/UBO create failed\n");
            return false;
        }

        if (!create_mesh_resources(rp)) { return false; }
        if (!upload_proton_marker(u) || !upload_axes_gizmo(u) || !upload_box_cube(u)) {
            return false;
        }
        if (!create_textures(u)) { return false; }
        if (!create_volume_pipeline(rp)) { return false; }
        return true;
    }

    // Record the resource updates + the one render pass of a frame.
    void render(QRhiCommandBuffer* cb, QRhiRenderTarget* rt, const FrameInput& in) {
        // Photon flash: a brief warm background right after a quantum jump.
        const float w = in.flash;
        const QColor clear_color =
            QColor::fromRgbF(0.04f + 0.55f * w, 0.05f + 0.42f * w, 0.09f + 0.18f * w, 1.0f);

        const QSize out_px = rt->pixelSize();
        const double aspect =
            static_cast<double>(out_px.width()) / std::max(1, out_px.height());
        // The far plane must always enclose the whole box BEHIND the target, or
        // the volume's back-face proxy geometry (we cull front faces) gets
        // clipped and leaves dark triangular holes when zoomed out. Track the
        // camera distance plus the box's center-to-corner reach -- the farthest
        // any box point can sit from the eye at any orientation.
        const double bx = std::max(std::abs(grid_.x.xmin), std::abs(grid_.x.xmax));
        const double by = std::max(std::abs(grid_.y.xmin), std::abs(grid_.y.xmax));
        const double bz = std::max(std::abs(grid_.z.xmin), std::abs(grid_.z.xmax));
        const double box_reach = std::sqrt(bx * bx + by * by + bz * bz);
        const double zfar = in.distance + box_reach + 1.0;
        const ses::Mat4 proj =
            ses::perspective(45.0 * 3.14159265358979323846 / 180.0, aspect, 0.1, zfar);
        const ses::Vec3d eye =
            ses::orbit_eye(in.azimuth, in.elevation, in.distance, ses::Vec3d{});
        const ses::Mat4 view = ses::look_at(eye, ses::Vec3d{}, ses::Vec3d{0.0, 1.0, 0.0});
        const ses::Mat4 mvp = proj * view;

        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();

        // Scene camera UBO (mesh pass): clip-space corrected for the backend.
        MeshUbo scene_u{};
        store_corrected_mvp(mvp, scene_u.mvp);
        scene_u.eye[0] = static_cast<float>(eye.x);
        scene_u.eye[1] = static_cast<float>(eye.y);
        scene_u.eye[2] = static_cast<float>(eye.z);
        scene_u.eye[3] = 0.0f;
        u->updateDynamicBuffer(scene_ubuf_.data(), 0, sizeof(MeshUbo), &scene_u);

        // Gizmo camera: the same orbit orientation at a FIXED distance so the
        // corner overlay keeps a constant screen size.
        const double d_g = 3.2;  // frames the length-1 arrows in the corner box
        const ses::Vec3d geye = ses::orbit_eye(in.azimuth, in.elevation, d_g, ses::Vec3d{});
        const ses::Mat4 gproj =
            ses::perspective(45.0 * 3.14159265358979323846 / 180.0, 1.0, 0.1, 50.0);
        const ses::Mat4 gview = ses::look_at(geye, ses::Vec3d{}, ses::Vec3d{0.0, 1.0, 0.0});
        const ses::Mat4 gmvp = gproj * gview;
        MeshUbo gizmo_u{};
        store_corrected_mvp(gmvp, gizmo_u.mvp);
        gizmo_u.eye[0] = static_cast<float>(geye.x);
        gizmo_u.eye[1] = static_cast<float>(geye.y);
        gizmo_u.eye[2] = static_cast<float>(geye.z);
        gizmo_u.eye[3] = 0.0f;
        u->updateDynamicBuffer(gizmo_ubuf_.data(), 0, sizeof(MeshUbo), &gizmo_u);

        // Billboarded "z" glyph verts (rebuilt per frame, dynamic buffer).
        const std::vector<float> zdata = build_z_label_verts(gview, geye);
        u->updateDynamicBuffer(zlabel_vbuf_.data(), 0,
                               static_cast<quint32>(zdata.size() * sizeof(float)),
                               zdata.data());

        // Volume UBO (Cloud pass).
        VolumeUbo vol_u{};
        store_corrected_mvp(mvp, vol_u.mvp);
        vol_u.eye[0] = static_cast<float>(eye.x);
        vol_u.eye[1] = static_cast<float>(eye.y);
        vol_u.eye[2] = static_cast<float>(eye.z);
        vol_u.box_min[0] = static_cast<float>(grid_.x.xmin);
        vol_u.box_min[1] = static_cast<float>(grid_.y.xmin);
        vol_u.box_min[2] = static_cast<float>(grid_.z.xmin);
        vol_u.box_max[0] = static_cast<float>(grid_.x.xmax);
        vol_u.box_max[1] = static_cast<float>(grid_.y.xmax);
        vol_u.box_max[2] = static_cast<float>(grid_.z.xmax);
        // The nucleus lives INSIDE the ray marcher (analytic sphere): fogged by
        // cloud in front, occluding cloud behind.
        vol_u.proton_center[0] = 0.0f;
        vol_u.proton_center[1] = 0.0f;
        vol_u.proton_center[2] = 0.0f;
        vol_u.proton_color[0] = 1.0f;
        vol_u.proton_color[1] = 0.45f;
        vol_u.proton_color[2] = 0.2f;
        vol_u.inv_peak = static_cast<float>(in.peak > 0.0 ? 1.0 / in.peak : 0.0);
        vol_u.absorbance = static_cast<float>(in.absorbance);
        vol_u.proton_radius = static_cast<float>(kProtonMarkerRadius);
        u->updateDynamicBuffer(volume_ubuf_.data(), 0, sizeof(VolumeUbo), &vol_u);

        // Staged uploads requested by the shell for the active view.
        if (in.volume_staging != nullptr) {
            upload_volume(u, *in.volume_staging);
        }
        if (in.mesh != nullptr && in.mesh_colors != nullptr) {
            upload_mesh(u, *in.mesh, *in.mesh_colors);
        }
        if (in.cloud) {
            ensure_volume_srb(in.psi_volume);
        }

        const QRhiDepthStencilClearValue ds_clear{1.0f, 0};
        cb->beginPass(rt, clear_color, ds_clear, u);
        const QRhiViewport full_vp(0.0f, 0.0f, static_cast<float>(out_px.width()),
                                   static_cast<float>(out_px.height()));
        const QRhiScissor full_sc(0, 0, out_px.width(), out_px.height());

        if (in.cloud) {
            if (!volume_srb_.isNull()) {
                cb->setGraphicsPipeline(volume_pipe_.data());
                cb->setViewport(full_vp);
                cb->setShaderResources(volume_srb_.data());
                const QRhiCommandBuffer::VertexInput cube_in(cube_vbuf_.data(), 0);
                cb->setVertexInput(0, 1, &cube_in);
                cb->draw(36);
            }
        } else {
            // Surface mode keeps the mesh proton (depth-tested against the
            // opaque isosurface, so occlusion is already correct there).
            cb->setGraphicsPipeline(mesh_pipe_.data());
            cb->setViewport(full_vp);
            cb->setScissor(full_sc);
            cb->setShaderResources(scene_srb_.data());
            const QRhiCommandBuffer::VertexInput proton_in(proton_vbuf_.data(), 0);
            cb->setVertexInput(0, 1, &proton_in);
            cb->draw(static_cast<quint32>(proton_vertex_count_));
            if (!mesh_vbuf_.isNull() && vertex_count_ > 0) {
                const QRhiCommandBuffer::VertexInput mesh_in(mesh_vbuf_.data(), 0);
                cb->setVertexInput(0, 1, &mesh_in);
                cb->draw(static_cast<quint32>(vertex_count_));
            }
        }

        // XYZ orientation gizmo in the bottom-left corner, over both views.
        // GL cleared a scissored depth region; mid-pass clears do not exist in
        // QRhi, so the corner viewport maps to depth range [0, 0.01] instead:
        // always in front of the scene, self-occlusion within the gizmo intact.
        if (gizmo_vertex_count_ > 0) {
            const int side =
                std::clamp(std::min(out_px.width(), out_px.height()) / 6, 96, 180);
            const int margin = side / 8;
            cb->setGraphicsPipeline(mesh_pipe_.data());
            cb->setViewport(QRhiViewport(static_cast<float>(margin),
                                         static_cast<float>(margin),
                                         static_cast<float>(side),
                                         static_cast<float>(side), 0.0f, 0.01f));
            cb->setScissor(QRhiScissor(margin, margin, side, side));
            cb->setShaderResources(gizmo_srb_.data());
            const QRhiCommandBuffer::VertexInput gizmo_in(gizmo_vbuf_.data(), 0);
            cb->setVertexInput(0, 1, &gizmo_in);
            cb->draw(static_cast<quint32>(gizmo_vertex_count_));
            // Label only Z; X, Y follow by the right-hand rule.
            const QRhiCommandBuffer::VertexInput z_in(zlabel_vbuf_.data(), 0);
            cb->setVertexInput(0, 1, &z_in);
            cb->draw(18);
        }

        cb->endPass();
    }

private:
    // std140-compatible per-pass uniform blocks (match mesh/volume .vert/.frag).
    struct MeshUbo {
        float mvp[16];
        float eye[4];
    };
    struct VolumeUbo {
        float mvp[16];
        float eye[4];
        float box_min[4];
        float box_max[4];
        float proton_center[4];
        float proton_color[4];
        float inv_peak = 0.0f;
        float absorbance = 0.0f;
        float proton_radius = 0.0f;
        float pad0 = 0.0f;
    };

    // Load a baked .qsb from the resource system (baked into the executable).
    static QShader load_render_shader(const char* path) {
        QFile f{QLatin1String(path)};  // braces: dodge the vexing parse
        if (!f.open(QIODevice::ReadOnly)) {
            std::fprintf(stderr, "SceneRenderer: cannot open %s\n", path);
            return {};
        }
        return QShader::fromSerialized(f.readAll());
    }

    // mvp (ses column-major doubles) -> backend-corrected floats. QRhi's
    // clipSpaceCorrMatrix maps the GL clip conventions the ses camera produces
    // to the backend's (Vulkan: inverted Y, 0..1 depth).
    void store_corrected_mvp(const ses::Mat4& mvp, float* out16) const {
        QMatrix4x4 m;
        float* d = m.data();  // column-major storage, matches ses::Mat4::m
        for (int i = 0; i < 16; ++i) {
            d[i] = static_cast<float>(mvp.m[i]);
        }
        const QMatrix4x4 corrected = rhi_->clipSpaceCorrMatrix() * m;
        std::memcpy(out16, corrected.constData(), 16 * sizeof(float));
    }

    // A small billboarded "z" glyph just past the +Z arrow tip, always facing
    // the gizmo camera: 18 verts in the mesh vertex format, rebuilt per frame.
    static std::vector<float> build_z_label_verts(const ses::Mat4& view,
                                                  const ses::Vec3d& eye) {
        const ses::Vec3d right{view.m[0], view.m[4], view.m[8]};
        const ses::Vec3d up{view.m[1], view.m[5], view.m[9]};
        const ses::Vec3d center{0.0, 0.0, 1.18};  // just past the length-1 tip
        const double s = 0.24;
        const ses::Vec3d nrm = normalized(eye - center);  // faces the camera

        // "z" as three strokes in [-0.4,0.4] x [-0.5,0.5]: top bar, diagonal
        // (top-right -> bottom-left), bottom bar. Each a rectangle (2 tris).
        std::vector<std::array<double, 2>> pts;
        auto quad = [&](std::array<double, 2> a, std::array<double, 2> b,
                        std::array<double, 2> c, std::array<double, 2> d) {
            pts.push_back(a);
            pts.push_back(b);
            pts.push_back(c);
            pts.push_back(a);
            pts.push_back(c);
            pts.push_back(d);
        };
        quad({-0.4, 0.5}, {0.4, 0.5}, {0.4, 0.34}, {-0.4, 0.34});      // top bar
        quad({-0.4, -0.34}, {0.4, -0.34}, {0.4, -0.5}, {-0.4, -0.5});  // bottom
        const double ax = 0.4, ay = 0.40, bx = -0.4, by = -0.40, ht = 0.11;
        const double dx = bx - ax, dy = by - ay;
        const double dl = std::sqrt(dx * dx + dy * dy);
        const double px = -dy / dl * ht, py = dx / dl * ht;  // perpendicular
        quad({ax + px, ay + py}, {ax - px, ay - py}, {bx - px, by - py},
             {bx + px, by + py});  // diagonal bar

        std::vector<float> verts;
        verts.reserve(pts.size() * 9);
        for (const std::array<double, 2>& p : pts) {
            const ses::Vec3d w = center + (p[0] * s) * right + (p[1] * s) * up;
            verts.push_back(static_cast<float>(w.x));
            verts.push_back(static_cast<float>(w.y));
            verts.push_back(static_cast<float>(w.z));
            verts.push_back(static_cast<float>(nrm.x));
            verts.push_back(static_cast<float>(nrm.y));
            verts.push_back(static_cast<float>(nrm.z));
            verts.push_back(0.75f);
            verts.push_back(0.85f);
            verts.push_back(1.0f);
        }
        return verts;
    }

    // (Re)build the volume SRB against the live psi volume texture: the
    // engine's bridge texture on the GPU path, the CPU fallback otherwise.
    // Lazy because the engine texture exists only after compute init + the
    // first write; a null texture skips the volume draw.
    void ensure_volume_srb(QRhiTexture* engine_tex) {
        QRhiTexture* tex = engine_tex;
        if (tex == nullptr && !fallback_tex_.isNull()) {
            tex = fallback_tex_.data();
        }
        if (tex == nullptr) {
            volume_srb_.reset();
            volume_tex_bound_ = nullptr;
            return;
        }
        if (!volume_srb_.isNull() && volume_tex_bound_ == tex) {
            return;
        }
        using B = QRhiShaderResourceBinding;
        const auto stages = B::VertexStage | B::FragmentStage;
        volume_srb_.reset(rhi_->newShaderResourceBindings());
        volume_srb_->setBindings({
            B::uniformBuffer(0, stages, volume_ubuf_.data()),
            B::sampledTexture(1, B::FragmentStage, tex, sampler_3d_.data()),
            B::sampledTexture(2, B::FragmentStage, phase_tex_.data(), sampler_1d_.data()),
        });
        if (!volume_srb_->create()) {
            std::fprintf(stderr, "SceneRenderer: volume SRB create failed\n");
            volume_srb_.reset();
            volume_tex_bound_ = nullptr;
            return;
        }
        volume_tex_bound_ = tex;
    }

    // The mesh graphics pipeline (isosurface + proton + gizmo + z label):
    // interleaved pos3/normal3/color3 vertices, depth-tested and -written, no
    // blend, no cull (the headlight shading uses abs(dot)); UsesScissor so the
    // gizmo can clip to its corner (scene draws set a full-rect scissor).
    bool create_mesh_resources(QRhiRenderPassDescriptor* rp) {
        using B = QRhiShaderResourceBinding;
        const auto stages = B::VertexStage | B::FragmentStage;
        scene_srb_.reset(rhi_->newShaderResourceBindings());
        scene_srb_->setBindings({ B::uniformBuffer(0, stages, scene_ubuf_.data()) });
        gizmo_srb_.reset(rhi_->newShaderResourceBindings());
        gizmo_srb_->setBindings({ B::uniformBuffer(0, stages, gizmo_ubuf_.data()) });
        if (!scene_srb_->create() || !gizmo_srb_->create()) {
            std::fprintf(stderr, "SceneRenderer: mesh SRB create failed\n");
            return false;
        }

        // Dynamic vertex buffer for the billboarded "z" glyph (18 verts,
        // rebuilt per frame).
        zlabel_vbuf_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::VertexBuffer,
                                           18 * 9 * sizeof(float)));
        if (!zlabel_vbuf_->create()) {
            std::fprintf(stderr, "SceneRenderer: z label buffer create failed\n");
            return false;
        }

        mesh_pipe_.reset(rhi_->newGraphicsPipeline());
        mesh_pipe_->setShaderStages({
            { QRhiShaderStage::Vertex, load_render_shader(":/shaders/mesh.vert.qsb") },
            { QRhiShaderStage::Fragment, load_render_shader(":/shaders/mesh.frag.qsb") },
        });
        QRhiVertexInputLayout mesh_layout;
        mesh_layout.setBindings({ QRhiVertexInputBinding(9 * sizeof(float)) });
        mesh_layout.setAttributes({
            QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float3, 0),
            QRhiVertexInputAttribute(0, 1, QRhiVertexInputAttribute::Float3,
                                     3 * sizeof(float)),
            QRhiVertexInputAttribute(0, 2, QRhiVertexInputAttribute::Float3,
                                     6 * sizeof(float)),
        });
        mesh_pipe_->setVertexInputLayout(mesh_layout);
        mesh_pipe_->setShaderResourceBindings(scene_srb_.data());
        mesh_pipe_->setRenderPassDescriptor(rp);
        mesh_pipe_->setTopology(QRhiGraphicsPipeline::Triangles);
        mesh_pipe_->setCullMode(QRhiGraphicsPipeline::None);
        mesh_pipe_->setDepthTest(true);
        mesh_pipe_->setDepthWrite(true);
        mesh_pipe_->setDepthOp(QRhiGraphicsPipeline::Less);
        mesh_pipe_->setFlags(QRhiGraphicsPipeline::UsesScissor);
        if (!mesh_pipe_->create()) {
            std::fprintf(stderr, "SceneRenderer: mesh pipeline create failed\n");
            return false;
        }
        return true;
    }

    // The volume graphics pipeline: the box proxy rasterizes its BACK faces
    // (cull front -- works with the eye inside the box too), no depth,
    // premultiplied-alpha blend over the clear color.
    bool create_volume_pipeline(QRhiRenderPassDescriptor* rp) {
        volume_pipe_.reset(rhi_->newGraphicsPipeline());
        volume_pipe_->setShaderStages({
            { QRhiShaderStage::Vertex, load_render_shader(":/shaders/volume.vert.qsb") },
            { QRhiShaderStage::Fragment, load_render_shader(":/shaders/volume.frag.qsb") },
        });
        QRhiVertexInputLayout cube_layout;
        cube_layout.setBindings({ QRhiVertexInputBinding(3 * sizeof(float)) });
        cube_layout.setAttributes({
            QRhiVertexInputAttribute(0, 0, QRhiVertexInputAttribute::Float3, 0),
        });
        volume_pipe_->setVertexInputLayout(cube_layout);
        // Layout template only: the live SRB (ensure_volume_srb) swaps the psi
        // texture between the engine bridge and the CPU fallback. The phase LUT
        // stands in for the (layout-compatible) 3D slot at pipeline build.
        using B = QRhiShaderResourceBinding;
        const auto stages = B::VertexStage | B::FragmentStage;
        volume_layout_srb_.reset(rhi_->newShaderResourceBindings());
        volume_layout_srb_->setBindings({
            B::uniformBuffer(0, stages, volume_ubuf_.data()),
            B::sampledTexture(1, B::FragmentStage, phase_tex_.data(), sampler_3d_.data()),
            B::sampledTexture(2, B::FragmentStage, phase_tex_.data(), sampler_1d_.data()),
        });
        if (!volume_layout_srb_->create()) {
            std::fprintf(stderr, "SceneRenderer: volume layout SRB create failed\n");
            return false;
        }
        volume_pipe_->setShaderResourceBindings(volume_layout_srb_.data());
        volume_pipe_->setRenderPassDescriptor(rp);
        volume_pipe_->setTopology(QRhiGraphicsPipeline::Triangles);
        volume_pipe_->setCullMode(QRhiGraphicsPipeline::Front);
        volume_pipe_->setDepthTest(false);
        volume_pipe_->setDepthWrite(false);
        QRhiGraphicsPipeline::TargetBlend blend;
        blend.enable = true;  // defaults are premultiplied One/OneMinusSrcAlpha
        volume_pipe_->setTargetBlends({ blend });
        if (!volume_pipe_->create()) {
            std::fprintf(stderr, "SceneRenderer: volume pipeline create failed\n");
            return false;
        }
        return true;
    }

    // Pack a ses::Mesh + per-vertex colors into the interleaved mesh format.
    static std::vector<float> interleave_mesh(const ses::Mesh& mesh,
                                              const float* solid_rgb,
                                              const std::vector<ses::Rgb>* colors) {
        std::vector<float> out;
        out.reserve(mesh.vertices.size() * 9);
        for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
            const ses::Vec3d& pt = mesh.vertices[i];
            const ses::Vec3d& n = mesh.normals[i];
            out.push_back(static_cast<float>(pt.x));
            out.push_back(static_cast<float>(pt.y));
            out.push_back(static_cast<float>(pt.z));
            out.push_back(static_cast<float>(n.x));
            out.push_back(static_cast<float>(n.y));
            out.push_back(static_cast<float>(n.z));
            if (colors != nullptr) {
                out.push_back(static_cast<float>((*colors)[i].r));
                out.push_back(static_cast<float>((*colors)[i].g));
                out.push_back(static_cast<float>((*colors)[i].b));
            } else {
                out.push_back(solid_rgb[0]);
                out.push_back(solid_rgb[1]);
                out.push_back(solid_rgb[2]);
            }
        }
        return out;
    }

    // XYZ orientation gizmo: three arrows from the origin -- X red, Y green,
    // Z blue -- baked into the mesh vertex format (per-vertex color). Uploaded
    // once; drawn each frame by the gizmo overlay in render().
    bool upload_axes_gizmo(QRhiResourceUpdateBatch* u) {
        struct Axis {
            ses::Vec3d dir;
            float r, g, b;
        };
        const Axis axes[3] = {
            {ses::Vec3d{1.0, 0.0, 0.0}, 0.95f, 0.25f, 0.25f},  // X red
            {ses::Vec3d{0.0, 1.0, 0.0}, 0.30f, 0.85f, 0.30f},  // Y green
            {ses::Vec3d{0.0, 0.0, 1.0}, 0.35f, 0.55f, 1.00f},  // Z blue
        };
        std::vector<float> interleaved;
        for (const Axis& ax : axes) {
            const ses::Mesh arrow = ses::arrow_mesh(ax.dir, 1.0, 0.045, 0.13, 0.32, 16);
            const float rgb[3] = {ax.r, ax.g, ax.b};
            const std::vector<float> part = interleave_mesh(arrow, rgb, nullptr);
            interleaved.insert(interleaved.end(), part.begin(), part.end());
        }
        gizmo_vertex_count_ = static_cast<int>(interleaved.size() / 9);
        gizmo_vbuf_.reset(rhi_->newBuffer(
            QRhiBuffer::Immutable, QRhiBuffer::VertexBuffer,
            static_cast<quint32>(interleaved.size() * sizeof(float))));
        if (!gizmo_vbuf_->create()) {
            std::fprintf(stderr, "SceneRenderer: gizmo buffer create failed\n");
            return false;
        }
        u->uploadStaticBuffer(gizmo_vbuf_.data(), interleaved.data());
        return true;
    }

    // Static warm-colored sphere at the nucleus, in the mesh vertex format.
    bool upload_proton_marker(QRhiResourceUpdateBatch* u) {
        const ses::Mesh sphere = ses::sphere_mesh(ses::Vec3d{}, kProtonMarkerRadius, 16, 24);
        const float rgb[3] = {1.0f, 0.45f, 0.20f};  // warm proton color
        const std::vector<float> interleaved = interleave_mesh(sphere, rgb, nullptr);
        proton_vertex_count_ = static_cast<int>(sphere.vertices.size());
        proton_vbuf_.reset(rhi_->newBuffer(
            QRhiBuffer::Immutable, QRhiBuffer::VertexBuffer,
            static_cast<quint32>(interleaved.size() * sizeof(float))));
        if (!proton_vbuf_->create()) {
            std::fprintf(stderr, "SceneRenderer: proton buffer create failed\n");
            return false;
        }
        u->uploadStaticBuffer(proton_vbuf_.data(), interleaved.data());
        return true;
    }

    bool upload_box_cube(QRhiResourceUpdateBatch* u) {
        const float lo[3] = {static_cast<float>(grid_.x.xmin),
                             static_cast<float>(grid_.y.xmin),
                             static_cast<float>(grid_.z.xmin)};
        const float hi[3] = {static_cast<float>(grid_.x.xmax),
                             static_cast<float>(grid_.y.xmax),
                             static_cast<float>(grid_.z.xmax)};
        // 12 triangles, CCW seen from OUTSIDE (verified per-face normals).
        const float v[] = {
            lo[0],lo[1],lo[2], lo[0],lo[1],hi[2], lo[0],hi[1],hi[2],
            lo[0],lo[1],lo[2], lo[0],hi[1],hi[2], lo[0],hi[1],lo[2],
            hi[0],lo[1],lo[2], hi[0],hi[1],lo[2], hi[0],hi[1],hi[2],
            hi[0],lo[1],lo[2], hi[0],hi[1],hi[2], hi[0],lo[1],hi[2],
            lo[0],lo[1],lo[2], hi[0],lo[1],lo[2], hi[0],lo[1],hi[2],
            lo[0],lo[1],lo[2], hi[0],lo[1],hi[2], lo[0],lo[1],hi[2],
            lo[0],hi[1],lo[2], lo[0],hi[1],hi[2], hi[0],hi[1],hi[2],
            lo[0],hi[1],lo[2], hi[0],hi[1],hi[2], hi[0],hi[1],lo[2],
            lo[0],lo[1],lo[2], lo[0],hi[1],lo[2], hi[0],hi[1],lo[2],
            lo[0],lo[1],lo[2], hi[0],hi[1],lo[2], hi[0],lo[1],lo[2],
            lo[0],lo[1],hi[2], hi[0],lo[1],hi[2], hi[0],hi[1],hi[2],
            lo[0],lo[1],hi[2], hi[0],hi[1],hi[2], lo[0],hi[1],hi[2],
        };
        cube_vbuf_.reset(rhi_->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::VertexBuffer,
                                         sizeof(v)));
        if (!cube_vbuf_->create()) {
            std::fprintf(stderr, "SceneRenderer: cube buffer create failed\n");
            return false;
        }
        u->uploadStaticBuffer(cube_vbuf_.data(), v);
        return true;
    }

    bool create_textures(QRhiResourceUpdateBatch* u) {
        // The TESTED cyclic phase colormap, baked to a repeating 1D texture.
        // QRhi has no 3-channel float format, so RGBA32F with alpha = 1; the
        // REPEAT sampler (sampler_1d_) preserves the cyclic wraparound.
        if (!rhi_->isFeatureSupported(QRhi::OneDimensionalTextures)) {
            std::fprintf(stderr, "SceneRenderer: 1D textures unsupported "
                                 "(the phase LUT needs them)\n");
            return false;
        }
        const std::vector<ses::Rgb> lut = ses::phase_lut(kPhaseLutSize);
        QByteArray lut_f(static_cast<qsizetype>(lut.size()) * 4 * sizeof(float),
                         Qt::Uninitialized);
        float* dst = reinterpret_cast<float*>(lut_f.data());
        for (std::size_t i = 0; i < lut.size(); ++i) {
            dst[4 * i + 0] = static_cast<float>(lut[i].r);
            dst[4 * i + 1] = static_cast<float>(lut[i].g);
            dst[4 * i + 2] = static_cast<float>(lut[i].b);
            dst[4 * i + 3] = 1.0f;
        }
        phase_tex_.reset(rhi_->newTexture(QRhiTexture::RGBA32F, QSize(kPhaseLutSize, 1), 1,
                                          QRhiTexture::OneDimensional));
        if (!phase_tex_->create()) {
            std::fprintf(stderr, "SceneRenderer: phase LUT texture create failed\n");
            return false;
        }
        QRhiTextureSubresourceUploadDescription sub(lut_f.constData(), lut_f.size());
        u->uploadTexture(phase_tex_.data(),
                         QRhiTextureUploadDescription({ QRhiTextureUploadEntry(0, 0, sub) }));
        // The psi display volume on the GPU path is the ENGINE's bridge texture
        // (passed per frame); the CPU fallback texture is created lazily by
        // upload_volume when the shell stages CPU data.
        return true;
    }

    // CPU staging path (GPU engine unavailable): convert the RG staging field
    // to RGBA texels and upload the whole volume, slice by slice.
    void upload_volume(QRhiResourceUpdateBatch* u, const std::vector<float>& psi_staging) {
        const int nx = grid_.x.n;
        const int ny = grid_.y.n;
        const int nz = grid_.z.n;
        if (fallback_tex_.isNull()) {
            fallback_tex_.reset(rhi_->newTexture(QRhiTexture::RGBA32F, nx, ny, nz, 1,
                                                 QRhiTexture::ThreeDimensional));
            if (!fallback_tex_->create()) {
                std::fprintf(stderr,
                             "SceneRenderer: fallback volume texture create failed\n");
                fallback_tex_.reset();
                return;
            }
        }
        const std::size_t cells = static_cast<std::size_t>(grid_.size());
        QByteArray rgba(static_cast<qsizetype>(cells) * 4 * sizeof(float),
                        Qt::Uninitialized);
        float* dst = reinterpret_cast<float*>(rgba.data());
        for (std::size_t i = 0; i < cells; ++i) {
            dst[4 * i + 0] = psi_staging[2 * i];
            dst[4 * i + 1] = psi_staging[2 * i + 1];
            dst[4 * i + 2] = 0.0f;
            dst[4 * i + 3] = 0.0f;
        }
        // For 3D textures the upload-entry layer is the z slice.
        const qsizetype slice_bytes =
            static_cast<qsizetype>(nx) * ny * 4 * sizeof(float);
        QVarLengthArray<QRhiTextureUploadEntry, 256> entries;
        for (int z = 0; z < nz; ++z) {
            QRhiTextureSubresourceUploadDescription sub(
                rgba.constData() + static_cast<qsizetype>(z) * slice_bytes, slice_bytes);
            entries.append(QRhiTextureUploadEntry(z, 0, sub));
        }
        QRhiTextureUploadDescription desc;
        desc.setEntries(entries.cbegin(), entries.cend());
        u->uploadTexture(fallback_tex_.data(), desc);
    }

    // Refill the isosurface vertex buffer from the current marching-cubes mesh
    // (Dynamic-sized: recreate when the mesh outgrows the buffer).
    void upload_mesh(QRhiResourceUpdateBatch* u, const ses::Mesh& mesh,
                     const std::vector<ses::Rgb>& colors) {
        const std::vector<float> interleaved = interleave_mesh(mesh, nullptr, &colors);
        vertex_count_ = static_cast<int>(mesh.vertices.size());
        const quint32 bytes = static_cast<quint32>(interleaved.size() * sizeof(float));
        if (bytes == 0) {
            return;
        }
        if (mesh_vbuf_.isNull() || mesh_vbuf_->size() < bytes) {
            mesh_vbuf_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::VertexBuffer,
                                             bytes));
            if (!mesh_vbuf_->create()) {
                std::fprintf(stderr, "SceneRenderer: mesh buffer create failed\n");
                mesh_vbuf_.reset();
                vertex_count_ = 0;
                return;
            }
        }
        u->updateDynamicBuffer(mesh_vbuf_.data(), 0, bytes, interleaved.data());
    }

    QRhi* rhi_ = nullptr;
    ses::Grid3D grid_{};

    // Mesh pass (isosurface + proton + gizmo + z label).
    QScopedPointer<QRhiGraphicsPipeline> mesh_pipe_;
    QScopedPointer<QRhiBuffer> scene_ubuf_, gizmo_ubuf_;
    QScopedPointer<QRhiShaderResourceBindings> scene_srb_, gizmo_srb_;
    QScopedPointer<QRhiBuffer> mesh_vbuf_;  // dynamic, grows with the mesh
    int vertex_count_ = 0;
    QScopedPointer<QRhiBuffer> proton_vbuf_;
    int proton_vertex_count_ = 0;
    QScopedPointer<QRhiBuffer> gizmo_vbuf_;
    int gizmo_vertex_count_ = 0;
    QScopedPointer<QRhiBuffer> zlabel_vbuf_;  // billboarded "z", rebuilt each frame

    // Volume pass (Cloud view).
    QScopedPointer<QRhiGraphicsPipeline> volume_pipe_;
    QScopedPointer<QRhiBuffer> volume_ubuf_;
    QScopedPointer<QRhiShaderResourceBindings> volume_layout_srb_;  // pipeline template
    QScopedPointer<QRhiShaderResourceBindings> volume_srb_;         // live (psi texture)
    QRhiTexture* volume_tex_bound_ = nullptr;
    QScopedPointer<QRhiBuffer> cube_vbuf_;
    QScopedPointer<QRhiTexture> phase_tex_;
    QScopedPointer<QRhiTexture> fallback_tex_;  // CPU staging path only
    QScopedPointer<QRhiSampler> sampler_3d_, sampler_1d_;
};

}  // namespace ses_shell
