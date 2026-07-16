module;
#include <volk.h>
#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <source_location>
#include <vector>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
export module ses.vk.render;
export import ses.vk.compute;
export import ses.grid;
export import ses.vec;
export import ses.marching_cubes;
export import ses.sphere;
export import ses.camera;
export import ses.colormap;
export import ses.volume;


// ses_vk::SceneRenderer: the whole scene -- isosurface mesh, proton marker,
// axes gizmo, billboarded z label, and the front-to-back |psi|^2 volume
// raymarch -- rendered by raw Vulkan into an OFFSCREEN color image the
// presentation shell samples. The GLSL sources are baked offline to SPIR-V,
// the camera's GL clip conventions are corrected to Vulkan by a fixed matrix
// (store_corrected_mvp), and every frame ends with the composite image handed
// over in SHADER_READ_ONLY.
//
// The presentation contract (what any shell must do): give render() the
// per-frame camera/staging inputs, then draw
// color_image()/color_view() as one textured quad. That is the ENTIRE
// remaining rendering responsibility of the framework.
// ses.vk GMF set, textually pre-claimed: volk.h supplies the VK_* macros
// (macros never cross module boundaries); vk_mem_alloc.h (same config as the
// module GMFs) keeps direct vma* calls compiling; both inoculate the TU
// against GMF/textual redefinitions.


export namespace ses_vk {

constexpr int kPhaseLutSize = 256;
constexpr double kProtonMarkerRadius = 0.35;  // symbolic (real ~1e-5 Bohr)

// SPIR-V blobs for the pipelines + post chain (offline-baked; caller owns).
struct RenderKernels {
    const unsigned char* mesh_vert = nullptr;
    std::size_t mesh_vert_size = 0;
    const unsigned char* mesh_frag = nullptr;
    std::size_t mesh_frag_size = 0;
    const unsigned char* volume_vert = nullptr;
    std::size_t volume_vert_size = 0;
    const unsigned char* volume_frag = nullptr;
    std::size_t volume_frag_size = 0;
    const unsigned char* slice_vert = nullptr;  // cross-section sheet
    std::size_t slice_vert_size = 0;
    const unsigned char* slice_frag = nullptr;
    std::size_t slice_frag_size = 0;
    const unsigned char* accum = nullptr;  // temporal accumulation
    std::size_t accum_size = 0;
    const unsigned char* bloom_down = nullptr;
    std::size_t bloom_down_size = 0;
    const unsigned char* bloom_up = nullptr;
    std::size_t bloom_up_size = 0;
    const unsigned char* compose = nullptr;  // tonemap + bloom + dither
    std::size_t compose_size = 0;
    const unsigned char* particles = nullptr;  // probability-flow advection
    std::size_t particles_size = 0;
    const unsigned char* occupancy = nullptr;  // block-max density
    std::size_t occupancy_size = 0;
    const unsigned char* occ_dilate = nullptr;  // conservative 3^3 dilation
    std::size_t occ_dilate_size = 0;
    const unsigned char* shadow = nullptr;  // key-light transmittance
    std::size_t shadow_size = 0;
    const unsigned char* flow_vert = nullptr;  // streakline vertices
    std::size_t flow_vert_size = 0;
    const unsigned char* flow_frag = nullptr;
    std::size_t flow_frag_size = 0;
};

class SceneRenderer {
public:
    // Per-frame inputs computed by the shell. Non-null mesh/volume_staging
    // pointers request a (re)upload before the pass records.
    struct FrameInput {
        bool cloud = true;  // Cloud (volume) vs Surface (isosurface) view
        double azimuth = 0.0;
        double elevation = 0.0;
        double distance = 0.0;
        double peak = 0.0;        // |psi|^2 brightness normalizer
        double absorbance = 0.0;  // Beer-Lambert opacity
        float flash = 0.0f;       // photon-flash background weight, 0..1
        bool accumulate = false;  // scene static: keep temporal averaging
        float frame_index = 0.0f;  // temporal jitter/dither rotation
        bool flow = false;          // draw the probability-flow particles
        bool flow_animate = false;  // advance the advection (false = paused)
        bool volume_changed = false;  // psi/absorbance changed: rebuild aux
        VkImageView psi_volume = VK_NULL_HANDLE;  // engine bridge; null -> CPU fallback
        VkImageView flow_velocity = VK_NULL_HANDLE;  // fp32 Bohmian v (streaklines)
        const ses::Mesh* mesh = nullptr;          // non-null: upload isosurface
        const std::vector<ses::Rgb>* mesh_colors = nullptr;
        const std::vector<float>* volume_staging = nullptr;  // RG staging
        // GPU-extracted isosurface (engine marching cubes): drawn indirect,
        // same interleaved pos3/normal3/color3 layout as the host mesh.
        VkBuffer gpu_mesh_vbuf = VK_NULL_HANDLE;
        VkBuffer gpu_mesh_indirect = VK_NULL_HANDLE;
        // Cross-section planes (Cloud view). clip cuts away a half-space;
        // slice paints psi on a plane. axis 0/1/2, offset in Bohr.
        bool clip_on = false;
        int clip_axis = 2;
        int clip_sign = 1;      // +1 hides the +normal half
        float clip_offset = 0.0f;
        bool slice_on = false;
        int slice_axis = 2;
        float slice_offset = 0.0f;
        int slice_map = 0;      // 0 density, 1 Re, 2 phase
    };

    SceneRenderer() = default;
    SceneRenderer(const SceneRenderer&) = delete;
    SceneRenderer& operator=(const SceneRenderer&) = delete;
    ~SceneRenderer() { destroy(); }

    bool initialize(DeviceContext& ctx, const ses::Grid3D& grid,
                    const RenderKernels& blobs) {
        ctx_ = &ctx;
        grid_ = grid;
        if (!arena_.create(ctx, 32, 32, 24, 0, 32, 24)) {
            return false;
        }
        if (!create_samplers() || !create_ubos() ||
            !create_pipelines(blobs) || !create_static_geometry() ||
            !create_phase_lut() || !create_post_kernels(blobs) ||
            !create_flow(blobs) || !create_volume_aux(blobs)) {
            return false;
        }
        scene_set_ = arena_.allocate(*ctx_, mesh_dsl_holder_.set);
        gizmo_set_ = arena_.allocate(*ctx_, mesh_dsl_holder_.set);
        volume_set_ = arena_.allocate(*ctx_, vol_dsl_holder_.set);
        if (scene_set_ == VK_NULL_HANDLE || gizmo_set_ == VK_NULL_HANDLE ||
            volume_set_ == VK_NULL_HANDLE) {
            return false;
        }
        const auto uniform = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        arena_.write_buffer(*ctx_, scene_set_, 0, uniform, scene_ubuf_.buf,
                            sizeof(MeshUbo));
        arena_.write_buffer(*ctx_, gizmo_set_, 0, uniform, gizmo_ubuf_.buf,
                            sizeof(MeshUbo));
        arena_.write_buffer(*ctx_, volume_set_, 0, uniform, volume_ubuf_.buf,
                            sizeof(VolumeUbo));
        arena_.write_sampled(*ctx_, volume_set_, 2, phase_tex_.view, samp_1d_);
        // Binding 1 (the psi volume) is pointed per frame; seed it with the
        // LUT so the set is never invalid.
        arena_.write_sampled(*ctx_, volume_set_, 1, phase_tex_.view, samp_3d_);
        // Occupancy (NEAREST: conservative block tests) + shadow volumes.
        arena_.write_sampled(*ctx_, volume_set_, 3, occ_b_.view, samp_nearest_,
                             VK_IMAGE_LAYOUT_GENERAL);
        arena_.write_sampled(*ctx_, volume_set_, 4, shadow_vol_.view, samp_3d_,
                             VK_IMAGE_LAYOUT_GENERAL);
        return true;
    }

    // (Re)create the offscreen target at the given pixel size.
    bool resize(std::uint32_t w, std::uint32_t h) {
        if (w == 0 || h == 0) {
            return false;
        }
        if (w == width_ && h == height_ && color_.view != VK_NULL_HANDLE) {
            return true;
        }
        vkDeviceWaitIdle(ctx_->device);
        destroy_target();
        width_ = w;
        height_ = h;
        accum_frames_ = 0;  // fresh targets: the early-out must not hold
        return create_target();
    }

    std::uint32_t width() const { return width_; }
    std::uint32_t height() const { return height_; }
    // What the presentation shell samples: the tonemapped RGBA8 composite
    // (handed over in SHADER_READ_ONLY after every frame).
    VkImage color_image() const { return present_.img; }
    VkImageView color_view() const { return present_.view; }

    // Record + submit the whole scene into the offscreen color image; on
    // return the image is in SHADER_READ_ONLY_OPTIMAL for the shell's blit.
    bool render(const FrameInput& in) {
        if (color_.view == VK_NULL_HANDLE) {
            return false;
        }
        // Converged early-out: once the temporal accumulation saturates
        // (accum.comp caps the running mean at kAccumCap samples) a static
        // scene's composed output cannot change -- skip the whole submission
        // and let the blit keep sampling the finished present image.
        if (in.accumulate) {
            if (accum_frames_ >= kAccumCap && in.volume_staging == nullptr &&
                in.mesh == nullptr) {
                return true;
            }
            ++accum_frames_;
        } else {
            accum_frames_ = 0;
        }
        update_ubos(in);
        if (in.volume_staging != nullptr) {
            upload_fallback_volume(*in.volume_staging);
        }
        if (in.mesh != nullptr && in.mesh_colors != nullptr) {
            upload_mesh(*in.mesh, *in.mesh_colors);
        }
        if (in.cloud) {
            point_volume_binding(in.psi_volume);
        }

        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return false;
        }
        VkCommandBuffer cb = shot.cb();

        // Rebuild the occupancy + shadow volumes when the displayed field
        // (or absorbance) changed; the pass's fragment reads then see them
        // through the compute-to-fragment edge below.
        if (in.cloud && (in.volume_changed || !aux_valid_) &&
            volume_bound_view_ != VK_NULL_HANDLE) {
            occ_k_.bind(cb, occ_set_);
            vkCmdDispatch(cb, 32, 32, 32);  // one workgroup per occ cell
            barrier_compute_to_compute(cb);
            dilate_k_.bind(cb, dilate_set_);
            vkCmdDispatch(cb, 8, 8, 8);
            barrier_compute_to_compute(cb);
            shadow_k_.bind(cb, shadow_set_);
            vkCmdDispatch(cb, 16, 16, 16);  // 64^3 / 4^3
            memory_barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_ACCESS_SHADER_WRITE_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_ACCESS_SHADER_READ_BIT);
            aux_valid_ = true;
        }

        // Advect the streaklines BEFORE the pass (their trail SSBO is
        // vertex-pulled by the line draw inside it). Dispatch only on the
        // enable-edge (one reset that collapses every trail to a fresh seed,
        // so the streaks grow from nothing) and on animated frames; a paused
        // frame leaves the frozen trail as-is.
        const bool flow_active = in.flow && in.cloud;
        if (flow_active) {
            const bool enabling = !flow_was_on_;
            const bool advancing = in.flow_animate && !enabling;
            if (enabling) {
                flow_frame_ = 0;
            } else if (advancing) {
                ++flow_frame_;
            }
            if (enabling || advancing) {
                FlowParams fp{};
                fp.box_min[0] = static_cast<float>(grid_.x.xmin);
                fp.box_min[1] = static_cast<float>(grid_.y.xmin);
                fp.box_min[2] = static_cast<float>(grid_.z.xmin);
                fp.box_max[0] = static_cast<float>(grid_.x.xmax);
                fp.box_max[1] = static_cast<float>(grid_.y.xmax);
                fp.box_max[2] = static_cast<float>(grid_.z.xmax);
                fp.dt = 0.18f;  // slow crawl: readable streaks from the start
                fp.inv_peak =
                    static_cast<float>(in.peak > 0.0 ? 1.0 / in.peak : 0.0);
                fp.lifetime = 600.0f;
                fp.frame = static_cast<float>(flow_frame_);
                fp.n_streaks = static_cast<std::int32_t>(kFlowStreaks);
                fp.trail_len = static_cast<std::int32_t>(kFlowTrail);
                fp.animate = advancing ? 1 : 0;
                fp.reset = enabling ? 1 : 0;
                std::memcpy(flow_ubo_.mapped, &fp, sizeof(fp));
                vmaFlushAllocation(ctx_->allocator, flow_ubo_.alloc, 0,
                                   VK_WHOLE_SIZE);
                point_flow_velocity_binding(in.flow_velocity);
                particles_k_.bind(cb, particles_set_);
                vkCmdDispatch(cb, (kFlowStreaks + 255) / 256, 1, 1);
                memory_barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_ACCESS_SHADER_WRITE_BIT,
                               VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                               VK_ACCESS_SHADER_READ_BIT);
            }
        }
        flow_was_on_ = flow_active;

        const float w = in.flash;
        // Dynamic rendering: transition the attachments explicitly (no render
        // pass to do it). loadOp=CLEAR discards the old color, so UNDEFINED
        // old-layout is fine, but the previous frame's readers (present sample
        // + post-chain compute imageLoads, which left it in GENERAL) must
        // finish first. Depth uses its own barrier (the helper is color-only).
        ses_vk::image_layout_barrier(
            cb, color_.img, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        {
            VkImageMemoryBarrier2 dib{};
            dib.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            dib.srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            dib.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            dib.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
            dib.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            dib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            dib.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            dib.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            dib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            dib.image = depth_.img;
            dib.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
            VkDependencyInfo ddep{};
            ddep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            ddep.imageMemoryBarrierCount = 1;
            ddep.pImageMemoryBarriers = &dib;
            vkCmdPipelineBarrier2(cb, &ddep);
        }
        VkRenderingAttachmentInfo color_att{};
        color_att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color_att.imageView = color_.view;
        color_att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_att.clearValue.color = {{0.04f + 0.55f * w, 0.05f + 0.42f * w,
                                       0.09f + 0.18f * w, 1.0f}};
        VkRenderingAttachmentInfo depth_att{};
        depth_att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth_att.imageView = depth_.view;
        depth_att.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_att.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_att.clearValue.depthStencil = {1.0f, 0};
        VkRenderingInfo rinfo{};
        rinfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        rinfo.renderArea = {{0, 0}, {width_, height_}};
        rinfo.layerCount = 1;
        rinfo.colorAttachmentCount = 1;
        rinfo.pColorAttachments = &color_att;
        rinfo.pDepthAttachment = &depth_att;
        vkCmdBeginRendering(cb, &rinfo);

        set_viewport_scissor(cb, 0, 0, width_, height_, 0.0f, 1.0f);
        const VkDeviceSize zero_off = 0;

        if (in.cloud) {
            if (volume_bound_view_ != VK_NULL_HANDLE) {
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  volume_pipe_);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        vol_pl_, 0, 1, &volume_set_, 0,
                                        nullptr);
                vkCmdBindVertexBuffers(cb, 0, 1, &cube_vbuf_.buf, &zero_off);
                vkCmdDraw(cb, 36, 1, 0, 0);
            }
            // Cross-section sheet over the (possibly clipped) fog: reuses the
            // volume descriptor set (UBO + psi + phase LUT), 6 procedural
            // verts. Drawn after the volume so it composites on top.
            if (in.slice_on && volume_bound_view_ != VK_NULL_HANDLE) {
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  slice_pipe_);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        vol_pl_, 0, 1, &volume_set_, 0,
                                        nullptr);
                vkCmdDraw(cb, 6, 1, 0, 0);
            }
            if (in.flow) {  // streaklines over the cloud (one strip per streak)
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  flow_pipe_);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        flow_pl_, 0, 1, &flow_set_, 0,
                                        nullptr);
                const std::int32_t tl = static_cast<std::int32_t>(kFlowTrail);
                vkCmdPushConstants(cb, flow_pl_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                                   sizeof(tl), &tl);
                vkCmdDraw(cb, kFlowTrail, kFlowStreaks, 0, 0);
            }
        } else {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh_pipe_);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    mesh_pl_, 0, 1, &scene_set_, 0, nullptr);
            vkCmdBindVertexBuffers(cb, 0, 1, &proton_vbuf_.buf, &zero_off);
            vkCmdDraw(cb, static_cast<std::uint32_t>(proton_vertex_count_), 1,
                      0, 0);
            if (in.gpu_mesh_vbuf != VK_NULL_HANDLE &&
                in.gpu_mesh_indirect != VK_NULL_HANDLE) {
                // GPU-extracted mesh: the vertex count lives on the GPU
                // (indirect command written by the scan pass).
                vkCmdBindVertexBuffers(cb, 0, 1, &in.gpu_mesh_vbuf,
                                       &zero_off);
                vkCmdDrawIndirect(cb, in.gpu_mesh_indirect, 0, 1,
                                  sizeof(VkDrawIndirectCommand));
            } else if (mesh_vbuf_.buf != VK_NULL_HANDLE &&
                       mesh_vertex_count_ > 0) {
                vkCmdBindVertexBuffers(cb, 0, 1, &mesh_vbuf_.buf, &zero_off);
                vkCmdDraw(cb, static_cast<std::uint32_t>(mesh_vertex_count_),
                          1, 0, 0);
            }
        }

        // XYZ gizmo in the bottom-left corner, over both views. The corner
        // viewport maps to depth range [0, 0.01]: always in front of the
        // scene, self-occlusion within the gizmo intact. Coordinates convert
        // from the GL-style bottom-left origin to Vulkan's top-left.
        if (gizmo_vertex_count_ > 0) {
            const int side = std::clamp(
                static_cast<int>(std::min(width_, height_)) / 6, 96, 180);
            const int margin = side / 8;
            const int vk_y = static_cast<int>(height_) - margin - side;
            set_viewport_scissor(cb, margin, vk_y, side, side, 0.0f, 0.01f);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh_pipe_);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    mesh_pl_, 0, 1, &gizmo_set_, 0, nullptr);
            vkCmdBindVertexBuffers(cb, 0, 1, &gizmo_vbuf_.buf, &zero_off);
            vkCmdDraw(cb, static_cast<std::uint32_t>(gizmo_vertex_count_), 1,
                      0, 0);
            vkCmdBindVertexBuffers(cb, 0, 1, &zlabel_vbuf_.buf, &zero_off);
            vkCmdDraw(cb, 18, 1, 0, 0);
        }

        vkCmdEndRendering(cb);
        // Move color to GENERAL for the post-chain compute imageLoads.
        ses_vk::image_layout_barrier(
            cb, color_.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);

        // Per-frame post parameters, then the post chain in the same cb.
        // Interactive (non-accumulating) frames SKIP the accumulation pass --
        // it degenerates to a full-res copy -- and feed the bloom + composite
        // from the scene color directly; accum_ goes stale, so the FIRST
        // accumulating frame runs the pass in reset mode to re-seed it
        // (accum_frames_ was already incremented above: ==1 on that frame).
        const bool use_accum = in.accumulate && !force_accum_reset_;
        if (!use_accum) {
            accum_frames_ = 0;  // a skipped/reset frame restarts the window
        }
        struct alignas(16) AccumParams {
            std::int32_t accumulate;
            std::int32_t p0, p1, p2;
        };
        const AccumParams ap{(use_accum && accum_frames_ > 1) ? 1 : 0, 0, 0, 0};
        force_accum_reset_ = false;
        std::memcpy(accum_ubo_.mapped, &ap, sizeof(ap));
        vmaFlushAllocation(ctx_->allocator, accum_ubo_.alloc, 0,
                           VK_WHOLE_SIZE);
        struct alignas(16) ComposeParams {
            float bloom_strength;
            float exposure;
            float frame;
            float p0;
        };
        const ComposeParams cp{0.35f, 1.0f, in.frame_index, 0.0f};
        std::memcpy(compose_ubo_.mapped, &cp, sizeof(cp));
        vmaFlushAllocation(ctx_->allocator, compose_ubo_.alloc, 0,
                           VK_WHOLE_SIZE);
        record_post(cb, use_accum);

        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        return ok;
    }

    void destroy() {
        if (ctx_ == nullptr) {
            return;
        }
        if (ctx_->device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(ctx_->device);
        }
        destroy_target();
        arena_.destroy(*ctx_);
        destroy_pipe(flow_pipe_);
        destroy_layout(flow_pl_);
        flow_dsl_holder_.destroy(*ctx_);
        particles_k_.destroy(*ctx_);
        ctx_->destroy_buffer(&flow_ubo_);
        ctx_->destroy_buffer(&flow_buf_);
        compose_k_.destroy(*ctx_);
        up_k_.destroy(*ctx_);
        down_k_.destroy(*ctx_);
        accum_k_.destroy(*ctx_);
        ctx_->destroy_buffer(&compose_ubo_);
        for (int i = 0; i < 3; ++i) {
            ctx_->destroy_buffer(&down_ubo_[i]);
        }
        ctx_->destroy_buffer(&accum_ubo_);
        destroy_pipe(mesh_pipe_);
        destroy_pipe(volume_pipe_);
        destroy_pipe(slice_pipe_);
        destroy_layout(mesh_pl_);
        destroy_layout(vol_pl_);
        mesh_dsl_holder_.destroy(*ctx_);
        vol_dsl_holder_.destroy(*ctx_);
        if (samp_3d_ != VK_NULL_HANDLE) {
            vkDestroySampler(ctx_->device, samp_3d_, nullptr);
            samp_3d_ = VK_NULL_HANDLE;
        }
        if (samp_1d_ != VK_NULL_HANDLE) {
            vkDestroySampler(ctx_->device, samp_1d_, nullptr);
            samp_1d_ = VK_NULL_HANDLE;
        }
        if (samp_nearest_ != VK_NULL_HANDLE) {
            vkDestroySampler(ctx_->device, samp_nearest_, nullptr);
            samp_nearest_ = VK_NULL_HANDLE;
        }
        shadow_k_.destroy(*ctx_);
        dilate_k_.destroy(*ctx_);
        occ_k_.destroy(*ctx_);
        ctx_->destroy_image(&occ_a_);
        ctx_->destroy_image(&occ_b_);
        ctx_->destroy_image(&shadow_vol_);
        ctx_->destroy_image(&phase_tex_);
        ctx_->destroy_image(&fallback_tex_);
        ctx_->destroy_buffer(&scene_ubuf_);
        ctx_->destroy_buffer(&gizmo_ubuf_);
        ctx_->destroy_buffer(&volume_ubuf_);
        ctx_->destroy_buffer(&proton_vbuf_);
        ctx_->destroy_buffer(&gizmo_vbuf_);
        ctx_->destroy_buffer(&cube_vbuf_);
        ctx_->destroy_buffer(&zlabel_vbuf_);
        ctx_->destroy_buffer(&mesh_vbuf_);
        ctx_->destroy_buffer(&staging_);
        ctx_ = nullptr;
    }

private:
    // std140-compatible per-pass uniform blocks (match the .vert/.frag).
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
        float jitter_frame = 0.0f;  // temporal raymarch jitter rotation
        float clip[4] = {0, 2, 1, 0};   // enable, axis, sign, offset
        float slice[4] = {0, 2, 0, 0};  // enable, axis, offset, map mode
    };

    struct DslHolder {
        VkDescriptorSetLayout set = VK_NULL_HANDLE;
        void destroy(DeviceContext& ctx) {
            if (set != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(ctx.device, set, nullptr);
                set = VK_NULL_HANDLE;
            }
        }
    };

    // GL clip conventions (the ses camera's output) -> Vulkan clip: y is
    // flipped and depth maps [-1,1] -> [0,1].
    static void store_corrected_mvp(const ses::Mat4& mvp, float* out16) {
        ses::Mat4 c{};  // column-major
        c.m[0] = 1.0;
        c.m[5] = -1.0;
        c.m[10] = 0.5;
        c.m[14] = 0.5;
        c.m[15] = 1.0;
        const ses::Mat4 corrected = c * mvp;
        for (int i = 0; i < 16; ++i) {
            out16[i] = static_cast<float>(corrected.m[i]);
        }
    }

    static void set_viewport_scissor(VkCommandBuffer cb, int x, int y, int w,
                                     int h, float dmin, float dmax) {
        VkViewport vp{};
        vp.x = static_cast<float>(x);
        vp.y = static_cast<float>(y);
        vp.width = static_cast<float>(w);
        vp.height = static_cast<float>(h);
        vp.minDepth = dmin;
        vp.maxDepth = dmax;
        vkCmdSetViewport(cb, 0, 1, &vp);
        VkRect2D sc{{x, y},
                    {static_cast<std::uint32_t>(w),
                     static_cast<std::uint32_t>(h)}};
        vkCmdSetScissor(cb, 0, 1, &sc);
    }

    bool create_samplers() {
        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(ctx_->device, &sci, nullptr, &samp_3d_) !=
            VK_SUCCESS) {
            return false;
        }
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;  // cyclic phase LUT
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        if (vkCreateSampler(ctx_->device, &sci, nullptr, &samp_1d_) !=
            VK_SUCCESS) {
            return false;
        }
        sci.magFilter = VK_FILTER_NEAREST;  // conservative occupancy tests
        sci.minFilter = VK_FILTER_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        return vkCreateSampler(ctx_->device, &sci, nullptr, &samp_nearest_) ==
               VK_SUCCESS;
    }

    // Fixed-size auxiliary volumes (grid-independent normalized coords):
    // occupancy 32^3 (build + dilated) and the 64^3 shadow transmittance,
    // rebuilt only when the displayed volume (or absorbance) changes.
    bool create_volume_aux(const RenderKernels& blobs) {
        const auto simg = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        const auto cis = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        const auto ubo = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        if (!occ_k_.create(*ctx_, blobs.occupancy, blobs.occupancy_size,
                           {{0, simg}, {1, ubo}, {2, cis}}) ||
            !dilate_k_.create(*ctx_, blobs.occ_dilate, blobs.occ_dilate_size,
                              {{0, simg}, {1, simg}}) ||
            !shadow_k_.create(*ctx_, blobs.shadow, blobs.shadow_size,
                              {{0, simg}, {1, ubo}, {2, cis}})) {
            return false;
        }
        const VkImageUsageFlags usage =
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        if (!create_aux_image(32, usage, &occ_a_) ||
            !create_aux_image(32, usage, &occ_b_) ||
            !create_aux_image(64, usage, &shadow_vol_)) {
            return false;
        }
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return false;
        }
        const DeviceContext::Image* imgs[3] = {&occ_a_, &occ_b_, &shadow_vol_};
        for (const DeviceContext::Image* im : imgs) {
            image_layout_barrier(shot.cb(), im->img, VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_GENERAL,
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_ACCESS_SHADER_READ_BIT |
                                     VK_ACCESS_SHADER_WRITE_BIT);
        }
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);

        occ_set_ = arena_.allocate(*ctx_, occ_k_.set_layout());
        dilate_set_ = arena_.allocate(*ctx_, dilate_k_.set_layout());
        shadow_set_ = arena_.allocate(*ctx_, shadow_k_.set_layout());
        if (occ_set_ == VK_NULL_HANDLE || dilate_set_ == VK_NULL_HANDLE ||
            shadow_set_ == VK_NULL_HANDLE) {
            return false;
        }
        arena_.write_image(*ctx_, occ_set_, 0, occ_a_.view);
        arena_.write_buffer(*ctx_, occ_set_, 1, ubo, volume_ubuf_.buf,
                            sizeof(VolumeUbo));
        arena_.write_image(*ctx_, dilate_set_, 0, occ_b_.view);
        arena_.write_image(*ctx_, dilate_set_, 1, occ_a_.view);
        arena_.write_image(*ctx_, shadow_set_, 0, shadow_vol_.view);
        arena_.write_buffer(*ctx_, shadow_set_, 1, ubo, volume_ubuf_.buf,
                            sizeof(VolumeUbo));
        // Binding 2 (psi) of occ/shadow follows point_volume_binding; seed.
        arena_.write_sampled(*ctx_, occ_set_, 2, phase_tex_.view, samp_3d_);
        arena_.write_sampled(*ctx_, shadow_set_, 2, phase_tex_.view, samp_3d_);
        return true;
    }

    bool create_aux_image(std::uint32_t n, VkImageUsageFlags usage,
                          DeviceContext::Image* out) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_3D;
        ici.format = VK_FORMAT_R32_SFLOAT;
        ici.extent = {n, n, n};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo alc{};
        alc.usage = VMA_MEMORY_USAGE_AUTO;
        if (vmaCreateImage(ctx_->allocator, &ici, &alc, &out->img, &out->alloc,
                           nullptr) != VK_SUCCESS) {
            return false;
        }
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = out->img;
        vci.viewType = VK_IMAGE_VIEW_TYPE_3D;
        vci.format = VK_FORMAT_R32_SFLOAT;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        return vkCreateImageView(ctx_->device, &vci, nullptr, &out->view) ==
               VK_SUCCESS;
    }

    bool create_ubos() {
        const MeshUbo m0{};
        const VolumeUbo v0{};
        return write_host(&scene_ubuf_, &m0, sizeof(m0),
                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) &&
               write_host(&gizmo_ubuf_, &m0, sizeof(m0),
                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) &&
               write_host(&volume_ubuf_, &v0, sizeof(v0),
                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) &&
               write_host(&zlabel_vbuf_, nullptr, 18 * 9 * sizeof(float),
                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    }

    bool write_host(Buffer* b, const void* data, VkDeviceSize size,
                    VkBufferUsageFlags usage) {
        if (!ctx_->create_host_buffer(size, usage, b)) {
            return false;
        }
        if (data != nullptr) {
            std::memcpy(b->mapped, data, size);
            vmaFlushAllocation(ctx_->allocator, b->alloc, 0, VK_WHOLE_SIZE);
        }
        return true;
    }


    bool create_target() {
        // COLOR_ATTACHMENT (scene draw) + STORAGE (compute post chain) +
        // SAMPLED (bloom-down samples it with a linear filter, and the shell
        // blit samples the finished image).
        if (!create_attachment(VK_FORMAT_R16G16B16A16_SFLOAT,
                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                   VK_IMAGE_USAGE_STORAGE_BIT |
                                   VK_IMAGE_USAGE_SAMPLED_BIT,
                               VK_IMAGE_ASPECT_COLOR_BIT, &color_, width_,
                               height_) ||
            !create_attachment(VK_FORMAT_D32_SFLOAT,
                               VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                               VK_IMAGE_ASPECT_DEPTH_BIT, &depth_, width_,
                               height_)) {
            return false;
        }
        return create_post_chain();
    }

    bool create_attachment(VkFormat fmt, VkImageUsageFlags usage,
                           VkImageAspectFlags aspect,
                           DeviceContext::Image* out, std::uint32_t w,
                           std::uint32_t h) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = fmt;
        ici.extent = {w, h, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo alc{};
        alc.usage = VMA_MEMORY_USAGE_AUTO;
        if (vmaCreateImage(ctx_->allocator, &ici, &alc, &out->img, &out->alloc,
                           nullptr) != VK_SUCCESS) {
            return false;
        }
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = out->img;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = fmt;
        vci.subresourceRange = {aspect, 0, 1, 0, 1};
        return vkCreateImageView(ctx_->device, &vci, nullptr, &out->view) ==
               VK_SUCCESS;
    }

    void destroy_target() {
        ctx_->destroy_image(&color_);
        ctx_->destroy_image(&depth_);
        ctx_->destroy_image(&accum_);
        for (int i = 0; i < 3; ++i) {
            ctx_->destroy_image(&bloom_[i]);
        }
        ctx_->destroy_image(&present_);
    }

    // The post-processing chain images (all storage, kept in GENERAL except
    // the present image, which round-trips to SHADER_READ_ONLY for the
    // shell) + their descriptor sets, rebuilt with the target on resize.
    bool create_post_chain() {
        const std::uint32_t hw = std::max(1u, width_ / 2);
        const std::uint32_t hh = std::max(1u, height_ / 2);
        const std::uint32_t qw = std::max(1u, hw / 2);
        const std::uint32_t qh = std::max(1u, hh / 2);
        const std::uint32_t ew = std::max(1u, qw / 2);
        const std::uint32_t eh = std::max(1u, qh / 2);
        const VkImageUsageFlags post_usage =
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        if (!create_attachment(VK_FORMAT_R16G16B16A16_SFLOAT, post_usage,
                               VK_IMAGE_ASPECT_COLOR_BIT, &accum_, width_,
                               height_) ||
            !create_attachment(VK_FORMAT_R16G16B16A16_SFLOAT, post_usage,
                               VK_IMAGE_ASPECT_COLOR_BIT, &bloom_[0], hw, hh) ||
            !create_attachment(VK_FORMAT_R16G16B16A16_SFLOAT, post_usage,
                               VK_IMAGE_ASPECT_COLOR_BIT, &bloom_[1], qw, qh) ||
            !create_attachment(VK_FORMAT_R16G16B16A16_SFLOAT, post_usage,
                               VK_IMAGE_ASPECT_COLOR_BIT, &bloom_[2], ew, eh) ||
            // TRANSFER_SRC: the shell's --dump-frame readback copies from it.
            !create_attachment(VK_FORMAT_R8G8B8A8_UNORM,
                               post_usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                               VK_IMAGE_ASPECT_COLOR_BIT, &present_, width_,
                               height_)) {
            return false;
        }
        bloom_size_[0] = {hw, hh};
        bloom_size_[1] = {qw, qh};
        bloom_size_[2] = {ew, eh};

        // One transition pass: everything to GENERAL.
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return false;
        }
        const DeviceContext::Image* imgs[5] = {&accum_, &bloom_[0], &bloom_[1],
                                               &bloom_[2], &present_};
        for (const DeviceContext::Image* im : imgs) {
            image_layout_barrier(shot.cb(), im->img, VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_GENERAL,
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_ACCESS_SHADER_READ_BIT |
                                     VK_ACCESS_SHADER_WRITE_BIT);
        }
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        present_layout_ = VK_IMAGE_LAYOUT_GENERAL;
        force_accum_reset_ = true;

        // (Re)point the post sets at the fresh images.
        const auto ubo = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        if (accum_set_ == VK_NULL_HANDLE) {
            accum_set_ = arena_.allocate(*ctx_, accum_k_.set_layout());
            for (int i = 0; i < 3; ++i) {
                down_set_[i] = arena_.allocate(*ctx_, down_k_.set_layout());
            }
            for (int i = 0; i < 2; ++i) {
                up_set_[i] = arena_.allocate(*ctx_, up_k_.set_layout());
            }
            compose_set_ = arena_.allocate(*ctx_, compose_k_.set_layout());
            // Interactive-frame variants: bloom + composite read the scene
            // color directly (the accumulation pass is skipped then).
            down0_color_set_ = arena_.allocate(*ctx_, down_k_.set_layout());
            compose_color_set_ = arena_.allocate(*ctx_, compose_k_.set_layout());
            if (accum_set_ == VK_NULL_HANDLE ||
                down_set_[0] == VK_NULL_HANDLE ||
                down_set_[1] == VK_NULL_HANDLE ||
                down_set_[2] == VK_NULL_HANDLE ||
                up_set_[0] == VK_NULL_HANDLE || up_set_[1] == VK_NULL_HANDLE ||
                compose_set_ == VK_NULL_HANDLE ||
                down0_color_set_ == VK_NULL_HANDLE ||
                compose_color_set_ == VK_NULL_HANDLE) {
                return false;
            }
        }
        arena_.write_image(*ctx_, accum_set_, 0, color_.view);
        arena_.write_image(*ctx_, accum_set_, 1, accum_.view);
        arena_.write_buffer(*ctx_, accum_set_, 2, ubo, accum_ubo_.buf, 16);
        // Downsample chain: accum -> half (bright pass), half -> quarter,
        // quarter -> eighth. Sampled reads run in GENERAL.
        const VkImageView down_src[3] = {accum_.view, bloom_[0].view,
                                         bloom_[1].view};
        for (int i = 0; i < 3; ++i) {
            arena_.write_sampled(*ctx_, down_set_[i], 0, down_src[i], samp_3d_,
                                 VK_IMAGE_LAYOUT_GENERAL);
            arena_.write_image(*ctx_, down_set_[i], 1, bloom_[i].view);
            arena_.write_buffer(*ctx_, down_set_[i], 2, ubo,
                                down_ubo_[i].buf, 16);
        }
        // Upsample additive: eighth -> quarter, quarter -> half.
        arena_.write_sampled(*ctx_, up_set_[0], 0, bloom_[2].view, samp_3d_,
                             VK_IMAGE_LAYOUT_GENERAL);
        arena_.write_image(*ctx_, up_set_[0], 1, bloom_[1].view);
        arena_.write_sampled(*ctx_, up_set_[1], 0, bloom_[1].view, samp_3d_,
                             VK_IMAGE_LAYOUT_GENERAL);
        arena_.write_image(*ctx_, up_set_[1], 1, bloom_[0].view);
        // Compose: accum + half-res bloom -> present.
        arena_.write_image(*ctx_, compose_set_, 0, accum_.view);
        arena_.write_sampled(*ctx_, compose_set_, 1, bloom_[0].view, samp_3d_,
                             VK_IMAGE_LAYOUT_GENERAL);
        arena_.write_image(*ctx_, compose_set_, 2, present_.view);
        arena_.write_buffer(*ctx_, compose_set_, 3, ubo, compose_ubo_.buf, 16);
        // The interactive variants read the scene color where the accum-fed
        // sets read accum_ (same params UBOs).
        arena_.write_sampled(*ctx_, down0_color_set_, 0, color_.view, samp_3d_,
                             VK_IMAGE_LAYOUT_GENERAL);
        arena_.write_image(*ctx_, down0_color_set_, 1, bloom_[0].view);
        arena_.write_buffer(*ctx_, down0_color_set_, 2, ubo, down_ubo_[0].buf,
                            16);
        arena_.write_image(*ctx_, compose_color_set_, 0, color_.view);
        arena_.write_sampled(*ctx_, compose_color_set_, 1, bloom_[0].view,
                             samp_3d_, VK_IMAGE_LAYOUT_GENERAL);
        arena_.write_image(*ctx_, compose_color_set_, 2, present_.view);
        arena_.write_buffer(*ctx_, compose_color_set_, 3, ubo,
                            compose_ubo_.buf, 16);

        // Constant down-pass params (bright threshold on the first).
        struct alignas(16) DownParams {
            std::int32_t bright;
            float threshold;
            float p0, p1;
        };
        const DownParams d0{1, 0.6f, 0, 0};  // bright pass off the accum
        const DownParams dn{0, 0.0f, 0, 0};
        std::memcpy(down_ubo_[0].mapped, &d0, sizeof(d0));
        std::memcpy(down_ubo_[1].mapped, &dn, sizeof(dn));
        std::memcpy(down_ubo_[2].mapped, &dn, sizeof(dn));
        for (int i = 0; i < 3; ++i) {
            vmaFlushAllocation(ctx_->allocator, down_ubo_[i].alloc, 0,
                               VK_WHOLE_SIZE);
        }
        return true;
    }

    VkShaderModule make_module(const unsigned char* spv, std::size_t size) {
        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = size;
        smci.pCode = reinterpret_cast<const std::uint32_t*>(spv);
        VkShaderModule mod = VK_NULL_HANDLE;
        vkCreateShaderModule(ctx_->device, &smci, nullptr, &mod);
        return mod;
    }

    bool create_pipelines(const RenderKernels& blobs) {
        // Set layouts. Mesh: UBO 0 (vert+frag). Volume: UBO 0 + samplers 1/2.
        const VkShaderStageFlags vf =
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        {
            const VkDescriptorSetLayoutBinding b{
                0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, vf, nullptr};
            VkDescriptorSetLayoutCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ci.bindingCount = 1;
            ci.pBindings = &b;
            if (vkCreateDescriptorSetLayout(ctx_->device, &ci, nullptr,
                                            &mesh_dsl_holder_.set) !=
                VK_SUCCESS) {
                return false;
            }
        }
        {
            const VkDescriptorSetLayoutBinding bs[5] = {
                {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, vf, nullptr},
                {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                 VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                 VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                 VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                 VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            };
            VkDescriptorSetLayoutCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ci.bindingCount = 5;
            ci.pBindings = bs;
            if (vkCreateDescriptorSetLayout(ctx_->device, &ci, nullptr,
                                            &vol_dsl_holder_.set) !=
                VK_SUCCESS) {
                return false;
            }
        }
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &mesh_dsl_holder_.set;
        if (vkCreatePipelineLayout(ctx_->device, &plci, nullptr, &mesh_pl_) !=
            VK_SUCCESS) {
            return false;
        }
        plci.pSetLayouts = &vol_dsl_holder_.set;
        if (vkCreatePipelineLayout(ctx_->device, &plci, nullptr, &vol_pl_) !=
            VK_SUCCESS) {
            return false;
        }

        // Mesh pipeline: interleaved pos3/normal3/color3, depth on, no cull,
        // no blend. Volume: pos3 cube proxy, cull FRONT, no depth,
        // premultiplied One/OneMinusSrcAlpha. Viewport+scissor dynamic.
        const VkVertexInputBindingDescription mesh_bind{
            0, 9 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
        const VkVertexInputAttributeDescription mesh_attrs[3] = {
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 3 * sizeof(float)},
            {2, 0, VK_FORMAT_R32G32B32_SFLOAT, 6 * sizeof(float)},
        };
        const VkVertexInputBindingDescription cube_bind{
            0, 3 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
        const VkVertexInputAttributeDescription cube_attr{
            0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};

        mesh_pipe_ = build_pipeline(
            blobs.mesh_vert, blobs.mesh_vert_size, blobs.mesh_frag,
            blobs.mesh_frag_size, mesh_pl_, &mesh_bind, mesh_attrs, 3,
            /*depth=*/true, VK_CULL_MODE_NONE, kBlendOff,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        volume_pipe_ = build_pipeline(
            blobs.volume_vert, blobs.volume_vert_size, blobs.volume_frag,
            blobs.volume_frag_size, vol_pl_, &cube_bind, &cube_attr, 1,
            /*depth=*/false, VK_CULL_MODE_FRONT_BIT, kBlendPremultiplied,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        // Cross-section sheet: procedural quad (no vertex input), same UBO +
        // psi/phase samplers as the volume, drawn double-sided over the fog.
        slice_pipe_ = build_pipeline(
            blobs.slice_vert, blobs.slice_vert_size, blobs.slice_frag,
            blobs.slice_frag_size, vol_pl_, nullptr, nullptr, 0,
            /*depth=*/false, VK_CULL_MODE_NONE, kBlendPremultiplied,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        return mesh_pipe_ != VK_NULL_HANDLE && volume_pipe_ != VK_NULL_HANDLE &&
               slice_pipe_ != VK_NULL_HANDLE;
    }

    enum BlendMode { kBlendOff, kBlendPremultiplied, kBlendAdditive };

    VkPipeline build_pipeline(const unsigned char* vs_spv, std::size_t vs_size,
                              const unsigned char* fs_spv, std::size_t fs_size,
                              VkPipelineLayout layout,
                              const VkVertexInputBindingDescription* bind,
                              const VkVertexInputAttributeDescription* attrs,
                              std::uint32_t attr_count, bool depth,
                              VkCullModeFlags cull, BlendMode blend,
                              VkPrimitiveTopology topo) {
        VkShaderModule vs = make_module(vs_spv, vs_size);
        VkShaderModule fs = make_module(fs_spv, fs_size);
        if (vs == VK_NULL_HANDLE || fs == VK_NULL_HANDLE) {
            std::fprintf(stderr, "vk_render: shader module create failed\n");
            return VK_NULL_HANDLE;
        }
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vs;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fs;
        stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vin{};
        vin.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vin.vertexBindingDescriptionCount = (bind != nullptr) ? 1 : 0;
        vin.pVertexBindingDescriptions = bind;
        vin.vertexAttributeDescriptionCount = attr_count;
        vin.pVertexAttributeDescriptions = attrs;

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType =
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = topo;

        VkPipelineViewportStateCreateInfo vpst{};
        vpst.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vpst.viewportCount = 1;
        vpst.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType =
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = cull;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType =
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = depth ? VK_TRUE : VK_FALSE;
        ds.depthWriteEnable = depth ? VK_TRUE : VK_FALSE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS;

        VkPipelineColorBlendAttachmentState ba{};
        ba.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        if (blend == kBlendPremultiplied) {
            ba.blendEnable = VK_TRUE;
            ba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            ba.colorBlendOp = VK_BLEND_OP_ADD;
            ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            ba.alphaBlendOp = VK_BLEND_OP_ADD;
        } else if (blend == kBlendAdditive) {  // glow sprites
            ba.blendEnable = VK_TRUE;
            ba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            ba.colorBlendOp = VK_BLEND_OP_ADD;
            ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            ba.alphaBlendOp = VK_BLEND_OP_ADD;
        }
        VkPipelineColorBlendStateCreateInfo cbst{};
        cbst.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cbst.attachmentCount = 1;
        cbst.pAttachments = &ba;

        const VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT,
                                       VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynst{};
        dynst.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynst.dynamicStateCount = 2;
        dynst.pDynamicStates = dyn;

        // Dynamic rendering: the pipeline declares attachment FORMATS instead
        // of referencing a render pass. Must match create_target's images.
        const VkFormat color_fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
        VkPipelineRenderingCreateInfo prci{};
        prci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        prci.colorAttachmentCount = 1;
        prci.pColorAttachmentFormats = &color_fmt;
        prci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
        VkGraphicsPipelineCreateInfo gpci{};
        gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gpci.pNext = &prci;
        gpci.stageCount = 2;
        gpci.pStages = stages;
        gpci.pVertexInputState = &vin;
        gpci.pInputAssemblyState = &ia;
        gpci.pViewportState = &vpst;
        gpci.pRasterizationState = &rs;
        gpci.pMultisampleState = &ms;
        gpci.pDepthStencilState = &ds;
        gpci.pColorBlendState = &cbst;
        gpci.pDynamicState = &dynst;
        gpci.layout = layout;
        gpci.renderPass = VK_NULL_HANDLE;
        VkPipeline pipe = VK_NULL_HANDLE;
        if (vkCreateGraphicsPipelines(ctx_->device, VK_NULL_HANDLE, 1, &gpci,
                                      nullptr, &pipe) != VK_SUCCESS) {
            std::fprintf(stderr, "vk_render: graphics pipeline create failed\n");
            pipe = VK_NULL_HANDLE;
        }
        vkDestroyShaderModule(ctx_->device, vs, nullptr);
        vkDestroyShaderModule(ctx_->device, fs, nullptr);
        return pipe;
    }

    bool create_post_kernels(const RenderKernels& blobs) {
        const auto simg = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        const auto cis = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        const auto ubo = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        if (!accum_k_.create(*ctx_, blobs.accum, blobs.accum_size,
                             {{0, simg}, {1, simg}, {2, ubo}}) ||
            !down_k_.create(*ctx_, blobs.bloom_down, blobs.bloom_down_size,
                            {{0, cis}, {1, simg}, {2, ubo}}) ||
            !up_k_.create(*ctx_, blobs.bloom_up, blobs.bloom_up_size,
                          {{0, cis}, {1, simg}}) ||
            !compose_k_.create(*ctx_, blobs.compose, blobs.compose_size,
                               {{0, simg}, {1, cis}, {2, simg}, {3, ubo}})) {
            return false;
        }
        // Tiny per-pass parameter blocks (host-mapped).
        const std::uint32_t zero16[4] = {0, 0, 0, 0};
        if (!write_host(&accum_ubo_, zero16, 16,
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ||
            !write_host(&down_ubo_[0], zero16, 16,
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ||
            !write_host(&down_ubo_[1], zero16, 16,
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ||
            !write_host(&down_ubo_[2], zero16, 16,
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ||
            !write_host(&compose_ubo_, zero16, 16,
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) {
            return false;
        }
        return true;
    }

    // Probability-flow streaklines: the trail SSBO, its advection kernel, and
    // the LINE_STRIP pipeline drawn inside the scene pass.
    bool create_flow(const RenderKernels& blobs) {
        const auto sbuf = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        const auto cis = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        const auto ubo = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        if (!particles_k_.create(*ctx_, blobs.particles, blobs.particles_size,
                                 {{0, sbuf}, {1, ubo}, {2, cis}, {3, cis}})) {
            return false;
        }
        // Flow draw set layout: SSBO + volume UBO + psi + LUT, vertex stage.
        {
            const VkDescriptorSetLayoutBinding bs[4] = {
                {0, sbuf, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
                {1, ubo, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
                {2, cis, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
                {3, cis, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
            };
            VkDescriptorSetLayoutCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ci.bindingCount = 4;
            ci.pBindings = bs;
            if (vkCreateDescriptorSetLayout(ctx_->device, &ci, nullptr,
                                            &flow_dsl_holder_.set) !=
                VK_SUCCESS) {
                return false;
            }
        }
        // Push constant: the trail length, so flow.vert can index
        // trail[instance*trail_len + vertex] and fade along the streak.
        const VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0,
                                      sizeof(std::int32_t)};
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &flow_dsl_holder_.set;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(ctx_->device, &plci, nullptr, &flow_pl_) !=
            VK_SUCCESS) {
            return false;
        }
        // Streaklines: instanced LINE_STRIP (one strip per streak), premultiplied
        // white with a tail-to-head alpha fade (weather-map Lagrangian flux).
        flow_pipe_ = build_pipeline(
            blobs.flow_vert, blobs.flow_vert_size, blobs.flow_frag,
            blobs.flow_frag_size, flow_pl_, nullptr, nullptr, 0,
            /*depth=*/false, VK_CULL_MODE_NONE, kBlendPremultiplied,
            VK_PRIMITIVE_TOPOLOGY_LINE_STRIP);
        if (flow_pipe_ == VK_NULL_HANDLE) {
            return false;
        }

        // Seed the trail buffer: each streak's trail_len slots collapsed to one
        // born-dead position (age 1e9) in the inner half of the box; the first
        // frame's rejection respawn spreads them onto |psi|^2.
        std::vector<float> seed(4 * kFlowStreaks * kFlowTrail);
        std::uint32_t rng = 0x9e3779b9u;
        auto rnd = [&rng]() {
            rng = rng * 747796405u + 2891336453u;
            std::uint32_t w = ((rng >> ((rng >> 28u) + 4u)) ^ rng) * 277803737u;
            w = (w >> 22u) ^ w;
            return static_cast<float>(w) / 4294967296.0f;
        };
        const float ext[3] = {
            static_cast<float>(grid_.x.xmax - grid_.x.xmin),
            static_cast<float>(grid_.y.xmax - grid_.y.xmin),
            static_cast<float>(grid_.z.xmax - grid_.z.xmin)};
        const float lo[3] = {static_cast<float>(grid_.x.xmin),
                             static_cast<float>(grid_.y.xmin),
                             static_cast<float>(grid_.z.xmin)};
        for (std::uint32_t i = 0; i < kFlowStreaks; ++i) {
            const float px = lo[0] + ext[0] * (0.25f + 0.5f * rnd());
            const float py = lo[1] + ext[1] * (0.25f + 0.5f * rnd());
            const float pz = lo[2] + ext[2] * (0.25f + 0.5f * rnd());
            for (std::uint32_t k = 0; k < kFlowTrail; ++k) {
                const std::size_t o = 4 * (i * kFlowTrail + k);
                seed[o + 0] = px;
                seed[o + 1] = py;
                seed[o + 2] = pz;
                seed[o + 3] = 1e9f;  // born dead: respawn onto |psi|^2
            }
        }
        const VkDeviceSize bytes = seed.size() * sizeof(float);
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = bytes;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo alc{};
        alc.usage = VMA_MEMORY_USAGE_AUTO;
        if (vmaCreateBuffer(ctx_->allocator, &bci, &alc, &flow_buf_.buf,
                            &flow_buf_.alloc, nullptr) != VK_SUCCESS) {
            return false;
        }
        if (!ensure_staging(bytes)) {
            return false;
        }
        std::memcpy(staging_.mapped, seed.data(), bytes);
        vmaFlushAllocation(ctx_->allocator, staging_.alloc, 0, VK_WHOLE_SIZE);
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return false;
        }
        const VkBufferCopy up{0, 0, bytes};
        vkCmdCopyBuffer(shot.cb(), staging_.buf, flow_buf_.buf, 1, &up);
        memory_barrier(shot.cb(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);

        const std::uint32_t zero16[4] = {0, 0, 0, 0};
        if (!write_host(&flow_ubo_, zero16, sizeof(FlowParams),
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) {
            return false;
        }
        particles_set_ = arena_.allocate(*ctx_, particles_k_.set_layout());
        flow_set_ = arena_.allocate(*ctx_, flow_dsl_holder_.set);
        if (particles_set_ == VK_NULL_HANDLE || flow_set_ == VK_NULL_HANDLE) {
            return false;
        }
        arena_.write_buffer(*ctx_, particles_set_, 0, sbuf, flow_buf_.buf);
        arena_.write_buffer(*ctx_, particles_set_, 1, ubo, flow_ubo_.buf,
                            sizeof(FlowParams));
        arena_.write_buffer(*ctx_, flow_set_, 0, sbuf, flow_buf_.buf);
        arena_.write_buffer(*ctx_, flow_set_, 1, ubo, volume_ubuf_.buf,
                            sizeof(VolumeUbo));
        arena_.write_sampled(*ctx_, flow_set_, 3, phase_tex_.view, samp_1d_);
        // Binding 2 (psi) of both sets is pointed per frame alongside the
        // volume set; seed with the LUT so they are never invalid.
        arena_.write_sampled(*ctx_, particles_set_, 2, phase_tex_.view,
                             samp_3d_);
        arena_.write_sampled(*ctx_, flow_set_, 2, phase_tex_.view, samp_3d_);
        // Binding 3 (velocity) of the advect set is pointed per frame; seed.
        arena_.write_sampled(*ctx_, particles_set_, 3, phase_tex_.view,
                             samp_3d_);
        return true;
    }

    // Record the post chain into the frame's command buffer: accumulation
    // (only while accumulating -- interactive frames feed the bloom and the
    // composite from the scene color directly), the dual-filter bloom
    // pyramid, and the tonemapped composite into the present image (handed
    // to SHADER_READ_ONLY for the shell).
    void record_post(VkCommandBuffer cb, bool use_accum) {
        auto dispatch_2d = [cb](const Kernel& k, VkDescriptorSet set,
                                std::uint32_t w, std::uint32_t h) {
            k.bind(cb, set);
            vkCmdDispatch(cb, (w + 15) / 16, (h + 15) / 16, 1);
        };
        if (use_accum) {
            // Scene (GENERAL after the pass) -> accumulation buffer.
            dispatch_2d(accum_k_, accum_set_, width_, height_);
            barrier_compute_to_compute(cb);
        }
        // Bright pass + down pyramid.
        dispatch_2d(down_k_, use_accum ? down_set_[0] : down0_color_set_,
                    bloom_size_[0].width, bloom_size_[0].height);
        barrier_compute_to_compute(cb);
        dispatch_2d(down_k_, down_set_[1], bloom_size_[1].width,
                    bloom_size_[1].height);
        barrier_compute_to_compute(cb);
        dispatch_2d(down_k_, down_set_[2], bloom_size_[2].width,
                    bloom_size_[2].height);
        barrier_compute_to_compute(cb);
        // Up pyramid (additive).
        dispatch_2d(up_k_, up_set_[0], bloom_size_[1].width,
                    bloom_size_[1].height);
        barrier_compute_to_compute(cb);
        dispatch_2d(up_k_, up_set_[1], bloom_size_[0].width,
                    bloom_size_[0].height);
        barrier_compute_to_compute(cb);
        // Composite -> present, then hand it to the shell's sampling.
        if (present_layout_ != VK_IMAGE_LAYOUT_GENERAL) {
            image_layout_barrier(
                cb, present_.img, present_layout_, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT);
        }
        dispatch_2d(compose_k_, use_accum ? compose_set_ : compose_color_set_,
                    width_, height_);
        image_layout_barrier(cb, present_.img, VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_ACCESS_SHADER_WRITE_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             VK_ACCESS_SHADER_READ_BIT);
        present_layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    void destroy_pipe(VkPipeline& p) {
        if (p != VK_NULL_HANDLE) {
            vkDestroyPipeline(ctx_->device, p, nullptr);
            p = VK_NULL_HANDLE;
        }
    }
    void destroy_layout(VkPipelineLayout& l) {
        if (l != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(ctx_->device, l, nullptr);
            l = VK_NULL_HANDLE;
        }
    }

    // ---- static geometry --------------------------------------------------
    static std::vector<float> interleave_mesh(
        const ses::Mesh& mesh, const float* solid_rgb,
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

    bool upload_device_vbuf(Buffer* dst, const void* data,
                            VkDeviceSize bytes) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = bytes;
        bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo alc{};
        alc.usage = VMA_MEMORY_USAGE_AUTO;
        if (vmaCreateBuffer(ctx_->allocator, &bci, &alc, &dst->buf,
                            &dst->alloc, nullptr) != VK_SUCCESS) {
            return false;
        }
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
        vkCmdCopyBuffer(shot.cb(), staging_.buf, dst->buf, 1, &up);
        memory_barrier(shot.cb(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                       VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT);
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        return ok;
    }

    bool ensure_staging(VkDeviceSize bytes) {
        if (bytes <= staging_bytes_) {
            return true;
        }
        vkDeviceWaitIdle(ctx_->device);
        ctx_->destroy_buffer(&staging_);
        if (!ctx_->create_host_buffer(bytes,
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                      &staging_)) {
            return false;
        }
        staging_bytes_ = bytes;
        return true;
    }

    bool create_static_geometry() {
        // Proton marker: warm sphere at the nucleus.
        {
            const ses::Mesh sphere =
                ses::sphere_mesh(ses::Vec3d{}, kProtonMarkerRadius, 16, 24);
            const float rgb[3] = {1.0f, 0.45f, 0.20f};
            const std::vector<float> iv = interleave_mesh(sphere, rgb, nullptr);
            proton_vertex_count_ = static_cast<int>(sphere.vertices.size());
            if (!upload_device_vbuf(&proton_vbuf_, iv.data(),
                                    iv.size() * sizeof(float))) {
                return false;
            }
        }
        // XYZ gizmo: three colored arrows.
        {
            struct Axis {
                ses::Vec3d dir;
                float r, g, b;
            };
            const Axis axes[3] = {
                {ses::Vec3d{1.0, 0.0, 0.0}, 0.95f, 0.25f, 0.25f},
                {ses::Vec3d{0.0, 1.0, 0.0}, 0.30f, 0.85f, 0.30f},
                {ses::Vec3d{0.0, 0.0, 1.0}, 0.35f, 0.55f, 1.00f},
            };
            std::vector<float> iv;
            for (const Axis& ax : axes) {
                const ses::Mesh arrow =
                    ses::arrow_mesh(ax.dir, 1.0, 0.045, 0.13, 0.32, 16);
                const float rgb[3] = {ax.r, ax.g, ax.b};
                const std::vector<float> part =
                    interleave_mesh(arrow, rgb, nullptr);
                iv.insert(iv.end(), part.begin(), part.end());
            }
            gizmo_vertex_count_ = static_cast<int>(iv.size() / 9);
            if (!upload_device_vbuf(&gizmo_vbuf_, iv.data(),
                                    iv.size() * sizeof(float))) {
                return false;
            }
        }
        // Volume proxy cube: 12 triangles, CCW seen from OUTSIDE.
        {
            const float lo[3] = {static_cast<float>(grid_.x.xmin),
                                 static_cast<float>(grid_.y.xmin),
                                 static_cast<float>(grid_.z.xmin)};
            const float hi[3] = {static_cast<float>(grid_.x.xmax),
                                 static_cast<float>(grid_.y.xmax),
                                 static_cast<float>(grid_.z.xmax)};
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
            if (!upload_device_vbuf(&cube_vbuf_, v, sizeof(v))) {
                return false;
            }
        }
        return true;
    }

    // The cyclic phase colormap as a repeating 1D RGBA8 texture.
    bool create_phase_lut() {
        // RGBA8 not a float format: a colormap's 1/255 steps are invisible, and
        // 8-bit texels filter at full rate on the raymarch's hottest per-step tap.
        const std::vector<ses::Rgb> lut = ses::phase_lut(kPhaseLutSize);
        std::vector<std::uint8_t> texels(4 * lut.size());
        auto quantize = [](double c) {
            return static_cast<std::uint8_t>(
                std::lround(std::clamp(c, 0.0, 1.0) * 255.0));
        };
        for (std::size_t i = 0; i < lut.size(); ++i) {
            texels[4 * i + 0] = quantize(lut[i].r);
            texels[4 * i + 1] = quantize(lut[i].g);
            texels[4 * i + 2] = quantize(lut[i].b);
            texels[4 * i + 3] = 255;
        }
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_1D;
        ici.format = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent = {kPhaseLutSize, 1, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage =
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo alc{};
        alc.usage = VMA_MEMORY_USAGE_AUTO;
        if (vmaCreateImage(ctx_->allocator, &ici, &alc, &phase_tex_.img,
                           &phase_tex_.alloc, nullptr) != VK_SUCCESS) {
            return false;
        }
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = phase_tex_.img;
        vci.viewType = VK_IMAGE_VIEW_TYPE_1D;
        vci.format = VK_FORMAT_R8G8B8A8_UNORM;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(ctx_->device, &vci, nullptr, &phase_tex_.view) !=
            VK_SUCCESS) {
            return false;
        }
        return upload_image(phase_tex_.img, texels.data(), texels.size(),
                            {kPhaseLutSize, 1, 1});
    }

    // staging -> image (whole extent), UNDEFINED/READ -> DST -> READ_ONLY.
    bool upload_image(VkImage img, const void* data, VkDeviceSize bytes,
                      VkExtent3D extent, bool first = true) {
        if (!ensure_staging(bytes)) {
            return false;
        }
        std::memcpy(staging_.mapped, data, bytes);
        vmaFlushAllocation(ctx_->allocator, staging_.alloc, 0, VK_WHOLE_SIZE);
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return false;
        }
        image_layout_barrier(
            shot.cb(), img,
            first ? VK_IMAGE_LAYOUT_UNDEFINED
                  : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            first ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                  : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            first ? 0 : VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = extent;
        vkCmdCopyBufferToImage(shot.cb(), staging_.buf, img,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &region);
        image_layout_barrier(shot.cb(), img,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_ACCESS_TRANSFER_WRITE_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             VK_ACCESS_SHADER_READ_BIT);
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        return ok;
    }

    // CPU staging path (GPU engine unavailable): the RG staging IS the texel
    // layout, whole volume in one buffer-to-image copy.
    void upload_fallback_volume(const std::vector<float>& psi_staging) {
        const std::uint32_t nx = static_cast<std::uint32_t>(grid_.x.n);
        const std::uint32_t ny = static_cast<std::uint32_t>(grid_.y.n);
        const std::uint32_t nz = static_cast<std::uint32_t>(grid_.z.n);
        bool first = false;
        if (fallback_tex_.img == VK_NULL_HANDLE) {
            if (!ctx_->create_storage_image_3d(nx, ny, nz,
                                               VK_FORMAT_R32G32_SFLOAT,
                                               &fallback_tex_)) {
                return;
            }
            first = true;
        }
        upload_image(fallback_tex_.img, psi_staging.data(),
                     psi_staging.size() * sizeof(float), {nx, ny, nz}, first);
    }

    // Point the psi bindings (raymarch, particle advection, streakline color)
    // at the live volume (engine bridge or fallback).
    void point_volume_binding(VkImageView engine_view) {
        VkImageView view = engine_view;
        if (view == VK_NULL_HANDLE) {
            view = fallback_tex_.view;
        }
        if (view == VK_NULL_HANDLE || view == volume_bound_view_) {
            volume_bound_view_ = view;
            return;
        }
        arena_.write_sampled(*ctx_, volume_set_, 1, view, samp_3d_);
        arena_.write_sampled(*ctx_, particles_set_, 2, view, samp_3d_);
        arena_.write_sampled(*ctx_, flow_set_, 2, view, samp_3d_);
        arena_.write_sampled(*ctx_, occ_set_, 2, view, samp_3d_);
        arena_.write_sampled(*ctx_, shadow_set_, 2, view, samp_3d_);
        aux_valid_ = false;  // fresh field: rebuild occupancy + shadow
        volume_bound_view_ = view;
    }
    // Re-point the advect set's velocity binding at the (flipping) engine
    // velocity volume each frame. Cached to skip redundant writes.
    void point_flow_velocity_binding(VkImageView engine_view) {
        if (engine_view == VK_NULL_HANDLE ||
            engine_view == flow_vel_bound_view_) {
            return;
        }
        arena_.write_sampled(*ctx_, particles_set_, 3, engine_view, samp_3d_);
        flow_vel_bound_view_ = engine_view;
    }

    // Grow-only host-visible mesh vertex buffer (isosurface refresh).
    void upload_mesh(const ses::Mesh& mesh,
                     const std::vector<ses::Rgb>& colors) {
        const std::vector<float> iv = interleave_mesh(mesh, nullptr, &colors);
        mesh_vertex_count_ = static_cast<int>(mesh.vertices.size());
        const VkDeviceSize bytes = iv.size() * sizeof(float);
        if (bytes == 0) {
            return;
        }
        if (mesh_vbuf_.buf == VK_NULL_HANDLE || mesh_vbuf_bytes_ < bytes) {
            vkDeviceWaitIdle(ctx_->device);
            ctx_->destroy_buffer(&mesh_vbuf_);
            if (!ctx_->create_host_buffer(
                    bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &mesh_vbuf_)) {
                mesh_vertex_count_ = 0;
                return;
            }
            mesh_vbuf_bytes_ = bytes;
        }
        std::memcpy(mesh_vbuf_.mapped, iv.data(), bytes);
        vmaFlushAllocation(ctx_->allocator, mesh_vbuf_.alloc, 0,
                           VK_WHOLE_SIZE);
    }

    // Per-frame camera math.
    void update_ubos(const FrameInput& in) {
        const double aspect = static_cast<double>(width_) /
                              std::max(1u, height_);
        const double bx =
            std::max(std::abs(grid_.x.xmin), std::abs(grid_.x.xmax));
        const double by =
            std::max(std::abs(grid_.y.xmin), std::abs(grid_.y.xmax));
        const double bz =
            std::max(std::abs(grid_.z.xmin), std::abs(grid_.z.xmax));
        const double box_reach = std::sqrt(bx * bx + by * by + bz * bz);
        const double zfar = in.distance + box_reach + 1.0;
        const double kPi = 3.14159265358979323846;
        const ses::Mat4 proj =
            ses::perspective(45.0 * kPi / 180.0, aspect, 0.1, zfar);
        const ses::Vec3d eye =
            ses::orbit_eye(in.azimuth, in.elevation, in.distance, ses::Vec3d{});
        const ses::Mat4 view =
            ses::look_at(eye, ses::Vec3d{}, ses::Vec3d{0.0, 1.0, 0.0});
        const ses::Mat4 mvp = proj * view;

        MeshUbo scene_u{};
        store_corrected_mvp(mvp, scene_u.mvp);
        scene_u.eye[0] = static_cast<float>(eye.x);
        scene_u.eye[1] = static_cast<float>(eye.y);
        scene_u.eye[2] = static_cast<float>(eye.z);
        std::memcpy(scene_ubuf_.mapped, &scene_u, sizeof(scene_u));

        const double d_g = 3.2;
        const ses::Vec3d geye =
            ses::orbit_eye(in.azimuth, in.elevation, d_g, ses::Vec3d{});
        const ses::Mat4 gproj =
            ses::perspective(45.0 * kPi / 180.0, 1.0, 0.1, 50.0);
        const ses::Mat4 gview =
            ses::look_at(geye, ses::Vec3d{}, ses::Vec3d{0.0, 1.0, 0.0});
        const ses::Mat4 gmvp = gproj * gview;
        MeshUbo gizmo_u{};
        store_corrected_mvp(gmvp, gizmo_u.mvp);
        gizmo_u.eye[0] = static_cast<float>(geye.x);
        gizmo_u.eye[1] = static_cast<float>(geye.y);
        gizmo_u.eye[2] = static_cast<float>(geye.z);
        std::memcpy(gizmo_ubuf_.mapped, &gizmo_u, sizeof(gizmo_u));

        const std::vector<float> zdata = build_z_label_verts(gview, geye);
        std::memcpy(zlabel_vbuf_.mapped, zdata.data(),
                    zdata.size() * sizeof(float));

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
        vol_u.proton_color[0] = 1.0f;
        vol_u.proton_color[1] = 0.45f;
        vol_u.proton_color[2] = 0.2f;
        vol_u.inv_peak =
            static_cast<float>(in.peak > 0.0 ? 1.0 / in.peak : 0.0);
        vol_u.absorbance = static_cast<float>(in.absorbance);
        vol_u.proton_radius = static_cast<float>(kProtonMarkerRadius);
        // Rotate the raymarch jitter ONLY while accumulating: a still frame
        // averages the rotating pattern into hundreds of effective samples,
        // while a moving/evolving scene keeps a STATIC dither (no shimmer --
        // the physics stays smooth on screen, as the smooth field it is).
        vol_u.jitter_frame = in.accumulate ? in.frame_index : 0.0f;
        vol_u.clip[0] = in.clip_on ? 1.0f : 0.0f;
        vol_u.clip[1] = static_cast<float>(in.clip_axis);
        vol_u.clip[2] = static_cast<float>(in.clip_sign);
        vol_u.clip[3] = in.clip_offset;
        vol_u.slice[0] = in.slice_on ? 1.0f : 0.0f;
        vol_u.slice[1] = static_cast<float>(in.slice_axis);
        vol_u.slice[2] = in.slice_offset;
        vol_u.slice[3] = static_cast<float>(in.slice_map);
        std::memcpy(volume_ubuf_.mapped, &vol_u, sizeof(vol_u));

        vmaFlushAllocation(ctx_->allocator, scene_ubuf_.alloc, 0,
                           VK_WHOLE_SIZE);
        vmaFlushAllocation(ctx_->allocator, gizmo_ubuf_.alloc, 0,
                           VK_WHOLE_SIZE);
        vmaFlushAllocation(ctx_->allocator, zlabel_vbuf_.alloc, 0,
                           VK_WHOLE_SIZE);
        vmaFlushAllocation(ctx_->allocator, volume_ubuf_.alloc, 0,
                           VK_WHOLE_SIZE);
    }

    // Billboarded "z" glyph just past the +Z arrow tip.
    static std::vector<float> build_z_label_verts(const ses::Mat4& view,
                                                  const ses::Vec3d& eye) {
        const ses::Vec3d right{view.m[0], view.m[4], view.m[8]};
        const ses::Vec3d up{view.m[1], view.m[5], view.m[9]};
        const ses::Vec3d center{0.0, 0.0, 1.18};
        const double s = 0.24;
        const ses::Vec3d nrm = normalized(eye - center);

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
        quad({-0.4, 0.5}, {0.4, 0.5}, {0.4, 0.34}, {-0.4, 0.34});
        quad({-0.4, -0.34}, {0.4, -0.34}, {0.4, -0.5}, {-0.4, -0.5});
        const double ax = 0.4, ay = 0.40, bx = -0.4, by = -0.40, ht = 0.11;
        const double dx = bx - ax, dy = by - ay;
        const double dl = std::sqrt(dx * dx + dy * dy);
        const double px = -dy / dl * ht, py = dx / dl * ht;
        quad({ax + px, ay + py}, {ax - px, ay - py}, {bx - px, by - py},
             {bx + px, by + py});

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

    DeviceContext* ctx_ = nullptr;
    ses::Grid3D grid_{};
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;

    DeviceContext::Image color_{};  // HDR scene (RGBA16F)
    DeviceContext::Image depth_{};
    // Post chain: temporal accumulation, bloom pyramid, tonemapped present.
    DeviceContext::Image accum_{};
    DeviceContext::Image bloom_[3]{};
    VkExtent2D bloom_size_[3]{};
    DeviceContext::Image present_{};  // RGBA8, what the shell samples
    VkImageLayout present_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    bool force_accum_reset_ = true;
    Kernel accum_k_;
    Kernel down_k_;
    Kernel up_k_;
    Kernel compose_k_;
    VkDescriptorSet accum_set_ = VK_NULL_HANDLE;
    VkDescriptorSet down_set_[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE,
                                    VK_NULL_HANDLE};
    VkDescriptorSet up_set_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet compose_set_ = VK_NULL_HANDLE;
    // Interactive-frame variants: read the scene color instead of accum_.
    VkDescriptorSet down0_color_set_ = VK_NULL_HANDLE;
    VkDescriptorSet compose_color_set_ = VK_NULL_HANDLE;
    Buffer accum_ubo_{};
    Buffer down_ubo_[3]{};
    Buffer compose_ubo_{};
    // Probability-flow particles.
    // Sparse streaklines: kFlowStreaks strips, each a kFlowTrail-vertex trail
    // of recent positions (weather-map Lagrangian flux look, low frame cost).
    static constexpr std::uint32_t kFlowStreaks = 1024;
    static constexpr std::uint32_t kFlowTrail = 40;
    struct alignas(16) FlowParams {
        float box_min[4];
        float box_max[4];
        float dt;
        float inv_peak;
        float lifetime;
        float frame;
        std::int32_t n_streaks;
        std::int32_t trail_len;
        std::int32_t animate;
        std::int32_t reset;
    };
    std::uint32_t flow_frame_ = 0;   // frames advected since enable (RNG + head)
    bool flow_was_on_ = false;       // enable-edge detect -> one reset dispatch
    // Occupancy skipping + self-shadow.
    Kernel occ_k_;
    Kernel dilate_k_;
    Kernel shadow_k_;
    DeviceContext::Image occ_a_{};
    DeviceContext::Image occ_b_{};
    DeviceContext::Image shadow_vol_{};
    VkDescriptorSet occ_set_ = VK_NULL_HANDLE;
    VkDescriptorSet dilate_set_ = VK_NULL_HANDLE;
    VkDescriptorSet shadow_set_ = VK_NULL_HANDLE;
    VkSampler samp_nearest_ = VK_NULL_HANDLE;
    bool aux_valid_ = false;
    // Converged-frame counter (mirrors accum.comp's sample cap).
    static constexpr int kAccumCap = 512;
    int accum_frames_ = 0;

    Kernel particles_k_;
    DslHolder flow_dsl_holder_;
    VkPipelineLayout flow_pl_ = VK_NULL_HANDLE;
    VkPipeline flow_pipe_ = VK_NULL_HANDLE;
    Buffer flow_buf_{};
    Buffer flow_ubo_{};
    VkDescriptorSet particles_set_ = VK_NULL_HANDLE;
    VkDescriptorSet flow_set_ = VK_NULL_HANDLE;

    DslHolder mesh_dsl_holder_;
    DslHolder vol_dsl_holder_;
    VkPipelineLayout mesh_pl_ = VK_NULL_HANDLE;
    VkPipelineLayout vol_pl_ = VK_NULL_HANDLE;
    VkPipeline mesh_pipe_ = VK_NULL_HANDLE;
    VkPipeline volume_pipe_ = VK_NULL_HANDLE;
    VkPipeline slice_pipe_ = VK_NULL_HANDLE;
    VkSampler samp_3d_ = VK_NULL_HANDLE;
    VkSampler samp_1d_ = VK_NULL_HANDLE;
    DescriptorArena arena_;
    VkDescriptorSet scene_set_ = VK_NULL_HANDLE;
    VkDescriptorSet gizmo_set_ = VK_NULL_HANDLE;
    VkDescriptorSet volume_set_ = VK_NULL_HANDLE;
    VkImageView volume_bound_view_ = VK_NULL_HANDLE;
    VkImageView flow_vel_bound_view_ = VK_NULL_HANDLE;

    Buffer scene_ubuf_{};
    Buffer gizmo_ubuf_{};
    Buffer volume_ubuf_{};
    Buffer proton_vbuf_{};
    Buffer gizmo_vbuf_{};
    Buffer cube_vbuf_{};
    Buffer zlabel_vbuf_{};
    Buffer mesh_vbuf_{};
    Buffer staging_{};
    VkDeviceSize staging_bytes_ = 0;
    VkDeviceSize mesh_vbuf_bytes_ = 0;
    DeviceContext::Image phase_tex_{};
    DeviceContext::Image fallback_tex_{};
    int proton_vertex_count_ = 0;
    int gizmo_vertex_count_ = 0;
    int mesh_vertex_count_ = 0;
};

}  // namespace ses_vk
