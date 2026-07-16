module;
#include <volk.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <source_location>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>
export module ses.vk.present;
export import ses.vk.compute;


// The shell's OWN presentation layer, raw Vulkan: a swapchain over the SDL
// window's surface, one fullscreen-triangle pass sampling the SceneRenderer's
// finished image (which is handed off in
// SHADER_READ_ONLY_OPTIMAL with fragment-read barriers -- exactly this), and
// a UI-record callback riding the same pass (ImGui draws there). Frame model
// matches the renderer's synchronous style: one frame in flight, fence-waited.
// ses.vk GMF set, textually pre-claimed: volk.h supplies the VK_* macros
// (macros never cross module boundaries) and inoculates against GMF/textual
// redefinitions. No VMA here: the presenter never names vma*.


export namespace ses_shell {

class SwapchainPresenter {
public:
    // The surface is owned by the caller (created via SDL); everything else
    // here is owned by the presenter. The blit SPIR-V pair comes baked.
    bool init(ses_vk::DeviceContext& ctx, VkSurfaceKHR surface,
              const unsigned char* vert_spv, std::size_t vert_size,
              const unsigned char* frag_spv, std::size_t frag_size) {
        ctx_ = &ctx;
        surface_ = surface;

        VkSurfaceFormatKHR pick{};
        if (!pick_format(&pick)) {
            return false;
        }
        format_ = pick;

        if (!create_pipeline(vert_spv, vert_size, frag_spv, frag_size) ||
            !create_descriptors() || !create_sync() || !create_swapchain()) {
            return false;
        }
        return true;
    }

    void release() {
        if (ctx_ == nullptr) {
            return;
        }
        vkDeviceWaitIdle(ctx_->device);
        destroy_swapchain();
        for (VkSemaphore s : render_done_) {
            vkDestroySemaphore(ctx_->device, s, nullptr);
        }
        render_done_.clear();
        if (acquire_sem_ != VK_NULL_HANDLE) {
            vkDestroySemaphore(ctx_->device, acquire_sem_, nullptr);
            acquire_sem_ = VK_NULL_HANDLE;
        }
        if (fence_ != VK_NULL_HANDLE) {
            vkDestroyFence(ctx_->device, fence_, nullptr);
            fence_ = VK_NULL_HANDLE;
        }
        if (pool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(ctx_->device, pool_, nullptr);
            pool_ = VK_NULL_HANDLE;
        }
        if (desc_pool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(ctx_->device, desc_pool_, nullptr);
            desc_pool_ = VK_NULL_HANDLE;
        }
        if (set_layout_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(ctx_->device, set_layout_, nullptr);
            set_layout_ = VK_NULL_HANDLE;
        }
        if (sampler_ != VK_NULL_HANDLE) {
            vkDestroySampler(ctx_->device, sampler_, nullptr);
            sampler_ = VK_NULL_HANDLE;
        }
        if (pipe_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(ctx_->device, pipe_, nullptr);
            pipe_ = VK_NULL_HANDLE;
        }
        if (pipe_layout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(ctx_->device, pipe_layout_, nullptr);
            pipe_layout_ = VK_NULL_HANDLE;
        }
        ctx_ = nullptr;
    }

    // For ImGui_ImplVulkan_Init: the swapchain color format (ImGui draws into
    // the present pass via dynamic rendering) and the image counts.
    VkFormat color_format() const { return format_.format; }
    std::uint32_t min_image_count() const { return min_images_; }
    std::uint32_t image_count() const {
        return static_cast<std::uint32_t>(images_.size());
    }

    // SDL resize events land here; the swapchain rebuilds on the next frame.
    void request_resize() { resize_requested_ = true; }

    // Acquire the next swapchain image. Called EARLY (loop top, before the
    // frame's compute): the FIFO/vsync backpressure wait then overlaps the
    // sim batch and the scene render instead of trailing them as dead time.
    // Idempotent until present() consumes the image; returns false when
    // minimized / mid-rebuild (present() will then skip the frame).
    bool acquire() {
        if (acquired_) {
            return true;
        }
        if (swapchain_ == VK_NULL_HANDLE || resize_requested_) {
            resize_requested_ = false;
            if (!recreate_swapchain()) {
                return false;
            }
        }
        if (extent_.width == 0 || extent_.height == 0) {
            return false;  // minimized
        }
        VkResult r = vkAcquireNextImageKHR(ctx_->device, swapchain_, UINT64_MAX,
                                           acquire_sem_, VK_NULL_HANDLE,
                                           &acquired_idx_);
        if (r == VK_ERROR_OUT_OF_DATE_KHR) {
            resize_requested_ = true;
            return false;
        }
        if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
            return false;
        }
        acquired_ = true;
        return true;
    }

    // [clear pass: fullscreen blit of scene_view (if any) + the UI callback]
    // -> submit -> present, on the image acquire() got (acquiring here if the
    // shell did not). One frame in flight, fence-waited (the scene itself was
    // already rendered synchronously before this). Returns false when the
    // frame was skipped (minimized / mid-rebuild).
    bool present(VkImageView scene_view,
                 const std::function<void(VkCommandBuffer)>& record_ui) {
        if (!acquire()) {
            return false;
        }
        const std::uint32_t idx = acquired_idx_;
        acquired_ = false;  // consumed by this frame's submission

        if (scene_view != last_view_) {
            update_descriptor(scene_view);
        }

        vkResetCommandPool(ctx_->device, pool_, 0);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb_, &bi);

        // Dynamic rendering (Vulkan 1.3 idiom): no render pass / framebuffer.
        // The acquired image is in an undefined layout -> transition it to
        // COLOR_ATTACHMENT for the clear+blit, and to PRESENT_SRC afterward.
        // The acquire semaphore (waited at COLOR_ATTACHMENT_OUTPUT on submit)
        // gates availability; render_done_ gates the present.
        ses_vk::image_layout_barrier(
            cb_, images_[idx], VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        VkRenderingAttachmentInfo color_att{};
        color_att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color_att.imageView = views_[idx];
        color_att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_att.clearValue.color = {{0.04f, 0.05f, 0.09f, 1.0f}};
        VkRenderingInfo rinfo{};
        rinfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        rinfo.renderArea = {{0, 0}, extent_};
        rinfo.layerCount = 1;
        rinfo.colorAttachmentCount = 1;
        rinfo.pColorAttachments = &color_att;
        vkCmdBeginRendering(cb_, &rinfo);
        const VkViewport vp{0.0f, 0.0f, static_cast<float>(extent_.width),
                            static_cast<float>(extent_.height), 0.0f, 1.0f};
        const VkRect2D sc{{0, 0}, extent_};
        vkCmdSetViewport(cb_, 0, 1, &vp);
        vkCmdSetScissor(cb_, 0, 1, &sc);
        if (scene_view != VK_NULL_HANDLE) {
            vkCmdBindPipeline(cb_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_);
            vkCmdBindDescriptorSets(cb_, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipe_layout_, 0, 1, &desc_set_, 0, nullptr);
            vkCmdDraw(cb_, 3, 1, 0, 0);
        }
        if (record_ui) {
            record_ui(cb_);
        }
        vkCmdEndRendering(cb_);
        ses_vk::image_layout_barrier(
            cb_, images_[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0);
        vkEndCommandBuffer(cb_);

        const VkPipelineStageFlags wait_stage =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &acquire_sem_;
        si.pWaitDstStageMask = &wait_stage;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb_;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &render_done_[idx];
        vkResetFences(ctx_->device, 1, &fence_);
        const VkResult sr = vkQueueSubmit(ctx_->queue, 1, &si, fence_);
        if (sr != VK_SUCCESS) {
            std::fprintf(stderr, "vk: present-blit submit %s\n", ses_vk::vk_result_str(sr));
            ctx_->device_lost = true;
            return false;
        }
        const VkResult wr = vkWaitForFences(ctx_->device, 1, &fence_, VK_TRUE,
                                            10ull * 1000 * 1000 * 1000);
        if (wr != VK_SUCCESS) {
            std::fprintf(stderr, "vk: present-blit fence wait %s\n",
                         ses_vk::vk_result_str(wr));
            ctx_->device_lost = true;
            return false;
        }

        VkPresentInfoKHR pi{};
        pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &render_done_[idx];
        pi.swapchainCount = 1;
        pi.pSwapchains = &swapchain_;
        pi.pImageIndices = &idx;
        const VkResult r = vkQueuePresentKHR(ctx_->queue, &pi);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) {
            resize_requested_ = true;
        } else if (r != VK_SUCCESS) {
            std::fprintf(stderr, "vk: queue present %s\n", ses_vk::vk_result_str(r));
            ctx_->device_lost = true;
        }
        return true;
    }

private:
    bool pick_format(VkSurfaceFormatKHR* out) {
        std::uint32_t n = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_->phys_dev, surface_, &n,
                                             nullptr);
        if (n == 0) {
            return false;
        }
        std::vector<VkSurfaceFormatKHR> fmts(n);
        vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_->phys_dev, surface_, &n,
                                             fmts.data());
        for (const VkSurfaceFormatKHR& f : fmts) {
            // UNORM (not SRGB): the scene image is tonemapped RGBA8 UNORM and
            // the blit must be a bit-copy, not a second gamma encode.
            if ((f.format == VK_FORMAT_B8G8R8A8_UNORM ||
                 f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
                f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                *out = f;
                return true;
            }
        }
        *out = fmts[0];
        return true;
    }

    bool create_pipeline(const unsigned char* vert_spv, std::size_t vert_size,
                         const unsigned char* frag_spv, std::size_t frag_size) {
        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_NEAREST;
        sci.minFilter = VK_FILTER_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(ctx_->device, &sci, nullptr, &sampler_) !=
            VK_SUCCESS) {
            return false;
        }

        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 1;
        li.pBindings = &b;
        if (vkCreateDescriptorSetLayout(ctx_->device, &li, nullptr,
                                        &set_layout_) != VK_SUCCESS) {
            return false;
        }
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &set_layout_;
        if (vkCreatePipelineLayout(ctx_->device, &pli, nullptr,
                                   &pipe_layout_) != VK_SUCCESS) {
            return false;
        }

        auto make_module = [this](const unsigned char* code,
                                  std::size_t size) {
            VkShaderModuleCreateInfo mi{};
            mi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            mi.codeSize = size;
            mi.pCode = reinterpret_cast<const std::uint32_t*>(code);
            VkShaderModule m = VK_NULL_HANDLE;
            vkCreateShaderModule(ctx_->device, &mi, nullptr, &m);
            return m;
        };
        VkShaderModule vs = make_module(vert_spv, vert_size);
        VkShaderModule fs = make_module(frag_spv, frag_size);
        if (vs == VK_NULL_HANDLE || fs == VK_NULL_HANDLE) {
            return false;
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
        vin.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType =
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vps{};
        vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vps.viewportCount = 1;
        vps.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState blend{};
        blend.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &blend;
        const VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT,
                                       VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        ds.dynamicStateCount = 2;
        ds.pDynamicStates = dyn;

        // Dynamic rendering: the blit pipeline declares the swapchain color
        // FORMAT instead of referencing a render pass (Vulkan 1.3 idiom).
        VkPipelineRenderingCreateInfo prci{};
        prci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        prci.colorAttachmentCount = 1;
        prci.pColorAttachmentFormats = &format_.format;
        VkGraphicsPipelineCreateInfo gpi{};
        gpi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gpi.pNext = &prci;
        gpi.stageCount = 2;
        gpi.pStages = stages;
        gpi.pVertexInputState = &vin;
        gpi.pInputAssemblyState = &ia;
        gpi.pViewportState = &vps;
        gpi.pRasterizationState = &rs;
        gpi.pMultisampleState = &ms;
        gpi.pColorBlendState = &cb;
        gpi.pDynamicState = &ds;
        gpi.layout = pipe_layout_;
        gpi.renderPass = VK_NULL_HANDLE;  // dynamic rendering (see prci above)
        const bool ok = vkCreateGraphicsPipelines(ctx_->device, VK_NULL_HANDLE,
                                                  1, &gpi, nullptr, &pipe_) ==
                        VK_SUCCESS;
        vkDestroyShaderModule(ctx_->device, vs, nullptr);
        vkDestroyShaderModule(ctx_->device, fs, nullptr);
        return ok;
    }

    bool create_descriptors() {
        VkDescriptorPoolSize size{};
        size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        size.descriptorCount = 1;
        VkDescriptorPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.maxSets = 1;
        pi.poolSizeCount = 1;
        pi.pPoolSizes = &size;
        if (vkCreateDescriptorPool(ctx_->device, &pi, nullptr, &desc_pool_) !=
            VK_SUCCESS) {
            return false;
        }
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = desc_pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &set_layout_;
        return vkAllocateDescriptorSets(ctx_->device, &ai, &desc_set_) ==
               VK_SUCCESS;
    }

    void update_descriptor(VkImageView view) {
        if (view == VK_NULL_HANDLE) {
            return;
        }
        // One frame in flight and the fence was waited: the set is idle.
        VkDescriptorImageInfo ii{};
        ii.sampler = sampler_;
        ii.imageView = view;
        ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = desc_set_;
        w.dstBinding = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &ii;
        vkUpdateDescriptorSets(ctx_->device, 1, &w, 0, nullptr);
        last_view_ = view;
    }

    bool create_sync() {
        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (vkCreateSemaphore(ctx_->device, &si, nullptr, &acquire_sem_) !=
            VK_SUCCESS) {
            return false;
        }
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(ctx_->device, &fi, nullptr, &fence_) != VK_SUCCESS) {
            return false;
        }
        VkCommandPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pci.queueFamilyIndex = ctx_->queue_family;
        if (vkCreateCommandPool(ctx_->device, &pci, nullptr, &pool_) !=
            VK_SUCCESS) {
            return false;
        }
        VkCommandBufferAllocateInfo cai{};
        cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cai.commandPool = pool_;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        return vkAllocateCommandBuffers(ctx_->device, &cai, &cb_) ==
               VK_SUCCESS;
    }

    bool create_swapchain() {
        VkSurfaceCapabilitiesKHR caps{};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx_->phys_dev, surface_,
                                                  &caps);
        extent_ = caps.currentExtent;
        if (extent_.width == 0 || extent_.height == 0) {
            return true;  // minimized: present() will skip until resize
        }
        min_images_ = std::max<std::uint32_t>(2, caps.minImageCount);
        std::uint32_t count = min_images_;
        if (caps.maxImageCount > 0) {
            count = std::min(count, caps.maxImageCount);
        }
        VkSwapchainCreateInfoKHR sci{};
        sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        sci.surface = surface_;
        sci.minImageCount = count;
        sci.imageFormat = format_.format;
        sci.imageColorSpace = format_.colorSpace;
        sci.imageExtent = extent_;
        sci.imageArrayLayers = 1;
        sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        sci.preTransform = caps.currentTransform;
        sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;  // vsync, always available
        sci.clipped = VK_TRUE;
        if (vkCreateSwapchainKHR(ctx_->device, &sci, nullptr, &swapchain_) !=
            VK_SUCCESS) {
            return false;
        }
        std::uint32_t n = 0;
        vkGetSwapchainImagesKHR(ctx_->device, swapchain_, &n, nullptr);
        images_.resize(n);
        vkGetSwapchainImagesKHR(ctx_->device, swapchain_, &n, images_.data());
        views_.resize(n, VK_NULL_HANDLE);
        for (std::uint32_t i = 0; i < n; ++i) {
            VkImageViewCreateInfo vci{};
            vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image = images_[i];
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = format_.format;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            if (vkCreateImageView(ctx_->device, &vci, nullptr, &views_[i]) !=
                VK_SUCCESS) {
                return false;
            }
        }
        // Present waits ride a PER-IMAGE semaphore: a single reused one races
        // the presentation engine.
        if (render_done_.size() != n) {
            for (VkSemaphore s : render_done_) {
                vkDestroySemaphore(ctx_->device, s, nullptr);
            }
            render_done_.assign(n, VK_NULL_HANDLE);
            VkSemaphoreCreateInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            for (std::uint32_t i = 0; i < n; ++i) {
                if (vkCreateSemaphore(ctx_->device, &si, nullptr,
                                      &render_done_[i]) != VK_SUCCESS) {
                    return false;
                }
            }
        }
        return true;
    }

    void destroy_swapchain() {
        for (VkImageView v : views_) {
            if (v != VK_NULL_HANDLE) {
                vkDestroyImageView(ctx_->device, v, nullptr);
            }
        }
        views_.clear();
        images_.clear();
        if (swapchain_ != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(ctx_->device, swapchain_, nullptr);
            swapchain_ = VK_NULL_HANDLE;
        }
    }

    bool recreate_swapchain() {
        vkDeviceWaitIdle(ctx_->device);
        destroy_swapchain();
        return create_swapchain() && swapchain_ != VK_NULL_HANDLE;
    }

    ses_vk::DeviceContext* ctx_ = nullptr;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;  // owned by the SDL shell
    VkSurfaceFormatKHR format_{};
    VkExtent2D extent_{};
    std::uint32_t min_images_ = 2;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage> images_;
    std::vector<VkImageView> views_;
    VkPipeline pipe_ = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet desc_set_ = VK_NULL_HANDLE;
    VkImageView last_view_ = VK_NULL_HANDLE;
    VkSemaphore acquire_sem_ = VK_NULL_HANDLE;
    std::vector<VkSemaphore> render_done_;  // one per swapchain image
    VkFence fence_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cb_ = VK_NULL_HANDLE;
    std::uint32_t acquired_idx_ = 0;  // valid while acquired_
    bool acquired_ = false;
    bool resize_requested_ = false;
};

// --dump-frame verification: read the SceneRenderer's finished image (RGBA8,
// SHADER_READ_ONLY_OPTIMAL -- transitioned round-trip here) back to the host
// and write a bottom-up 24-bit BMP.
inline bool dump_scene_bmp(ses_vk::DeviceContext& ctx, VkImage img,
                           std::uint32_t w, std::uint32_t h,
                           const char* path) {
    if (img == VK_NULL_HANDLE || w == 0 || h == 0) {
        return false;
    }
    ses_vk::Buffer host{};
    const VkDeviceSize bytes = VkDeviceSize(w) * h * 4;
    if (!ctx.create_host_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                &host)) {
        return false;
    }
    {
        ses_vk::OneShot shot;
        if (!shot.begin(ctx)) {
            ctx.destroy_buffer(&host);
            return false;
        }
        // Round-trip: sample-optimal -> transfer-src, copy, back.
        ses_vk::image_layout_barrier(
            shot.cb(), img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {w, h, 1};
        vkCmdCopyImageToBuffer(shot.cb(), img,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, host.buf,
                               1, &region);
        ses_vk::image_layout_barrier(
            shot.cb(), img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        shot.submit_and_wait(ctx);
    }

    // Hand-rolled BMP: 54-byte header, BGR rows padded to 4 bytes, bottom-up.
    std::FILE* f = std::fopen(path, "wb");
    if (f == nullptr) {
        ctx.destroy_buffer(&host);
        return false;
    }
    const std::uint32_t row = (w * 3 + 3) & ~3u;
    const std::uint32_t data_size = row * h;
    const std::uint32_t file_size = 54 + data_size;
    unsigned char hdr[54] = {'B', 'M'};
    auto put32 = [&hdr](int off, std::uint32_t v) {
        hdr[off] = static_cast<unsigned char>(v);
        hdr[off + 1] = static_cast<unsigned char>(v >> 8);
        hdr[off + 2] = static_cast<unsigned char>(v >> 16);
        hdr[off + 3] = static_cast<unsigned char>(v >> 24);
    };
    put32(2, file_size);
    put32(10, 54);
    put32(14, 40);
    put32(18, w);
    put32(22, h);
    hdr[26] = 1;   // planes
    hdr[28] = 24;  // bpp
    put32(34, data_size);
    std::fwrite(hdr, 1, 54, f);
    const unsigned char* px = static_cast<const unsigned char*>(host.mapped);
    std::vector<unsigned char> line(row, 0);
    for (std::uint32_t y = 0; y < h; ++y) {
        const unsigned char* src = px + std::size_t(h - 1 - y) * w * 4;
        for (std::uint32_t x = 0; x < w; ++x) {
            line[3 * x + 0] = src[4 * x + 2];  // B
            line[3 * x + 1] = src[4 * x + 1];  // G
            line[3 * x + 2] = src[4 * x + 0];  // R
        }
        std::fwrite(line.data(), 1, row, f);
    }
    std::fclose(f);
    ctx.destroy_buffer(&host);
    return true;
}

}  // namespace ses_shell
