#pragma once

// ses_vk compute infrastructure: the thin, owned layer between raw Vulkan
// and the kernels -- descriptor/pipeline-layout construction, one-shot frame
// lifecycle, barrier placement. No Qt.
//
//   Kernel          shader module + set layout + pipeline layout + pipeline,
//                   built from an embedded SPIR-V blob and a binding spec.
//   DescriptorArena a per-run descriptor pool; allocates sets against a
//                   Kernel's layout and points bindings at buffers.
//   OneShot         command pool + primary command buffer + fence: record,
//                   submit, fence-wait.
//   barrier_*       the explicit hazard edges between recorded commands;
//                   every one is spelled out and the validation layer
//                   polices them.

#include "vk_device.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <vector>

namespace ses_vk {

// One descriptor binding of a compute kernel: index + type. Order in the
// kernel's spec list is irrelevant; indices match the shader's layout().
struct BindingDesc {
    std::uint32_t binding;
    VkDescriptorType type;
};

// Shader module + descriptor set layout + pipeline layout + compute pipeline.
class Kernel {
public:
    Kernel() = default;
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;

    bool create(DeviceContext& ctx, const unsigned char* spv,
                std::size_t spv_size, std::initializer_list<BindingDesc> spec) {
        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spv_size;
        smci.pCode = reinterpret_cast<const std::uint32_t*>(spv);
        if (vkCreateShaderModule(ctx.device, &smci, nullptr, &module_) !=
            VK_SUCCESS) {
            std::fprintf(stderr, "vk: shader module create failed\n");
            return false;
        }
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.reserve(spec.size());
        for (const BindingDesc& b : spec) {
            VkDescriptorSetLayoutBinding lb{};
            lb.binding = b.binding;
            lb.descriptorType = b.type;
            lb.descriptorCount = 1;
            lb.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings.push_back(lb);
        }
        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = static_cast<std::uint32_t>(bindings.size());
        dslci.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(ctx.device, &dslci, nullptr, &dsl_) !=
            VK_SUCCESS) {
            std::fprintf(stderr, "vk: descriptor set layout create failed\n");
            return false;
        }
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsl_;
        if (vkCreatePipelineLayout(ctx.device, &plci, nullptr, &layout_) !=
            VK_SUCCESS) {
            std::fprintf(stderr, "vk: pipeline layout create failed\n");
            return false;
        }
        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = module_;
        cpci.stage.pName = "main";
        cpci.layout = layout_;
        if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpci,
                                     nullptr, &pipe_) != VK_SUCCESS) {
            std::fprintf(stderr, "vk: compute pipeline create failed\n");
            return false;
        }
        return true;
    }

    void destroy(DeviceContext& ctx) {
        if (pipe_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx.device, pipe_, nullptr);
        if (layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(ctx.device, layout_, nullptr);
        if (dsl_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(ctx.device, dsl_, nullptr);
        if (module_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(ctx.device, module_, nullptr);
        pipe_ = VK_NULL_HANDLE;
        layout_ = VK_NULL_HANDLE;
        dsl_ = VK_NULL_HANDLE;
        module_ = VK_NULL_HANDLE;
    }

    VkDescriptorSetLayout set_layout() const { return dsl_; }
    VkPipelineLayout pipeline_layout() const { return layout_; }
    VkPipeline pipeline() const { return pipe_; }

    void bind(VkCommandBuffer cb, VkDescriptorSet set) const {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0,
                                1, &set, 0, nullptr);
    }

    // For kernels whose UBO binding is UNIFORM_BUFFER_DYNAMIC: bind with one
    // dynamic offset (a slot in a multi-slot parameter buffer).
    void bind(VkCommandBuffer cb, VkDescriptorSet set,
              std::uint32_t dynamic_offset) const {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0,
                                1, &set, 1, &dynamic_offset);
    }

private:
    VkShaderModule module_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipe_ = VK_NULL_HANDLE;
};

// A GROWABLE descriptor arena: allocation from the newest pool, and when it
// runs dry another pool of the same shape is chained (the resident-state
// atlas alone needs hundreds of sets). Sets are freed wholesale with the
// pools.
class DescriptorArena {
public:
    DescriptorArena() = default;
    DescriptorArena(const DescriptorArena&) = delete;
    DescriptorArena& operator=(const DescriptorArena&) = delete;

    bool create(DeviceContext& ctx, std::uint32_t max_sets,
                std::uint32_t storage_descs, std::uint32_t uniform_descs,
                std::uint32_t dynamic_uniform_descs = 0,
                std::uint32_t storage_images = 0,
                std::uint32_t combined_samplers = 0) {
        max_sets_ = max_sets;
        storage_ = storage_descs;
        uniform_ = uniform_descs;
        dynamic_ = dynamic_uniform_descs;
        images_ = storage_images;
        samplers_ = combined_samplers;
        return add_pool(ctx);
    }

    VkDescriptorSet allocate(DeviceContext& ctx, VkDescriptorSetLayout dsl) {
        VkDescriptorSet set = try_allocate(ctx, dsl);
        if (set == VK_NULL_HANDLE) {
            if (!add_pool(ctx)) {
                return VK_NULL_HANDLE;
            }
            set = try_allocate(ctx, dsl);
            if (set == VK_NULL_HANDLE) {
                std::fprintf(stderr, "vk: descriptor set alloc failed\n");
            }
        }
        return set;
    }

    void write_buffer(DeviceContext& ctx, VkDescriptorSet set,
                      std::uint32_t binding, VkDescriptorType type,
                      VkBuffer buf, VkDeviceSize range = VK_WHOLE_SIZE) {
        const VkDescriptorBufferInfo info{buf, 0, range};
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = set;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = type;
        w.pBufferInfo = &info;
        vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
    }

    void write_image(DeviceContext& ctx, VkDescriptorSet set,
                     std::uint32_t binding, VkImageView view,
                     VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL) {
        const VkDescriptorImageInfo info{VK_NULL_HANDLE, view, layout};
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = set;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.pImageInfo = &info;
        vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
    }

    void write_sampled(DeviceContext& ctx, VkDescriptorSet set,
                       std::uint32_t binding, VkImageView view,
                       VkSampler sampler,
                       VkImageLayout layout =
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        const VkDescriptorImageInfo info{sampler, view, layout};
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = set;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &info;
        vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
    }

    void destroy(DeviceContext& ctx) {
        for (VkDescriptorPool p : pools_) {
            vkDestroyDescriptorPool(ctx.device, p, nullptr);
        }
        pools_.clear();
    }

private:
    bool add_pool(DeviceContext& ctx) {
        VkDescriptorPoolSize sizes[5] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, storage_},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uniform_},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
             dynamic_ > 0 ? dynamic_ : 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, images_ > 0 ? images_ : 1},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
             samplers_ > 0 ? samplers_ : 1},
        };
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = max_sets_;
        dpci.poolSizeCount = 5;
        dpci.pPoolSizes = sizes;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        if (vkCreateDescriptorPool(ctx.device, &dpci, nullptr, &pool) !=
            VK_SUCCESS) {
            std::fprintf(stderr, "vk: descriptor pool create failed\n");
            return false;
        }
        pools_.push_back(pool);
        return true;
    }

    VkDescriptorSet try_allocate(DeviceContext& ctx,
                                 VkDescriptorSetLayout dsl) {
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = pools_.back();
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &dsl;
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(ctx.device, &dsai, &set) != VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }
        return set;
    }

    std::vector<VkDescriptorPool> pools_;
    std::uint32_t max_sets_ = 0;
    std::uint32_t storage_ = 0;
    std::uint32_t uniform_ = 0;
    std::uint32_t dynamic_ = 0;
    std::uint32_t images_ = 0;
    std::uint32_t samplers_ = 0;
};

// One-shot record/submit/wait: command pool + primary command buffer + fence.
class OneShot {
public:
    OneShot() = default;
    OneShot(const OneShot&) = delete;
    OneShot& operator=(const OneShot&) = delete;

    bool begin(DeviceContext& ctx) {
        VkCommandPoolCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.queueFamilyIndex = ctx.queue_family;
        if (vkCreateCommandPool(ctx.device, &cpci, nullptr, &pool_) !=
            VK_SUCCESS) {
            std::fprintf(stderr, "vk: command pool create failed\n");
            return false;
        }
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = pool_;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(ctx.device, &cbai, &cb_) != VK_SUCCESS) {
            std::fprintf(stderr, "vk: command buffer alloc failed\n");
            return false;
        }
        VkCommandBufferBeginInfo cbbi{};
        cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb_, &cbbi);
        return true;
    }

    VkCommandBuffer cb() const { return cb_; }

    bool submit_and_wait(DeviceContext& ctx) {
        vkEndCommandBuffer(cb_);
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(ctx.device, &fci, nullptr, &fence_) != VK_SUCCESS) {
            std::fprintf(stderr, "vk: fence create failed\n");
            return false;
        }
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb_;
        if (vkQueueSubmit(ctx.queue, 1, &si, fence_) != VK_SUCCESS) {
            std::fprintf(stderr, "vk: queue submit failed\n");
            return false;
        }
        if (vkWaitForFences(ctx.device, 1, &fence_, VK_TRUE,
                            10ull * 1000 * 1000 * 1000) != VK_SUCCESS) {
            std::fprintf(stderr, "vk: fence wait failed/timed out\n");
            return false;
        }
        return true;
    }

    void destroy(DeviceContext& ctx) {
        if (fence_ != VK_NULL_HANDLE) {
            vkDestroyFence(ctx.device, fence_, nullptr);
            fence_ = VK_NULL_HANDLE;
        }
        if (pool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(ctx.device, pool_, nullptr);
            pool_ = VK_NULL_HANDLE;
        }
        cb_ = VK_NULL_HANDLE;
    }

private:
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cb_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
};

// The hazard edges. Global memory barriers (not per-buffer) -- correct and
// simple; granularity can be revisited if profiling ever demands it.
inline void memory_barrier(VkCommandBuffer cb, VkPipelineStageFlags src_stage,
                           VkAccessFlags src_access,
                           VkPipelineStageFlags dst_stage,
                           VkAccessFlags dst_access) {
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = src_access;
    mb.dstAccessMask = dst_access;
    vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 1, &mb, 0, nullptr, 0,
                         nullptr);
}

inline void barrier_transfer_to_compute(VkCommandBuffer cb) {
    memory_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

// Read-after-write between dispatches sharing a buffer -- required between
// every aliasing dispatch of the step body.
inline void barrier_compute_to_compute(VkCommandBuffer cb) {
    memory_barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_ACCESS_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

inline void barrier_compute_to_transfer(VkCommandBuffer cb) {
    memory_barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_ACCESS_TRANSFER_READ_BIT);
}

inline void barrier_transfer_to_host(VkCommandBuffer cb) {
    memory_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                   VK_ACCESS_HOST_READ_BIT);
}

// Image layout transition (one mip, one layer, color aspect).
inline void image_layout_barrier(VkCommandBuffer cb, VkImage img,
                                 VkImageLayout from, VkImageLayout to,
                                 VkPipelineStageFlags src_stage,
                                 VkAccessFlags src_access,
                                 VkPipelineStageFlags dst_stage,
                                 VkAccessFlags dst_access) {
    VkImageMemoryBarrier ib{};
    ib.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    ib.srcAccessMask = src_access;
    ib.dstAccessMask = dst_access;
    ib.oldLayout = from;
    ib.newLayout = to;
    ib.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = img;
    ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr,
                         1, &ib);
}

}  // namespace ses_vk
