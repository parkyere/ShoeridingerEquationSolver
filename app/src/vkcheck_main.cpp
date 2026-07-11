// sesolver_vkcheck: framework-free Vulkan verification harness (M5 Stage 0).
//
// The raw-Vulkan analog of sesolver_qrhicheck and the seed of the eventual
// framework-free compute core: NO Qt anywhere in this binary. volk loads the
// loader, VMA allocates, the kernels are the SAME Vulkan-GLSL sources the
// QRhi engine bakes with qsb -- here compiled offline by glslangValidator to
// plain SPIR-V and embedded as C arrays (tools/cmake/bin2h.cmake). Results
// compare against the SAME CPU double references the core tests pin, at the
// SAME tolerances as qrhicheck, so the two harnesses cross-check each other
// kernel by kernel as the M5 port proceeds.
//
// Validation layers: set SES_VK_VALIDATION=1 (and have the layer discoverable
// via VK_ADD_LAYER_PATH or the SDK registry). Any validation ERROR fails the
// run even if the numbers pass -- hand-authored barriers are load-bearing in
// a raw-Vulkan engine, and this is the tripwire that keeps them honest.
//
// Exit codes: 0 = all checks PASS, 1 = FAIL, 77 = SKIP (no Vulkan runtime /
// device on this machine; ctest maps 77 to SKIP).

#define VMA_IMPLEMENTATION
#include "vk_device.hpp"

#include <core/complex.hpp>

#include <phase_multiply_spv.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// Complex<double> -> interleaved rg32f, byte-identical to the qrhicheck /
// engine upload format so all harnesses see the same fp32 inputs.
std::vector<float> to_rg32f(const std::vector<ses::Complex<double>>& src) {
    std::vector<float> out(2 * src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        out[2 * i] = static_cast<float>(src[i].real());
        out[2 * i + 1] = static_cast<float>(src[i].imag());
    }
    return out;
}

// psi <- psi (complex-*) phase: same data and tolerance (1e-5) as
// qrhicheck's check_phase_multiply, run through hand-rolled Vulkan --
// descriptor set, pipeline, staging copies, and every barrier explicit.
bool check_phase_multiply(ses_vk::DeviceContext& ctx) {
    const std::size_t n = 4096;
    std::vector<ses::Complex<double>> psi_d(n);
    std::vector<ses::Complex<double>> phase_d(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        psi_d[i] = ses::Complex<double>{std::sin(0.37 * x) + 0.2,
                                        std::cos(1.13 * x) - 0.1};
        phase_d[i] = ses::Complex<double>{std::cos(2.9 * x), std::sin(2.9 * x)};
    }
    const std::vector<float> psi_f = to_rg32f(psi_d);
    const std::vector<float> phase_f = to_rg32f(phase_d);
    const VkDeviceSize bytes = psi_f.size() * sizeof(float);

    bool pass = false;
    ses_vk::Buffer psi{};
    ses_vk::Buffer phase{};
    ses_vk::Buffer staging{};  // [0,bytes) psi up + result down; [bytes,2b) phase up
    ses_vk::Buffer ubo{};
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    // Single-exit cleanup keeps every failure path leak-free without RAII
    // scaffolding this early in the port (M5.1 factors real wrappers).
    const auto cleanup = [&] {
        if (fence != VK_NULL_HANDLE) vkDestroyFence(ctx.device, fence, nullptr);
        if (cmd_pool != VK_NULL_HANDLE)
            vkDestroyCommandPool(ctx.device, cmd_pool, nullptr);
        if (pool != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(ctx.device, pool, nullptr);
        if (pipe != VK_NULL_HANDLE) vkDestroyPipeline(ctx.device, pipe, nullptr);
        if (layout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(ctx.device, layout, nullptr);
        if (dsl != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(ctx.device, dsl, nullptr);
        if (shader != VK_NULL_HANDLE)
            vkDestroyShaderModule(ctx.device, shader, nullptr);
        ctx.destroy_buffer(&ubo);
        ctx.destroy_buffer(&staging);
        ctx.destroy_buffer(&phase);
        ctx.destroy_buffer(&psi);
    };

    struct alignas(16) Params {
        std::uint32_t n;
        std::uint32_t pad0, pad1, pad2;
    };

    do {
        if (!ctx.create_device_buffer(bytes, &psi) ||
            !ctx.create_device_buffer(bytes, &phase) ||
            !ctx.create_host_buffer(2 * bytes,
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    &staging) ||
            !ctx.create_host_buffer(sizeof(Params),
                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &ubo)) {
            std::fprintf(stderr, "buffer create failed\n");
            break;
        }
        std::memcpy(staging.mapped, psi_f.data(), bytes);
        std::memcpy(static_cast<char*>(staging.mapped) + bytes, phase_f.data(),
                    bytes);
        const Params params{static_cast<std::uint32_t>(n), 0, 0, 0};
        std::memcpy(ubo.mapped, &params, sizeof(params));
        vmaFlushAllocation(ctx.allocator, staging.alloc, 0, VK_WHOLE_SIZE);
        vmaFlushAllocation(ctx.allocator, ubo.alloc, 0, VK_WHOLE_SIZE);

        // Shader + pipeline from the embedded SPIR-V (bin2h aligns to 4).
        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = k_phase_multiply_spv_size;
        smci.pCode = reinterpret_cast<const std::uint32_t*>(k_phase_multiply_spv);
        if (vkCreateShaderModule(ctx.device, &smci, nullptr, &shader) !=
            VK_SUCCESS) {
            std::fprintf(stderr, "shader module create failed\n");
            break;
        }

        const VkDescriptorSetLayoutBinding bindings[3] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
             VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
             VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
             VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 3;
        dslci.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(ctx.device, &dslci, nullptr, &dsl) !=
            VK_SUCCESS) {
            std::fprintf(stderr, "descriptor set layout create failed\n");
            break;
        }
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsl;
        if (vkCreatePipelineLayout(ctx.device, &plci, nullptr, &layout) !=
            VK_SUCCESS) {
            std::fprintf(stderr, "pipeline layout create failed\n");
            break;
        }
        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = shader;
        cpci.stage.pName = "main";
        cpci.layout = layout;
        if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpci,
                                     nullptr, &pipe) != VK_SUCCESS) {
            std::fprintf(stderr, "compute pipeline create failed\n");
            break;
        }

        const VkDescriptorPoolSize sizes[2] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        };
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = 2;
        dpci.pPoolSizes = sizes;
        if (vkCreateDescriptorPool(ctx.device, &dpci, nullptr, &pool) !=
            VK_SUCCESS) {
            std::fprintf(stderr, "descriptor pool create failed\n");
            break;
        }
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &dsl;
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(ctx.device, &dsai, &set) != VK_SUCCESS) {
            std::fprintf(stderr, "descriptor set alloc failed\n");
            break;
        }
        const VkDescriptorBufferInfo psi_info{psi.buf, 0, VK_WHOLE_SIZE};
        const VkDescriptorBufferInfo phase_info{phase.buf, 0, VK_WHOLE_SIZE};
        const VkDescriptorBufferInfo ubo_info{ubo.buf, 0, sizeof(Params)};
        VkWriteDescriptorSet writes[3]{};
        for (int i = 0; i < 3; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = static_cast<std::uint32_t>(i);
            writes[i].descriptorCount = 1;
        }
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &psi_info;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &phase_info;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[2].pBufferInfo = &ubo_info;
        vkUpdateDescriptorSets(ctx.device, 3, writes, 0, nullptr);

        VkCommandPoolCreateInfo cpci2{};
        cpci2.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci2.queueFamilyIndex = ctx.queue_family;
        if (vkCreateCommandPool(ctx.device, &cpci2, nullptr, &cmd_pool) !=
            VK_SUCCESS) {
            std::fprintf(stderr, "command pool create failed\n");
            break;
        }
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = cmd_pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(ctx.device, &cbai, &cb) != VK_SUCCESS) {
            std::fprintf(stderr, "command buffer alloc failed\n");
            break;
        }

        // Record: upload copies -> transfer-to-compute barrier -> dispatch ->
        // compute-to-transfer barrier -> readback copy -> transfer-to-host
        // barrier. This is the barrier discipline QRhi supplied implicitly;
        // here it is the thing under test (validation layers police it).
        VkCommandBufferBeginInfo cbbi{};
        cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &cbbi);

        VkBufferCopy up0{0, 0, bytes};
        vkCmdCopyBuffer(cb, staging.buf, psi.buf, 1, &up0);
        VkBufferCopy up1{bytes, 0, bytes};
        vkCmdCopyBuffer(cb, staging.buf, phase.buf, 1, &up1);

        VkMemoryBarrier to_compute{};
        to_compute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        to_compute.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_compute.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &to_compute, 0, nullptr, 0, nullptr);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0,
                                1, &set, 0, nullptr);
        vkCmdDispatch(cb, static_cast<std::uint32_t>((n + 255) / 256), 1, 1);

        VkMemoryBarrier to_transfer{};
        to_transfer.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        to_transfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &to_transfer,
                             0, nullptr, 0, nullptr);

        VkBufferCopy down{0, 0, bytes};
        vkCmdCopyBuffer(cb, psi.buf, staging.buf, 1, &down);

        VkMemoryBarrier to_host{};
        to_host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        to_host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host, 0,
                             nullptr, 0, nullptr);
        vkEndCommandBuffer(cb);

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(ctx.device, &fci, nullptr, &fence) != VK_SUCCESS) {
            std::fprintf(stderr, "fence create failed\n");
            break;
        }
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        if (vkQueueSubmit(ctx.queue, 1, &si, fence) != VK_SUCCESS) {
            std::fprintf(stderr, "queue submit failed\n");
            break;
        }
        if (vkWaitForFences(ctx.device, 1, &fence, VK_TRUE,
                            5ull * 1000 * 1000 * 1000) != VK_SUCCESS) {
            std::fprintf(stderr, "fence wait failed/timed out\n");
            break;
        }
        vmaInvalidateAllocation(ctx.allocator, staging.alloc, 0, VK_WHOLE_SIZE);

        const float* out = static_cast<const float*>(staging.mapped);
        double max_err = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const ses::Complex<double> expected = psi_d[i] * phase_d[i];
            max_err = std::max(max_err, std::abs(out[2 * i] - expected.real()));
            max_err =
                std::max(max_err, std::abs(out[2 * i + 1] - expected.imag()));
        }
        pass = max_err < 1e-5;
        std::printf(
            "phase-multiply kernel (raw Vulkan): max |gpu - cpu| = %.3e  [%s]\n",
            max_err, pass ? "PASS" : "FAIL");
    } while (false);

    cleanup();
    return pass;
}

}  // namespace

int main() {
    const char* env = std::getenv("SES_VK_VALIDATION");
    const bool want_validation = (env != nullptr && env[0] == '1');

    ses_vk::DeviceContext ctx;
    const ses_vk::Boot boot = ctx.create(want_validation);
    if (boot == ses_vk::Boot::no_driver) {
        std::printf("vkcheck: no Vulkan runtime/device -- SKIP\n");
        return 77;
    }
    if (boot != ses_vk::Boot::ok) {
        std::fprintf(stderr, "vkcheck: device bootstrap failed\n");
        return 1;
    }
    std::printf("vkcheck: device '%s'%s\n", ctx.device_name,
                ctx.validation_active ? " [validation ON]" : "");

    int failures = 0;
    if (!check_phase_multiply(ctx)) {
        ++failures;
    }

    const int verrs = ses_vk::g_validation_errors.load();
    if (ctx.validation_active && verrs != 0) {
        std::fprintf(stderr, "vkcheck: %d validation error(s)  [FAIL]\n", verrs);
        ++failures;
    }
    if (failures == 0) {
        std::printf("vkcheck: all checks PASS\n");
        return 0;
    }
    std::fprintf(stderr, "vkcheck: %d check(s) FAILED\n", failures);
    return 1;
}
