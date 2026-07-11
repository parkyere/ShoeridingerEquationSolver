#pragma once

// Free VRAM in bytes via Vulkan's VK_EXT_memory_budget (heap budget minus
// current usage, summed over device-local heaps), or ses::kVramUnknown when
// the extension / entry points are absent. Physical-device property queries
// only need the extension SUPPORTED, not enabled on the logical device.

// volk (inside vk_device.hpp) must own the vulkan.h inclusion before any Qt
// header pulls its own Vulkan integration.
#include "vk_device.hpp"

#include <core/vram_budget.hpp>

#include <QVulkanFunctions>
#include <QVulkanInstance>

#include <rhi/qrhi.h>
#include <rhi/qrhi_platform.h>  // QRhiVulkanNativeHandles

#include <cstdint>
#include <cstring>
#include <vector>

namespace ses_shell {

inline std::int64_t query_free_vram_bytes(QRhi* rhi) {
    const QRhiVulkanNativeHandles* h =
        static_cast<const QRhiVulkanNativeHandles*>(rhi->nativeHandles());
    if (h == nullptr || h->inst == nullptr || h->physDev == VK_NULL_HANDLE) {
        return ses::kVramUnknown;
    }
    QVulkanFunctions* f = h->inst->functions();
    uint32_t n_ext = 0;
    f->vkEnumerateDeviceExtensionProperties(h->physDev, nullptr, &n_ext, nullptr);
    std::vector<VkExtensionProperties> exts(n_ext);
    f->vkEnumerateDeviceExtensionProperties(h->physDev, nullptr, &n_ext, exts.data());
    bool budget = false;
    for (const VkExtensionProperties& e : exts) {
        if (std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
            budget = true;
            break;
        }
    }
    if (!budget) {
        return ses::kVramUnknown;
    }
    auto get_props2 = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2>(
        h->inst->getInstanceProcAddr("vkGetPhysicalDeviceMemoryProperties2"));
    if (get_props2 == nullptr) {
        get_props2 = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2>(
            h->inst->getInstanceProcAddr("vkGetPhysicalDeviceMemoryProperties2KHR"));
    }
    if (get_props2 == nullptr) {
        return ses::kVramUnknown;
    }
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bud{};
    bud.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    VkPhysicalDeviceMemoryProperties2 props{};
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    props.pNext = &bud;
    get_props2(h->physDev, &props);
    std::int64_t free_total = 0;
    bool any = false;
    for (uint32_t i = 0; i < props.memoryProperties.memoryHeapCount; ++i) {
        if ((props.memoryProperties.memoryHeaps[i].flags &
             VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) {
            continue;
        }
        if (bud.heapBudget[i] > bud.heapUsage[i]) {
            free_total += static_cast<std::int64_t>(bud.heapBudget[i] - bud.heapUsage[i]);
        }
        any = true;
    }
    return any ? free_total : ses::kVramUnknown;
}

}  // namespace ses_shell
