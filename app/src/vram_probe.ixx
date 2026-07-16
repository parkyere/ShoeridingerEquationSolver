module;
#include <volk.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdint>
#include <cstring>
#include <vector>
export module ses.app.vram_probe;
export import ses.vram_budget;


// Free VRAM in bytes via VK_EXT_memory_budget (heap budget minus current
// usage, summed over device-local heaps), or ses::kVramUnknown when the
// extension / entry points are absent. Pure volk -- call AFTER a
// DeviceContext create()/adopt() has run volkLoadInstance. Physical-device
// property queries only need the extension SUPPORTED, not enabled.
// ses.vk.device's GMF set, textually pre-claimed: volk.h both supplies the
// VK_* macros (macros never cross module boundaries) and inoculates the TU
// against GMF/textual std redefinitions.


export namespace ses_shell {

inline std::int64_t query_free_vram_bytes(VkPhysicalDevice pd) {
    if (pd == VK_NULL_HANDLE) {
        return ses::kVramUnknown;
    }
    uint32_t n_ext = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n_ext, nullptr);
    std::vector<VkExtensionProperties> exts(n_ext);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n_ext, exts.data());
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
    const PFN_vkGetPhysicalDeviceMemoryProperties2 get_props2 =
        vkGetPhysicalDeviceMemoryProperties2 != nullptr
            ? vkGetPhysicalDeviceMemoryProperties2
            : vkGetPhysicalDeviceMemoryProperties2KHR;
    if (get_props2 == nullptr) {
        return ses::kVramUnknown;
    }
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bud{};
    bud.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    VkPhysicalDeviceMemoryProperties2 props{};
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    props.pNext = &bud;
    get_props2(pd, &props);
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
