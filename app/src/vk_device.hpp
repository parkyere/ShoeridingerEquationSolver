#pragma once

// ses_vk: framework-free Vulkan bootstrap (M5 Stage 0).
//
// The seed of the framework-free compute core: a DeviceContext that OWNS a
// self-created VkInstance/VkDevice (headless path -- checks, future cluster
// runs), designed so a later stage can also ADOPT externally supplied handles
// (the QRhiWidget-provided device in the GUI, per the 2026-07-11 de-Qt
// analysis). No Qt anywhere in this header or its includes:
//   - volk is the loader: it defines VK_NO_PROTOTYPES itself, dlopens
//     vulkan-1, and declares canonically named global function pointers, so
//     downstream code (and later VkFFT) compiles and links unmodified.
//   - VMA owns device memory. Configured for dynamic function fetch so it
//     rides volk's pointers (VMA_STATIC_VULKAN_FUNCTIONS=0).
//
// Validation: when create(want_validation=true) finds VK_LAYER_KHRONOS_
// validation (vcpkg classic install or LunarG SDK; point VK_ADD_LAYER_PATH at
// the layer dir), the layer + a debug-utils messenger are enabled and every
// validation ERROR is counted in validation_errors -- harnesses fail on a
// nonzero count, which makes hand-authored barriers actually testable.

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
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace ses_vk {

// Validation-error tally, bumped by the debug-utils callback. Checked by
// harnesses after their GPU work: nonzero means a barrier/usage bug even if
// the numbers happened to come out right on this driver.
inline std::atomic<int> g_validation_errors{0};

inline VKAPI_ATTR VkBool32 VKAPI_CALL debug_utils_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*types*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void* /*user*/) {
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
        g_validation_errors.fetch_add(1, std::memory_order_relaxed);
    }
    std::fprintf(stderr, "[vk-validation] %s\n",
                 (data != nullptr && data->pMessage != nullptr) ? data->pMessage
                                                                : "(null)");
    return VK_FALSE;
}

enum class Boot {
    ok,
    no_driver,  // loader/instance/device unavailable -> harness SKIP (77)
    error,      // present but broken -> harness FAIL
};

// A VkBuffer plus its VMA allocation (and the persistent map, when host
// visible). Plain aggregate; DeviceContext::destroy_buffer releases it.
struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    void* mapped = nullptr;  // non-null only for host-visible buffers
};

struct DeviceContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkPhysicalDevice phys_dev = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
    bool validation_active = false;
    char device_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE] = {};

    DeviceContext() = default;
    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    ~DeviceContext() { destroy(); }

    // Self-create the whole chain (headless path). A later stage adds
    // adopt(instance, phys_dev, device, family, queue) for the GUI's
    // QRhi-owned device; that path must switch volkLoadDevice (single global
    // device table) to volkLoadDeviceTable so two devices can coexist.
    Boot create(bool want_validation) {
        if (volkInitialize() != VK_SUCCESS) {
            return Boot::no_driver;  // no vulkan-1 runtime on this machine
        }

        // Instance. Validation layer + debug-utils only when asked AND the
        // layer is actually discoverable (else proceed without, with a note).
        const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";
        std::vector<const char*> layers;
        std::vector<const char*> exts;
        if (want_validation) {
            std::uint32_t count = 0;
            vkEnumerateInstanceLayerProperties(&count, nullptr);
            std::vector<VkLayerProperties> props(count);
            vkEnumerateInstanceLayerProperties(&count, props.data());
            for (const VkLayerProperties& p : props) {
                if (std::strcmp(p.layerName, kValidationLayer) == 0) {
                    validation_active = true;
                    break;
                }
            }
            if (validation_active) {
                layers.push_back(kValidationLayer);
                exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            } else {
                std::fprintf(stderr,
                             "validation requested but %s not discoverable "
                             "(set VK_ADD_LAYER_PATH); continuing without\n",
                             kValidationLayer);
            }
        }

        VkApplicationInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        ai.pApplicationName = "sesolver";
        ai.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &ai;
        ici.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
        ici.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
        ici.enabledExtensionCount = static_cast<std::uint32_t>(exts.size());
        ici.ppEnabledExtensionNames = exts.empty() ? nullptr : exts.data();
        if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
            return Boot::no_driver;
        }
        volkLoadInstance(instance);

        if (validation_active) {
            VkDebugUtilsMessengerCreateInfoEXT mi{};
            mi.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            mi.messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            mi.messageType =
                VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            mi.pfnUserCallback = debug_utils_callback;
            if (vkCreateDebugUtilsMessengerEXT(instance, &mi, nullptr,
                                               &messenger) != VK_SUCCESS) {
                messenger = VK_NULL_HANDLE;  // non-fatal: layer still logs
            }
        }

        // Physical device: prefer the first discrete GPU, else the first
        // anything (mirrors what QRhi lands on for this single-GPU box).
        std::uint32_t dev_count = 0;
        vkEnumeratePhysicalDevices(instance, &dev_count, nullptr);
        if (dev_count == 0) {
            return Boot::no_driver;
        }
        std::vector<VkPhysicalDevice> devs(dev_count);
        vkEnumeratePhysicalDevices(instance, &dev_count, devs.data());
        phys_dev = devs[0];
        for (VkPhysicalDevice d : devs) {
            VkPhysicalDeviceProperties p{};
            vkGetPhysicalDeviceProperties(d, &p);
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                phys_dev = d;
                break;
            }
        }
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(phys_dev, &props);
        std::memcpy(device_name, props.deviceName, sizeof(device_name));

        // Queue family: first with compute. (Graphics too, when available,
        // to mirror the Qt-adopt path's combined family.)
        std::uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &qf_count, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qf_count);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &qf_count, qf.data());
        bool found = false;
        for (std::uint32_t i = 0; i < qf_count; ++i) {
            const VkQueueFlags flags = qf[i].queueFlags;
            if ((flags & VK_QUEUE_COMPUTE_BIT) != 0) {
                queue_family = i;
                found = true;
                if ((flags & VK_QUEUE_GRAPHICS_BIT) != 0) {
                    break;  // combined family preferred
                }
            }
        }
        if (!found) {
            return Boot::error;
        }

        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = queue_family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        if (vkCreateDevice(phys_dev, &dci, nullptr, &device) != VK_SUCCESS) {
            return Boot::error;
        }
        volkLoadDevice(device);
        vkGetDeviceQueue(device, queue_family, 0, &queue);

        // VMA rides volk's dynamically fetched entry points.
        VmaVulkanFunctions fns{};
        fns.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        fns.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
        VmaAllocatorCreateInfo aci{};
        aci.physicalDevice = phys_dev;
        aci.device = device;
        aci.instance = instance;
        aci.pVulkanFunctions = &fns;
        aci.vulkanApiVersion = VK_API_VERSION_1_1;
        if (vmaCreateAllocator(&aci, &allocator) != VK_SUCCESS) {
            return Boot::error;
        }
        return Boot::ok;
    }

    // Device-local buffer (storage + transfer both ways).
    bool create_device_buffer(VkDeviceSize size, Buffer* out) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo alc{};
        alc.usage = VMA_MEMORY_USAGE_AUTO;
        return vmaCreateBuffer(allocator, &bci, &alc, &out->buf, &out->alloc,
                               nullptr) == VK_SUCCESS;
    }

    // Host-visible persistently mapped buffer (staging or UBO).
    bool create_host_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                            Buffer* out) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = usage;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo alc{};
        alc.usage = VMA_MEMORY_USAGE_AUTO;
        alc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(allocator, &bci, &alc, &out->buf, &out->alloc,
                            &info) != VK_SUCCESS) {
            return false;
        }
        out->mapped = info.pMappedData;
        return out->mapped != nullptr;
    }

    void destroy_buffer(Buffer* b) {
        if (b->buf != VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator, b->buf, b->alloc);
            b->buf = VK_NULL_HANDLE;
            b->alloc = VK_NULL_HANDLE;
            b->mapped = nullptr;
        }
    }

    void destroy() {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
        }
        if (allocator != VK_NULL_HANDLE) {
            vmaDestroyAllocator(allocator);
            allocator = VK_NULL_HANDLE;
        }
        if (device != VK_NULL_HANDLE) {
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
        }
        if (messenger != VK_NULL_HANDLE) {
            vkDestroyDebugUtilsMessengerEXT(instance, messenger, nullptr);
            messenger = VK_NULL_HANDLE;
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }
    }
};

}  // namespace ses_vk
