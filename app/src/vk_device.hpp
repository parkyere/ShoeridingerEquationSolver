#pragma once

// ses_vk: framework-free Vulkan bootstrap.
//
// A DeviceContext either OWNS a self-created VkInstance/VkDevice (headless
// path -- checks, future cluster runs) or ADOPTS externally supplied handles
// (a host framework's device; the SDL3 shell here uses the create path).
// Framework-neutral -- volk + VMA only:
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
    // The engine's queue: a COMPUTE-only family when the hardware has one
    // (async compute -- physics overlaps the graphics queue's rendering),
    // else aliases the main queue/family (serial submission).
    // ALL engine submissions go here so engine resources never cross queue
    // families (EXCLUSIVE sharing stays legal); only the display volume is
    // created CONCURRENT across the two families.
    std::uint32_t compute_family = 0;
    VkQueue compute_queue = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
    bool validation_active = false;
    char device_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE] = {};

    DeviceContext() = default;
    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    ~DeviceContext() { destroy(); }

    // ADOPT externally supplied handles (a host framework's device, for a
    // shell that owns Vulkan itself; the SDL3 shell instead uses create_*):
    // the core code stays framework-free -- these are Khronos-standard
    // handles, dependency-injected. The context creates its OWN VmaAllocator
    // on the shared device (never touch the owner's) and destroys only what it
    // made. One device per process on this path (volkLoadDevice's global
    // table); a multi-device build would switch to volkLoadDeviceTable.
    Boot adopt(VkInstance inst, VkPhysicalDevice pd, VkDevice dev,
               std::uint32_t family, VkQueue q) {
        if (volkInitialize() != VK_SUCCESS) {
            return Boot::no_driver;
        }
        instance = inst;
        phys_dev = pd;
        device = dev;
        queue_family = family;
        queue = q;
        compute_family = family;  // adopted device: single shared queue
        compute_queue = q;
        owns_device_ = false;
        volkLoadInstance(instance);
        volkLoadDevice(device);
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(phys_dev, &props);
        std::memcpy(device_name, props.deviceName, sizeof(device_name));
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

    // Self-create the whole chain (headless path).
    Boot create(bool want_validation) {
        const Boot inst = create_instance(want_validation, {});
        return inst == Boot::ok ? create_device(VK_NULL_HANDLE) : inst;
    }

    // Instance half of the owning path. `extra_exts` carries the window
    // system's surface extensions (SDL_Vulkan_GetInstanceExtensions) -- the
    // GUI shell creates its VkSurfaceKHR between this and create_device().
    Boot create_instance(bool want_validation,
                         const std::vector<const char*>& extra_exts) {
        if (volkInitialize() != VK_SUCCESS) {
            return Boot::no_driver;  // no vulkan-1 runtime on this machine
        }

        // Validation layer + debug-utils only when asked AND the layer is
        // actually discoverable (else proceed without, with a note).
        const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";
        std::vector<const char*> layers;
        std::vector<const char*> exts{extra_exts};
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
        return Boot::ok;
    }

    // Device half of the owning path. A non-null `present_surface` makes
    // this a PRESENTING device: the queue family must support presenting to
    // it and VK_KHR_swapchain is enabled (the headless checks pass null).
    Boot create_device(VkSurfaceKHR present_surface) {
        // Physical device: prefer the first discrete GPU, else the first
        // anything.
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

        // Queue family: first with compute, preferring one that also has
        // graphics (the engine's image barriers use fragment stages) and --
        // when presenting -- one that can present to the surface. On desktop
        // hardware the combined graphics family presents, so the preferences
        // coincide.
        std::uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &qf_count, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qf_count);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &qf_count, qf.data());
        bool found = false;
        for (std::uint32_t i = 0; i < qf_count; ++i) {
            const VkQueueFlags flags = qf[i].queueFlags;
            if ((flags & VK_QUEUE_COMPUTE_BIT) == 0) {
                continue;
            }
            if (present_surface != VK_NULL_HANDLE) {
                VkBool32 can_present = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(phys_dev, i, present_surface,
                                                     &can_present);
                if (can_present != VK_TRUE) {
                    continue;
                }
            }
            queue_family = i;
            found = true;
            if ((flags & VK_QUEUE_GRAPHICS_BIT) != 0) {
                break;  // combined family preferred
            }
        }
        if (!found) {
            return Boot::error;
        }

        // Async-compute family: COMPUTE without GRAPHICS (NVIDIA exposes
        // one) so engine batches can overlap the graphics queue's rendering.
        // Absent one, the engine queue aliases the main queue (serial).
        compute_family = queue_family;
        for (std::uint32_t i = 0; i < qf_count; ++i) {
            const VkQueueFlags flags = qf[i].queueFlags;
            if ((flags & VK_QUEUE_COMPUTE_BIT) != 0 &&
                (flags & VK_QUEUE_GRAPHICS_BIT) == 0) {
                compute_family = i;
                break;
            }
        }

        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qcis[2] = {};
        qcis[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qcis[0].queueFamilyIndex = queue_family;
        qcis[0].queueCount = 1;
        qcis[0].pQueuePriorities = &prio;
        qcis[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qcis[1].queueFamilyIndex = compute_family;
        qcis[1].queueCount = 1;
        qcis[1].pQueuePriorities = &prio;
        const char* kSwapchainExt = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = compute_family != queue_family ? 2u : 1u;
        dci.pQueueCreateInfos = qcis;
        if (present_surface != VK_NULL_HANDLE) {
            dci.enabledExtensionCount = 1;
            dci.ppEnabledExtensionNames = &kSwapchainExt;
        }
        if (vkCreateDevice(phys_dev, &dci, nullptr, &device) != VK_SUCCESS) {
            return Boot::error;
        }
        volkLoadDevice(device);
        vkGetDeviceQueue(device, queue_family, 0, &queue);
        if (compute_family != queue_family) {
            vkGetDeviceQueue(device, compute_family, 0, &compute_queue);
        } else {
            compute_queue = queue;
        }

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

    // Device-local buffer (storage + transfer both ways); `extra` adds
    // consumer-specific usage (vertex / indirect for the GPU mesh path).
    bool create_device_buffer(VkDeviceSize size, Buffer* out,
                              VkBufferUsageFlags extra = 0,
                              bool share_across_queues = false) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | extra;
        // Written by the engine's compute queue, read on the graphics queue
        // (marching-cubes vertex/indirect buffers): CONCURRENT skips the
        // queue-family ownership transfer -- host fences already order the
        // accesses. EXCLUSIVE when the families coincide (no dedicated
        // compute queue) or the buffer never crosses.
        const std::uint32_t families[2] = {queue_family, compute_family};
        if (share_across_queues && compute_family != queue_family) {
            bci.sharingMode = VK_SHARING_MODE_CONCURRENT;
            bci.queueFamilyIndexCount = 2;
            bci.pQueueFamilyIndices = families;
        } else {
            bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }
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

    // A 3D storage image (+ its view). STORAGE for the compute bridge,
    // SAMPLED so the renderer can sample the same image.
    struct Image {
        VkImage img = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VmaAllocation alloc = VK_NULL_HANDLE;
    };

    bool create_storage_image_3d(std::uint32_t w, std::uint32_t h,
                                 std::uint32_t d, VkFormat format,
                                 Image* out,
                                 bool share_across_queues = false) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_3D;
        ici.format = format;
        ici.extent = {w, h, d};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        // Written by the engine's compute queue, sampled by the graphics
        // queue: CONCURRENT skips ownership-transfer barriers (host fences
        // already order the accesses).
        const std::uint32_t families[2] = {queue_family, compute_family};
        if (share_across_queues && compute_family != queue_family) {
            ici.sharingMode = VK_SHARING_MODE_CONCURRENT;
            ici.queueFamilyIndexCount = 2;
            ici.pQueueFamilyIndices = families;
        }
        VmaAllocationCreateInfo alc{};
        alc.usage = VMA_MEMORY_USAGE_AUTO;
        if (vmaCreateImage(allocator, &ici, &alc, &out->img, &out->alloc,
                           nullptr) != VK_SUCCESS) {
            return false;
        }
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = out->img;
        vci.viewType = VK_IMAGE_VIEW_TYPE_3D;
        vci.format = format;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(device, &vci, nullptr, &out->view) !=
            VK_SUCCESS) {
            vmaDestroyImage(allocator, out->img, out->alloc);
            out->img = VK_NULL_HANDLE;
            out->alloc = VK_NULL_HANDLE;
            return false;
        }
        return true;
    }

    void destroy_image(Image* im) {
        if (im->view != VK_NULL_HANDLE) {
            vkDestroyImageView(device, im->view, nullptr);
            im->view = VK_NULL_HANDLE;
        }
        if (im->img != VK_NULL_HANDLE) {
            vmaDestroyImage(allocator, im->img, im->alloc);
            im->img = VK_NULL_HANDLE;
            im->alloc = VK_NULL_HANDLE;
        }
    }

    void destroy() {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
        }
        if (oneshot_fence != VK_NULL_HANDLE) {
            vkDestroyFence(device, oneshot_fence, nullptr);
            oneshot_fence = VK_NULL_HANDLE;
        }
        if (oneshot_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, oneshot_pool, nullptr);
            oneshot_pool = VK_NULL_HANDLE;
            oneshot_cb = VK_NULL_HANDLE;
        }
        if (compute_oneshot_fence != VK_NULL_HANDLE) {
            vkDestroyFence(device, compute_oneshot_fence, nullptr);
            compute_oneshot_fence = VK_NULL_HANDLE;
        }
        if (compute_oneshot_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, compute_oneshot_pool, nullptr);
            compute_oneshot_pool = VK_NULL_HANDLE;
            compute_oneshot_cb = VK_NULL_HANDLE;
        }
        if (allocator != VK_NULL_HANDLE) {
            vmaDestroyAllocator(allocator);
            allocator = VK_NULL_HANDLE;
        }
        if (!owns_device_) {  // adopted handles belong to their owner
            device = VK_NULL_HANDLE;
            instance = VK_NULL_HANDLE;
            phys_dev = VK_NULL_HANDLE;
            queue = VK_NULL_HANDLE;
            return;
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

    // The OneShot scratch sets (vk_compute.hpp): one persistent transient
    // pool + primary cb + fence per queue, reset per submission. The main
    // set serves the renderer/presenter (graphics queue); the compute set
    // serves the engine (compute queue -- pools are per-family).
    VkCommandPool oneshot_pool = VK_NULL_HANDLE;
    VkCommandBuffer oneshot_cb = VK_NULL_HANDLE;
    VkFence oneshot_fence = VK_NULL_HANDLE;
    VkCommandPool compute_oneshot_pool = VK_NULL_HANDLE;
    VkCommandBuffer compute_oneshot_cb = VK_NULL_HANDLE;
    VkFence compute_oneshot_fence = VK_NULL_HANDLE;

private:
    bool owns_device_ = true;

public:
};

}  // namespace ses_vk
