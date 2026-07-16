// The single VMA implementation TU for the whole ses_vk world. It used to be
// stamped per-binary (main.cpp / vkcheck_main.cpp defining VMA_IMPLEMENTATION
// before the textual vk_device.hpp chain); modules severed that textual path,
// so the implementation lives here, compiled once into ses_app_modules.
// Configuration mirrors ses.vk.device's GMF exactly: VMA rides volk's
// dynamically fetched pointers.
#include <volk.h>

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
