#include "mercan_arch.h"
#include "mercan_internal.hpp"

#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace {
std::mutex g_arch_mutex;
std::vector<std::unique_ptr<mercan_architecture_v1>> g_architectures;

bool valid_arch_descriptor(const mercan_architecture_v1 * arch) {
    return arch && arch->abi_version == MERCAN_ARCH_ABI_VERSION &&
           arch->struct_size >= sizeof(mercan_architecture_v1) &&
           arch->name && *arch->name;
}
}

extern "C" {

int mercan_arch_register_v1(const mercan_architecture_v1 * architecture) {
    if (!valid_arch_descriptor(architecture)) return -1;
    std::lock_guard<std::mutex> lock(g_arch_mutex);
    for (const auto & item : g_architectures) {
        if (std::strcmp(item->name, architecture->name) == 0) return 1;
    }
    g_architectures.emplace_back(std::make_unique<mercan_architecture_v1>(*architecture));
    return 0;
}

const mercan_architecture_v1 * mercan_arch_find_v1(const char * name) {
    mercan_ensure_builtin_plugins();
    if (!name || !*name) return nullptr;
    std::lock_guard<std::mutex> lock(g_arch_mutex);
    for (const auto & item : g_architectures) {
        if (std::strcmp(item->name, name) == 0) return item.get();
    }
    return nullptr;
}

size_t mercan_arch_count_v1(void) {
    mercan_ensure_builtin_plugins();
    std::lock_guard<std::mutex> lock(g_arch_mutex);
    return g_architectures.size();
}

const mercan_architecture_v1 * mercan_arch_at_v1(size_t index) {
    mercan_ensure_builtin_plugins();
    std::lock_guard<std::mutex> lock(g_arch_mutex);
    if (index >= g_architectures.size()) return nullptr;
    return g_architectures[index].get();
}

}
