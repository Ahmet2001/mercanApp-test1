#include "mercan_plugin.h"

#include <cstdio>
#include <cstring>

namespace {

int anka_probe(const mercan_metadata_v1 * metadata) {
    if (!metadata || !metadata->get_string) return -1;
    const char * arch = metadata->get_string(metadata, "general.architecture");
    return arch && std::strcmp(arch, "anka") == 0 ? 100 : 0;
}

int anka_validate(const mercan_metadata_v1 * metadata, char * error, size_t error_capacity) {
    const char * arch = metadata && metadata->get_string ? metadata->get_string(metadata, "general.architecture") : nullptr;
    if (!arch || std::strcmp(arch, "anka") != 0) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "expected general.architecture=anka");
        return -1;
    }
    return 0;
}

const mercan_architecture_v1 ANKA_ARCH = {
    MERCAN_ARCH_ABI_VERSION,
    sizeof(mercan_architecture_v1),
    "anka",
    "Anka external demo architecture",
    "ndsurf004",
    0,
    anka_probe,
    anka_validate,
};

int anka_init(const mercan_plugin_host_v1 * host, char * error, size_t error_capacity) {
    if (!host || host->abi_version != MERCAN_PLUGIN_ABI_VERSION ||
        host->struct_size < MERCAN_PLUGIN_HOST_V1_BASE_SIZE || !host->register_architecture) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "incompatible Mercan plugin host");
        return -1;
    }
    const int rc = host->register_architecture(&ANKA_ARCH);
    if (rc != 0 && rc != 1) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "architecture registration failed: %d", rc);
        return -2;
    }
    return 0;
}

const mercan_plugin_v1 ANKA_PLUGIN = {
    MERCAN_PLUGIN_ABI_VERSION,
    sizeof(mercan_plugin_v1),
    "mercan-arch-anka",
    "0.1.0",
    0,
    anka_init,
};

} // namespace

extern "C" MERCAN_PLUGIN_EXPORT const mercan_plugin_v1 * mercan_plugin_entry_v1(void) {
    return &ANKA_PLUGIN;
}
