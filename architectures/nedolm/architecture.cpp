#include "mercan_arch.h"
#include "mercan_internal.hpp"

#include <cstdio>
#include <cstring>

namespace {

int nedolm_probe(const mercan_metadata_v1 * metadata) {
    if (!metadata || !metadata->get_string) return -1;
    const char * arch = metadata->get_string(metadata, "general.architecture");
    if (!arch) return 0;
    return std::strcmp(arch, "nedolm") == 0 ? 100 : 0;
}

int nedolm_validate(const mercan_metadata_v1 * metadata, char * error, size_t error_capacity) {
    const char * arch = metadata && metadata->get_string
        ? metadata->get_string(metadata, "general.architecture") : nullptr;
    if (!arch || std::strcmp(arch, "nedolm") != 0) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "expected general.architecture=nedolm");
        return -1;
    }

    const char * format = metadata->get_string(metadata, "mercan.format");
    if (format && std::strcmp(format, "mercan") != 0) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "unsupported mercan.format=%s", format);
        return -2;
    }

    const int64_t runtime_abi = metadata->get_i64
        ? metadata->get_i64(metadata, "mercan.runtime_abi", 1) : 1;
    if (runtime_abi > 1) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "model requires Mercan runtime ABI %lld, runtime supports 1", static_cast<long long>(runtime_abi));
        return -3;
    }
    return 0;
}

const mercan_architecture_v1 NEDOLM_ARCH = {
    MERCAN_ARCH_ABI_VERSION,
    sizeof(mercan_architecture_v1),
    "nedolm",
    "NedoLM",
    "ndsurf004",
    MERCAN_ARCH_BACKEND_MANAGED_GRAPH | MERCAN_ARCH_BUILTIN,
    nedolm_probe,
    nedolm_validate,
};

}

void mercan_register_nedolm_architecture(void) {
    (void) mercan_arch_register_v1(&NEDOLM_ARCH);
}
