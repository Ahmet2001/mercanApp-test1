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

    return 0;
}

const mercan_architecture_v1 NEDOLM_ARCH = {
    MERCAN_ARCH_ABI_VERSION,
    sizeof(mercan_architecture_v1),
    "nedolm",
    "NedoLM",
    "ndsurf004",
    MERCAN_ARCH_BACKEND_MANAGED_GRAPH | MERCAN_ARCH_BUILTIN | MERCAN_ARCH_GRAPH_ABI_V1_PRIMITIVES | MERCAN_ARCH_TENSOR_ABI_V1,
    nedolm_probe,
    nedolm_validate,
};

}

void mercan_register_nedolm_architecture(void) {
    (void) mercan_arch_register_v1(&NEDOLM_ARCH);
}
