#include "mercan_arch.h"
#include "mercan_tokenizer.h"

#include <cstdio>
#include <cstring>

static int hello_probe(const mercan_metadata_v1 * metadata) {
    const char * arch = metadata && metadata->get_string
        ? metadata->get_string(metadata, "general.architecture") : nullptr;
    return arch && std::strcmp(arch, "hello") == 0 ? 100 : 0;
}

static int hello_validate(const mercan_metadata_v1 * metadata, char * error, size_t error_capacity) {
    if (!hello_probe(metadata)) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "expected general.architecture=hello");
        return -1;
    }
    const int64_t layers = metadata->get_i64
        ? metadata->get_i64(metadata, "hello.block_count", -1) : -1;
    if (layers <= 0) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "hello.block_count must be positive");
        return -2;
    }
    return 0;
}

static const mercan_architecture_v1 HELLO_ARCH = {
    MERCAN_ARCH_ABI_VERSION,
    sizeof(mercan_architecture_v1),
    "hello",
    "Hello Architecture Example",
    nullptr,
    MERCAN_ARCH_BACKEND_MANAGED_GRAPH,
    hello_probe,
    hello_validate,
};

/*
 * In an in-tree integration, call this once during built-in registration.
 * A future external plugin loader will obtain the same descriptor from an
 * exported mercan_arch_init() entry point and keep the shared object loaded.
 */
extern "C" const mercan_architecture_v1 * mercan_arch_init(void) {
    return &HELLO_ARCH;
}
