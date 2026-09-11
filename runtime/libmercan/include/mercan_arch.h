#ifndef MERCAN_ARCH_H
#define MERCAN_ARCH_H

#include "mercan.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MERCAN_ARCH_ABI_VERSION 1u

/*
 * Stable, read-only metadata view passed to architecture/tokenizer plugins.
 * Returned string pointers are valid only for the duration of the callback.
 */
typedef struct mercan_metadata_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    void * userdata;

    int (*has_key)(const struct mercan_metadata_v1 * metadata, const char * key);
    const char * (*get_string)(const struct mercan_metadata_v1 * metadata, const char * key);
    int64_t (*get_i64)(const struct mercan_metadata_v1 * metadata, const char * key, int64_t fallback);
    double (*get_f64)(const struct mercan_metadata_v1 * metadata, const char * key, double fallback);
} mercan_metadata_v1;

enum mercan_arch_flags_v1 {
    /* Graph/tensor execution is currently provided by the compiled Mercan backend. */
    MERCAN_ARCH_BACKEND_MANAGED_GRAPH = 1ull << 0,
    /* Architecture is bundled with the Mercan runtime rather than loaded externally. */
    MERCAN_ARCH_BUILTIN = 1ull << 1,
    /* Architecture exercises Mercan Graph ABI v1 primitives inside its backend graph. */
    MERCAN_ARCH_GRAPH_ABI_V1_PRIMITIVES = 1ull << 2,
    /* Architecture resolves model weights through Mercan Tensor ABI v1. */
    MERCAN_ARCH_TENSOR_ABI_V1 = 1ull << 3,
    /* Architecture consumes the opaque runtime-owned KV cache view ABI v1. */
    MERCAN_ARCH_KV_ABI_V1 = 1ull << 4,
};

typedef struct mercan_architecture_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    const char * name;
    const char * display_name;
    const char * default_tokenizer;
    uint64_t flags;

    /* Return >0 for a match, 0 for no match, <0 for malformed metadata. */
    int (*probe)(const mercan_metadata_v1 * metadata);

    /* Return 0 when valid. On error, optionally write a human-readable message. */
    int (*validate)(const mercan_metadata_v1 * metadata, char * error, size_t error_capacity);
} mercan_architecture_v1;

/* Minimum v1 descriptor prefix: identity/defaults/flags. Callbacks are optional. */
#define MERCAN_ARCHITECTURE_V1_BASE_SIZE ((size_t) offsetof(mercan_architecture_v1, probe))
#define MERCAN_ARCH_HAS_V1(desc, member) \
    ((desc) != NULL && (desc)->struct_size >= \
        (offsetof(mercan_architecture_v1, member) + sizeof((desc)->member)))

/* Registry API. The descriptor memory and strings must remain valid while registered. */
MERCAN_API int mercan_arch_register_v1(const mercan_architecture_v1 * architecture);
MERCAN_API const mercan_architecture_v1 * mercan_arch_find_v1(const char * name);
MERCAN_API size_t mercan_arch_count_v1(void);
MERCAN_API const mercan_architecture_v1 * mercan_arch_at_v1(size_t index);

#ifdef __cplusplus
}
#endif

#endif
