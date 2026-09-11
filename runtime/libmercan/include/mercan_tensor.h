#ifndef MERCAN_TENSOR_H
#define MERCAN_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MERCAN_TENSOR_ABI_VERSION 1u

typedef uint64_t mercan_tensor_handle_v1;
#define MERCAN_TENSOR_NONE_V1 ((mercan_tensor_handle_v1) 0)

typedef struct mercan_tensor_resolver_v1 mercan_tensor_resolver_v1;

enum mercan_tensor_decl_flags_v1 {
    MERCAN_TENSOR_REQUIRED_V1 = 1u << 0,
    MERCAN_TENSOR_OPTIONAL_V1 = 1u << 1,
    MERCAN_TENSOR_WEIGHT_V1   = 1u << 2,
    MERCAN_TENSOR_STATE_V1    = 1u << 3,
};

typedef struct mercan_tensor_api_v1 {
    uint32_t abi_version;
    uint32_t struct_size;

    /* Declare the architecture-facing name and requirement for a tensor. */
    int (*declare_tensor)(mercan_tensor_resolver_v1 * resolver,
                          const char * name,
                          uint32_t flags);

    /* Non-failing lookup. Returns MERCAN_TENSOR_NONE_V1 when absent. */
    mercan_tensor_handle_v1 (*tensor_by_name)(mercan_tensor_resolver_v1 * resolver,
                                              const char * name);

    /* Required lookup. Returns NONE and stores a diagnostic when absent. */
    mercan_tensor_handle_v1 (*require_tensor)(mercan_tensor_resolver_v1 * resolver,
                                              const char * name);

    /* Append-only v1 extension fields. */
    int (*has_tensor)(mercan_tensor_resolver_v1 * resolver, const char * name);
    const char * (*last_error)(mercan_tensor_resolver_v1 * resolver);
    size_t (*declared_count)(mercan_tensor_resolver_v1 * resolver);
    const char * (*declared_name)(mercan_tensor_resolver_v1 * resolver, size_t index);
    uint32_t (*declared_flags)(mercan_tensor_resolver_v1 * resolver, size_t index);
} mercan_tensor_api_v1;

#define MERCAN_TENSOR_API_V1_BASE_SIZE \
    (offsetof(mercan_tensor_api_v1, require_tensor) + sizeof(((mercan_tensor_api_v1 *) 0)->require_tensor))

#define MERCAN_TENSOR_API_HAS_V1(api, member) \
    ((api) != NULL && \
     (api)->struct_size >= offsetof(mercan_tensor_api_v1, member) + sizeof(((mercan_tensor_api_v1 *) 0)->member) && \
     (api)->member != NULL)

struct mercan_tensor_resolver_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    void * userdata;
    const mercan_tensor_api_v1 * api;
};

static inline int mercan_tensor_resolver_valid_v1(const mercan_tensor_resolver_v1 * resolver) {
    return resolver && resolver->abi_version == MERCAN_TENSOR_ABI_VERSION &&
           resolver->struct_size >= sizeof(mercan_tensor_resolver_v1) &&
           resolver->api && resolver->api->abi_version == MERCAN_TENSOR_ABI_VERSION &&
           resolver->api->struct_size >= MERCAN_TENSOR_API_V1_BASE_SIZE;
}

#ifdef __cplusplus
}
#endif

#endif
