#ifndef MERCAN_KV_H
#define MERCAN_KV_H

#include "mercan_tensor.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MERCAN_KV_ABI_VERSION 1u

typedef uint64_t mercan_kv_cache_handle_v1;
#define MERCAN_KV_CACHE_NONE_V1 ((mercan_kv_cache_handle_v1) 0)

typedef enum mercan_kv_cache_kind_v1 {
    MERCAN_KV_CACHE_KIND_UNKNOWN_V1 = 0,
    MERCAN_KV_CACHE_KIND_BASE_V1 = 1,
    MERCAN_KV_CACHE_KIND_SLIDING_WINDOW_V1 = 2,
} mercan_kv_cache_kind_v1;

typedef struct mercan_kv_resolver_v1 mercan_kv_resolver_v1;

typedef struct mercan_kv_api_v1 {
    uint32_t abi_version;
    uint32_t struct_size;

    mercan_kv_cache_handle_v1 (*cache_by_kind)(mercan_kv_resolver_v1 * resolver,
                                                mercan_kv_cache_kind_v1 kind);
    mercan_kv_cache_kind_v1 (*kind)(mercan_kv_resolver_v1 * resolver,
                                    mercan_kv_cache_handle_v1 cache);
    int64_t (*window_size)(mercan_kv_resolver_v1 * resolver,
                           mercan_kv_cache_handle_v1 cache);

    /* Runtime-generated per-batch cache placement/mask tensors. */
    mercan_tensor_handle_v1 (*k_indices)(mercan_kv_resolver_v1 * resolver,
                                         mercan_kv_cache_handle_v1 cache);
    mercan_tensor_handle_v1 (*v_indices)(mercan_kv_resolver_v1 * resolver,
                                         mercan_kv_cache_handle_v1 cache);
    mercan_tensor_handle_v1 (*attention_mask)(mercan_kv_resolver_v1 * resolver,
                                              mercan_kv_cache_handle_v1 cache);

    const char * (*last_error)(mercan_kv_resolver_v1 * resolver);
} mercan_kv_api_v1;

struct mercan_kv_resolver_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    void * userdata;
    const mercan_kv_api_v1 * api;
};

#define MERCAN_KV_API_V1_BASE_SIZE \
    (offsetof(mercan_kv_api_v1, last_error) + sizeof(((mercan_kv_api_v1 *)0)->last_error))
#define MERCAN_KV_RESOLVER_V1_BASE_SIZE sizeof(mercan_kv_resolver_v1)

static inline int mercan_kv_resolver_valid_v1(const mercan_kv_resolver_v1 * resolver) {
    return resolver && resolver->abi_version == MERCAN_KV_ABI_VERSION &&
           resolver->struct_size >= MERCAN_KV_RESOLVER_V1_BASE_SIZE &&
           resolver->api && resolver->api->abi_version == MERCAN_KV_ABI_VERSION &&
           resolver->api->struct_size >= MERCAN_KV_API_V1_BASE_SIZE;
}

#ifdef __cplusplus
}
#endif

#endif
