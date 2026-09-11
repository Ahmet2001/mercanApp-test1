#ifndef MERCAN_GRAPH_H
#define MERCAN_GRAPH_H

#include "mercan_tensor.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MERCAN_GRAPH_ABI_VERSION 1u

typedef struct mercan_graph_builder_v1 mercan_graph_builder_v1;

typedef enum mercan_rope_mode_v1 {
    MERCAN_ROPE_MODE_UNSUPPORTED_V1 = -1,
    MERCAN_ROPE_MODE_NORMAL_V1 = 0,
    MERCAN_ROPE_MODE_NEOX_V1 = 1,
} mercan_rope_mode_v1;

typedef struct mercan_graph_api_v1 {
    uint32_t abi_version;
    uint32_t struct_size;

    int64_t (*dim)(mercan_graph_builder_v1 * builder, mercan_tensor_handle_v1 tensor, uint32_t axis);
    size_t (*stride_bytes)(mercan_graph_builder_v1 * builder, mercan_tensor_handle_v1 tensor, uint32_t axis);

    mercan_tensor_handle_v1 (*get_rows)(mercan_graph_builder_v1 * builder,
                                        mercan_tensor_handle_v1 src,
                                        mercan_tensor_handle_v1 indices);
    mercan_tensor_handle_v1 (*cast_f32)(mercan_graph_builder_v1 * builder,
                                        mercan_tensor_handle_v1 src);
    mercan_tensor_handle_v1 (*swiglu_split)(mercan_graph_builder_v1 * builder,
                                            mercan_tensor_handle_v1 gate,
                                            mercan_tensor_handle_v1 up);
    mercan_tensor_handle_v1 (*view_2d)(mercan_graph_builder_v1 * builder,
                                       mercan_tensor_handle_v1 src,
                                       int64_t ne0,
                                       int64_t ne1,
                                       size_t row_stride_bytes,
                                       size_t offset_bytes);
    mercan_tensor_handle_v1 (*mul)(mercan_graph_builder_v1 * builder,
                                   mercan_tensor_handle_v1 a,
                                   mercan_tensor_handle_v1 b);
    mercan_tensor_handle_v1 (*add)(mercan_graph_builder_v1 * builder,
                                   mercan_tensor_handle_v1 a,
                                   mercan_tensor_handle_v1 b);
    mercan_tensor_handle_v1 (*concat)(mercan_graph_builder_v1 * builder,
                                      mercan_tensor_handle_v1 a,
                                      mercan_tensor_handle_v1 b,
                                      int32_t dim);

    /* Append-only extension fields. Probe with MERCAN_GRAPH_API_HAS_V1(). */
    mercan_tensor_handle_v1 (*matmul)(mercan_graph_builder_v1 * builder,
                                      mercan_tensor_handle_v1 weight,
                                      mercan_tensor_handle_v1 input);
    mercan_tensor_handle_v1 (*rms_norm)(mercan_graph_builder_v1 * builder,
                                        mercan_tensor_handle_v1 src,
                                        float eps);
    mercan_tensor_handle_v1 (*rope_ext)(mercan_graph_builder_v1 * builder,
                                        mercan_tensor_handle_v1 src,
                                        mercan_tensor_handle_v1 positions,
                                        mercan_tensor_handle_v1 freq_factors,
                                        int32_t n_dims,
                                        mercan_rope_mode_v1 mode,
                                        int32_t n_ctx_orig,
                                        float freq_base,
                                        float freq_scale,
                                        float ext_factor,
                                        float attn_factor,
                                        float beta_fast,
                                        float beta_slow);

    /* Runtime-owned cached self-attention. Mask/KV-cache layout remains backend-private. */
    mercan_tensor_handle_v1 (*self_attention)(mercan_graph_builder_v1 * builder,
                                              mercan_tensor_handle_v1 q,
                                              mercan_tensor_handle_v1 k,
                                              mercan_tensor_handle_v1 v,
                                              mercan_tensor_handle_v1 out_weight,
                                              mercan_tensor_handle_v1 out_bias,
                                              mercan_tensor_handle_v1 out_scale,
                                              float kq_scale,
                                              int32_t layer_index);
} mercan_graph_api_v1;

/* The original Graph ABI v1 prefix ends at concat. Later v1 fields are append-only. */
#define MERCAN_GRAPH_API_V1_BASE_SIZE \
    (offsetof(mercan_graph_api_v1, concat) + sizeof(((mercan_graph_api_v1 *) 0)->concat))

#define MERCAN_GRAPH_API_HAS_V1(api, member) \
    ((api) != NULL && \
     (api)->struct_size >= offsetof(mercan_graph_api_v1, member) + sizeof(((mercan_graph_api_v1 *) 0)->member) && \
     (api)->member != NULL)

struct mercan_graph_builder_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    void * userdata;
    const mercan_graph_api_v1 * api;

    /* Append-only v1 extension. Runtime-owned tensor resolver for model weights/state. */
    mercan_tensor_resolver_v1 * tensors;
};

#define MERCAN_GRAPH_BUILDER_V1_BASE_SIZE \
    (offsetof(mercan_graph_builder_v1, api) + sizeof(((mercan_graph_builder_v1 *) 0)->api))

#define MERCAN_GRAPH_BUILDER_HAS_V1(builder, member) \
    ((builder) != NULL && \
     (builder)->struct_size >= offsetof(mercan_graph_builder_v1, member) + sizeof((builder)->member))

static inline int mercan_graph_builder_valid_v1(const mercan_graph_builder_v1 * builder) {
    return builder && builder->abi_version == MERCAN_GRAPH_ABI_VERSION &&
           builder->struct_size >= MERCAN_GRAPH_BUILDER_V1_BASE_SIZE &&
           builder->api && builder->api->abi_version == MERCAN_GRAPH_ABI_VERSION &&
           builder->api->struct_size >= MERCAN_GRAPH_API_V1_BASE_SIZE;
}

#ifdef __cplusplus
}
#endif

#endif
