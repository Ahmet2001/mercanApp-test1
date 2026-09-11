#ifndef MERCAN_GRAPH_H
#define MERCAN_GRAPH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MERCAN_GRAPH_ABI_VERSION 1u

typedef uint64_t mercan_tensor_handle_v1;

#define MERCAN_TENSOR_NONE_V1 ((mercan_tensor_handle_v1) 0)

typedef struct mercan_graph_builder_v1 mercan_graph_builder_v1;

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
} mercan_graph_api_v1;

struct mercan_graph_builder_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    void * userdata;
    const mercan_graph_api_v1 * api;
};

static inline int mercan_graph_builder_valid_v1(const mercan_graph_builder_v1 * builder) {
    return builder && builder->abi_version == MERCAN_GRAPH_ABI_VERSION &&
           builder->struct_size >= sizeof(mercan_graph_builder_v1) &&
           builder->api && builder->api->abi_version == MERCAN_GRAPH_ABI_VERSION &&
           builder->api->struct_size >= sizeof(mercan_graph_api_v1);
}

#ifdef __cplusplus
}
#endif

#endif
