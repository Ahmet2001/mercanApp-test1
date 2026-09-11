#include "mercan_graph.h"

// Minimal examples of backend-independent graph code.  This file intentionally
// includes no llama.cpp or ggml headers.
extern "C" mercan_tensor_handle_v1 mercan_example_gate_branch(
        mercan_graph_builder_v1 * builder,
        mercan_tensor_handle_v1 branch,
        mercan_tensor_handle_v1 gate) {
    if (!mercan_graph_builder_valid_v1(builder)) return MERCAN_TENSOR_NONE_V1;
    if (!branch || !gate) return MERCAN_TENSOR_NONE_V1;
    return builder->api->mul(builder, branch, gate);
}

// Newer v1 operations are append-only.  Always probe them with struct_size
// before calling so a plugin can still load against an older v1 runtime.
extern "C" mercan_tensor_handle_v1 mercan_example_norm_linear(
        mercan_graph_builder_v1 * builder,
        mercan_tensor_handle_v1 input,
        mercan_tensor_handle_v1 norm_weight,
        mercan_tensor_handle_v1 linear_weight,
        float eps) {
    if (!mercan_graph_builder_valid_v1(builder)) return MERCAN_TENSOR_NONE_V1;
    const mercan_graph_api_v1 * api = builder->api;
    if (!MERCAN_GRAPH_API_HAS_V1(api, rms_norm) || !MERCAN_GRAPH_API_HAS_V1(api, matmul)) {
        return MERCAN_TENSOR_NONE_V1;
    }

    mercan_tensor_handle_v1 x = api->rms_norm(builder, input, eps);
    if (!x) return MERCAN_TENSOR_NONE_V1;
    x = api->mul(builder, x, norm_weight);
    if (!x) return MERCAN_TENSOR_NONE_V1;
    return api->matmul(builder, linear_weight, x);
}
