#include "mercan_graph.h"

// Minimal example of backend-independent graph code.  This file intentionally
// includes no llama.cpp or ggml headers.
extern "C" mercan_tensor_handle_v1 mercan_example_gate_branch(
        mercan_graph_builder_v1 * builder,
        mercan_tensor_handle_v1 branch,
        mercan_tensor_handle_v1 gate) {
    if (!mercan_graph_builder_valid_v1(builder)) return MERCAN_TENSOR_NONE_V1;
    if (!branch || !gate) return MERCAN_TENSOR_NONE_V1;
    return builder->api->mul(builder, branch, gate);
}
