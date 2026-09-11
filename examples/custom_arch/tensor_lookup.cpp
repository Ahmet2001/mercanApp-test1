#include "mercan_graph.h"
#include "mercan_tensor.h"

#include <string>

// Third-party architecture code sees only stable names + opaque handles.
// It never includes ggml.h or llama-model.h.
static mercan_tensor_handle_v1 require_weight(
        mercan_graph_builder_v1 * graph,
        const char * name) {
    if (!graph || !MERCAN_GRAPH_BUILDER_HAS_V1(graph, tensors) ||
        !mercan_tensor_resolver_valid_v1(graph->tensors)) {
        return MERCAN_TENSOR_NONE_V1;
    }

    mercan_tensor_resolver_v1 * tensors = graph->tensors;
    if (tensors->api->declare_tensor(
            tensors, name, MERCAN_TENSOR_REQUIRED_V1 | MERCAN_TENSOR_WEIGHT_V1) != 0) {
        return MERCAN_TENSOR_NONE_V1;
    }
    return tensors->api->require_tensor(tensors, name);
}

int mercan_example_tensor_lookup(mercan_graph_builder_v1 * graph, int layer) {
    const std::string prefix = "blk." + std::to_string(layer);
    const auto q = require_weight(graph, (prefix + ".attn_q.weight").c_str());
    const auto k = require_weight(graph, (prefix + ".attn_k.weight").c_str());
    const auto v = require_weight(graph, (prefix + ".attn_v.weight").c_str());
    return q != MERCAN_TENSOR_NONE_V1 && k != MERCAN_TENSOR_NONE_V1 && v != MERCAN_TENSOR_NONE_V1 ? 0 : -1;
}
