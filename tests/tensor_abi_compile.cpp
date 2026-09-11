#include "mercan_graph.h"
#include "mercan_tensor.h"

#include <cstring>

struct fake_catalog {
    const char * name = "blk.0.attn_q.weight";
    mercan_tensor_handle_v1 value = 42;
};

static int declare_tensor(mercan_tensor_resolver_v1 *, const char * name, uint32_t flags) {
    return name && *name && (flags & MERCAN_TENSOR_REQUIRED_V1) ? 0 : -1;
}

static mercan_tensor_handle_v1 by_name(mercan_tensor_resolver_v1 * r, const char * name) {
    auto * cat = static_cast<fake_catalog *>(r->userdata);
    return cat && name && std::strcmp(cat->name, name) == 0 ? cat->value : MERCAN_TENSOR_NONE_V1;
}

static mercan_tensor_handle_v1 require_tensor(mercan_tensor_resolver_v1 * r, const char * name) {
    return by_name(r, name);
}

int main() {
    fake_catalog cat;
    mercan_tensor_api_v1 api{};
    api.abi_version = MERCAN_TENSOR_ABI_VERSION;
    api.struct_size = MERCAN_TENSOR_API_V1_BASE_SIZE;
    api.declare_tensor = declare_tensor;
    api.tensor_by_name = by_name;
    api.require_tensor = require_tensor;

    mercan_tensor_resolver_v1 resolver{};
    resolver.abi_version = MERCAN_TENSOR_ABI_VERSION;
    resolver.struct_size = sizeof(resolver);
    resolver.userdata = &cat;
    resolver.api = &api;

    mercan_graph_builder_v1 graph{};
    graph.abi_version = MERCAN_GRAPH_ABI_VERSION;
    graph.struct_size = sizeof(graph);
    graph.tensors = &resolver;

    if (!mercan_tensor_resolver_valid_v1(&resolver)) return 1;
    if (!MERCAN_GRAPH_BUILDER_HAS_V1(&graph, tensors)) return 2;
    if (api.declare_tensor(&resolver, cat.name, MERCAN_TENSOR_REQUIRED_V1 | MERCAN_TENSOR_WEIGHT_V1) != 0) return 3;
    if (api.require_tensor(&resolver, cat.name) != 42) return 4;
    if (api.tensor_by_name(&resolver, "missing") != MERCAN_TENSOR_NONE_V1) return 5;
    return 0;
}
