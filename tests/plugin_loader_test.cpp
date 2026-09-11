#include "mercan_arch.h"
#include "mercan_plugin.h"

#include <cassert>
#include <cstring>
#include <iostream>

namespace {
constexpr mercan_tensor_handle_v1 H_TOKENS = 7;
constexpr mercan_tensor_handle_v1 H_EMBD_W = 11;
constexpr mercan_tensor_handle_v1 H_OUT_W = 12;
constexpr mercan_tensor_handle_v1 H_HIDDEN = 21;
constexpr mercan_tensor_handle_v1 H_LOGITS = 22;
int g_declared = 0;
int g_outputs = 0;
bool g_finalized = false;

int t_declare(mercan_tensor_resolver_v1 *, const char * name, uint32_t flags) {
    assert(name);
    assert(flags & MERCAN_TENSOR_REQUIRED_V1);
    ++g_declared;
    return 0;
}
mercan_tensor_handle_v1 t_by_name(mercan_tensor_resolver_v1 *, const char * name) {
    if (std::strcmp(name, "token_embd.weight") == 0) return H_EMBD_W;
    if (std::strcmp(name, "output.weight") == 0) return H_OUT_W;
    return MERCAN_TENSOR_NONE_V1;
}
mercan_tensor_handle_v1 t_require(mercan_tensor_resolver_v1 * r, const char * name) { return t_by_name(r, name); }

mercan_tensor_handle_v1 g_get_rows(mercan_graph_builder_v1 *, mercan_tensor_handle_v1 src, mercan_tensor_handle_v1 idx) {
    assert(src == H_EMBD_W && idx == H_TOKENS);
    return H_HIDDEN;
}
mercan_tensor_handle_v1 g_matmul(mercan_graph_builder_v1 *, mercan_tensor_handle_v1 w, mercan_tensor_handle_v1 x) {
    assert(w == H_OUT_W && x == H_HIDDEN);
    return H_LOGITS;
}
int g_set_output(mercan_graph_builder_v1 *, mercan_graph_output_kind_v1 kind, mercan_tensor_handle_v1 h) {
    if (kind == MERCAN_GRAPH_OUTPUT_EMBEDDING_V1) assert(h == H_HIDDEN);
    else if (kind == MERCAN_GRAPH_OUTPUT_LOGITS_V1) assert(h == H_LOGITS);
    else assert(false);
    ++g_outputs;
    return 0;
}
int g_finalize(mercan_graph_builder_v1 *, mercan_tensor_handle_v1 h) {
    assert(h == H_LOGITS);
    g_finalized = true;
    return 0;
}
}

int main(int argc, char ** argv) {
    assert(argc == 2);
    const int rc = mercan_plugin_load_v1(argv[1]);
    if (rc != 0) {
        std::cerr << mercan_plugin_last_error_v1() << "\n";
        return 2;
    }
    assert(mercan_plugin_count_v1() >= 1);
    const mercan_architecture_v1 * anka = mercan_arch_find_v1("anka");
    assert(anka != nullptr);
    assert(std::strcmp(anka->display_name, "Anka external demo architecture") == 0);
    assert(anka->flags & MERCAN_ARCH_GRAPH_CALLBACK_V1);
    assert(MERCAN_ARCH_HAS_V1(anka, build_graph));
    assert(anka->build_graph != nullptr);

    mercan_tensor_api_v1 tensor_api{};
    tensor_api.abi_version = MERCAN_TENSOR_ABI_VERSION;
    tensor_api.struct_size = sizeof(tensor_api);
    tensor_api.declare_tensor = t_declare;
    tensor_api.tensor_by_name = t_by_name;
    tensor_api.require_tensor = t_require;
    mercan_tensor_resolver_v1 tensors{MERCAN_TENSOR_ABI_VERSION, sizeof(mercan_tensor_resolver_v1), nullptr, &tensor_api};

    mercan_graph_api_v1 graph_api{};
    graph_api.abi_version = MERCAN_GRAPH_ABI_VERSION;
    graph_api.struct_size = sizeof(graph_api);
    graph_api.get_rows = g_get_rows;
    graph_api.matmul = g_matmul;
    graph_api.set_output = g_set_output;
    graph_api.finalize = g_finalize;
    mercan_graph_builder_v1 builder{MERCAN_GRAPH_ABI_VERSION, sizeof(mercan_graph_builder_v1), nullptr, &graph_api, &tensors, nullptr};

    mercan_arch_graph_invocation_v1 invocation{};
    invocation.abi_version = MERCAN_ARCH_GRAPH_INVOCATION_ABI_VERSION;
    invocation.struct_size = sizeof(invocation);
    invocation.builder = &builder;
    invocation.input_tokens = H_TOKENS;
    invocation.n_tokens = 3;
    invocation.n_outputs = 3;
    char error[256] = {};
    assert(mercan_arch_build_graph_v1(anka, &invocation, error, sizeof(error)) == 0);
    assert(g_declared == 2);
    assert(g_outputs == 2);
    assert(g_finalized);

    assert(mercan_plugin_load_v1(argv[1]) == 1); // idempotent path load
    std::cout << "PLUGIN_GRAPH_CALLBACK_OK " << mercan_plugin_name_v1(0) << "\n";
    return 0;
}
