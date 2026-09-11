#include "mercan_plugin.h"

#include <cstdio>
#include <cstring>

namespace {

int anka_probe(const mercan_metadata_v1 * metadata) {
    if (!metadata || !metadata->get_string) return -1;
    const char * arch = metadata->get_string(metadata, "general.architecture");
    return arch && std::strcmp(arch, "anka") == 0 ? 100 : 0;
}

int anka_byte_probe(const mercan_metadata_v1 * metadata) {
    const char * type = metadata && metadata->get_string ? metadata->get_string(metadata, "mercan.tokenizer.type") : nullptr;
    return type && std::strcmp(type, "anka-byte") == 0 ? 100 : 0;
}
int anka_byte_validate(const mercan_metadata_v1 *, char *, size_t) { return 0; }
void * anka_byte_create(const mercan_metadata_v1 *, char *, size_t) { return reinterpret_cast<void *>(1); }
void anka_byte_destroy(void *) {}
int32_t anka_byte_encode(void *, const char * text, size_t len, bool, bool, mercan_token * out, int32_t cap) {
    if (!text) return 0; if (cap < static_cast<int32_t>(len)) return -static_cast<int32_t>(len);
    for (size_t i=0;i<len;++i) out[i]=static_cast<unsigned char>(text[i]); return static_cast<int32_t>(len);
}
int32_t anka_byte_piece(void *, mercan_token token, char * out, int32_t cap, bool) {
    if (token < 0 || token > 255) return 0; if (cap < 1) return -1; out[0]=static_cast<char>(token); return 1;
}
const mercan_tokenizer_v1 ANKA_BYTE = {
    MERCAN_TOKENIZER_ABI_VERSION, sizeof(mercan_tokenizer_v1), "anka-byte", "Anka byte tokenizer", 0,
    anka_byte_probe, anka_byte_validate, anka_byte_create, anka_byte_destroy, anka_byte_encode, anka_byte_piece,
};

int anka_validate(const mercan_metadata_v1 * metadata, char * error, size_t error_capacity) {
    const char * arch = metadata && metadata->get_string ? metadata->get_string(metadata, "general.architecture") : nullptr;
    if (!arch || std::strcmp(arch, "anka") != 0) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "expected general.architecture=anka");
        return -1;
    }
    return 0;
}

int anka_build_graph(const mercan_arch_graph_invocation_v1 * invocation, char * error, size_t error_capacity) {
    auto fail = [&](const char * message) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "%s", message);
        return -1;
    };
    if (!invocation || invocation->abi_version != MERCAN_ARCH_GRAPH_INVOCATION_ABI_VERSION ||
        invocation->struct_size < MERCAN_ARCH_GRAPH_INVOCATION_V1_BASE_SIZE)
        return fail("invalid Anka graph invocation");
    mercan_graph_builder_v1 * builder = invocation->builder;
    if (!mercan_graph_builder_valid_v1(builder) || !builder->tensors ||
        !mercan_tensor_resolver_valid_v1(builder->tensors))
        return fail("Anka requires Graph ABI v1 with Tensor ABI v1 resolver");
    if (invocation->input_tokens == MERCAN_TENSOR_NONE_V1)
        return fail("Anka requires input token ids");

    const mercan_graph_api_v1 * graph = builder->api;
    const mercan_tensor_api_v1 * tensors = builder->tensors->api;
    if (!graph->get_rows || !MERCAN_GRAPH_API_HAS_V1(graph, matmul) ||
        !MERCAN_GRAPH_API_HAS_V1(graph, set_output) || !MERCAN_GRAPH_API_HAS_V1(graph, finalize))
        return fail("Anka requires get_rows, matmul, set_output and finalize");

    if (tensors->declare_tensor(builder->tensors, "token_embd.weight",
            MERCAN_TENSOR_REQUIRED_V1 | MERCAN_TENSOR_WEIGHT_V1) != 0 ||
        tensors->declare_tensor(builder->tensors, "output.weight",
            MERCAN_TENSOR_REQUIRED_V1 | MERCAN_TENSOR_WEIGHT_V1) != 0)
        return fail("Anka tensor declaration failed");

    const mercan_tensor_handle_v1 token_embd = tensors->require_tensor(builder->tensors, "token_embd.weight");
    const mercan_tensor_handle_v1 output = tensors->require_tensor(builder->tensors, "output.weight");
    if (token_embd == MERCAN_TENSOR_NONE_V1 || output == MERCAN_TENSOR_NONE_V1)
        return fail("Anka required weights are missing");

    const mercan_tensor_handle_v1 hidden = graph->get_rows(builder, token_embd, invocation->input_tokens);
    if (hidden == MERCAN_TENSOR_NONE_V1) return fail("Anka embedding lookup failed");
    const mercan_tensor_handle_v1 logits = graph->matmul(builder, output, hidden);
    if (logits == MERCAN_TENSOR_NONE_V1) return fail("Anka output projection failed");
    if (graph->set_output(builder, MERCAN_GRAPH_OUTPUT_EMBEDDING_V1, hidden) != 0 ||
        graph->set_output(builder, MERCAN_GRAPH_OUTPUT_LOGITS_V1, logits) != 0 ||
        graph->finalize(builder, logits) != 0)
        return fail("Anka graph finalization failed");
    return 0;
}

const mercan_architecture_v1 ANKA_ARCH = {
    MERCAN_ARCH_ABI_VERSION,
    sizeof(mercan_architecture_v1),
    "anka",
    "Anka external demo architecture",
    "anka-byte",
    MERCAN_ARCH_GRAPH_ABI_V1_PRIMITIVES | MERCAN_ARCH_TENSOR_ABI_V1 | MERCAN_ARCH_GRAPH_CALLBACK_V1,
    anka_probe,
    anka_validate,
    anka_build_graph,
};

int anka_init(const mercan_plugin_host_v1 * host, char * error, size_t error_capacity) {
    if (!host || host->abi_version != MERCAN_PLUGIN_ABI_VERSION ||
        host->struct_size < MERCAN_PLUGIN_HOST_V1_BASE_SIZE || !host->register_architecture) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "incompatible Mercan plugin host");
        return -1;
    }
    if (!host->register_tokenizer) { if (error && error_capacity) std::snprintf(error, error_capacity, "Mercan host has no tokenizer registry"); return -2; }
    const int trc = host->register_tokenizer(&ANKA_BYTE);
    if (trc != 0 && trc != 1) { if (error && error_capacity) std::snprintf(error, error_capacity, "tokenizer registration failed: %d", trc); return -3; }
    const int rc = host->register_architecture(&ANKA_ARCH);
    if (rc != 0 && rc != 1) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "architecture registration failed: %d", rc);
        return -2;
    }
    return 0;
}

const mercan_plugin_v1 ANKA_PLUGIN = {
    MERCAN_PLUGIN_ABI_VERSION,
    sizeof(mercan_plugin_v1),
    "mercan-arch-anka",
    "0.1.0",
    0,
    anka_init,
};

} // namespace

extern "C" MERCAN_PLUGIN_EXPORT const mercan_plugin_v1 * mercan_plugin_entry_v1(void) {
    return &ANKA_PLUGIN;
}
