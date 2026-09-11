#include "mercan_plugin.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

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
    if (!text) return 0;
    if (cap < static_cast<int32_t>(len)) return -static_cast<int32_t>(len);
    for (size_t i = 0; i < len; ++i) out[i] = static_cast<unsigned char>(text[i]);
    return static_cast<int32_t>(len);
}
int32_t anka_byte_piece(void *, mercan_token token, char * out, int32_t cap, bool) {
    if (token < 0 || token > 255) return 0;
    if (cap < 1) return -1;
    out[0] = static_cast<char>(token);
    return 1;
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
    const int64_t blocks = metadata && metadata->get_i64 ? metadata->get_i64(metadata, "anka.block_count", 1) : 1;
    if (blocks <= 0 || blocks > 256) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "anka.block_count must be in [1, 256]");
        return -2;
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
    if (!MERCAN_GRAPH_BUILDER_HAS_V1(builder, kv) || !builder->kv || !mercan_kv_resolver_valid_v1(builder->kv))
        return fail("Anka transformer requires runtime-owned KV Cache ABI v1");
    if (invocation->input_tokens == MERCAN_TENSOR_NONE_V1 || invocation->input_positions == MERCAN_TENSOR_NONE_V1)
        return fail("Anka requires token and position inputs");

    const mercan_graph_api_v1 * graph = builder->api;
    const mercan_tensor_api_v1 * tensors = builder->tensors->api;
    const mercan_kv_api_v1 * kv = builder->kv->api;
    if (!graph->get_rows || !graph->mul || !graph->add || !MERCAN_GRAPH_API_HAS_V1(graph, matmul) ||
        !MERCAN_GRAPH_API_HAS_V1(graph, rms_norm) || !MERCAN_GRAPH_API_HAS_V1(graph, self_attention) ||
        !MERCAN_GRAPH_API_HAS_V1(graph, set_output) || !MERCAN_GRAPH_API_HAS_V1(graph, finalize) || !graph->swiglu_split)
        return fail("Anka requires transformer Graph ABI v1 primitives");

    const mercan_kv_cache_handle_v1 base_cache = kv->cache_by_kind(builder->kv, MERCAN_KV_CACHE_KIND_BASE_V1);
    if (base_cache == MERCAN_KV_CACHE_NONE_V1 || kv->k_indices(builder->kv, base_cache) == MERCAN_TENSOR_NONE_V1 ||
        kv->v_indices(builder->kv, base_cache) == MERCAN_TENSOR_NONE_V1 ||
        kv->attention_mask(builder->kv, base_cache) == MERCAN_TENSOR_NONE_V1)
        return fail("Anka requires runtime-generated KV placement and causal mask tensors");

    auto require = [&](const std::string & name) -> mercan_tensor_handle_v1 {
        if (tensors->declare_tensor(builder->tensors, name.c_str(), MERCAN_TENSOR_REQUIRED_V1 | MERCAN_TENSOR_WEIGHT_V1) != 0)
            return MERCAN_TENSOR_NONE_V1;
        return tensors->require_tensor(builder->tensors, name.c_str());
    };
    auto block_name = [](int64_t layer, const char * suffix) {
        return std::string("blk.") + std::to_string(layer) + "." + suffix;
    };

    const mercan_tensor_handle_v1 token_embd = require("token_embd.weight");
    const mercan_tensor_handle_v1 pos_embd = require("position_embd.weight");
    const mercan_tensor_handle_v1 output_norm = require("output_norm.weight");
    const mercan_tensor_handle_v1 output = require("output.weight");
    if (!token_embd || !pos_embd || !output_norm || !output) return fail("Anka top-level weights are missing");

    mercan_tensor_handle_v1 hidden = graph->get_rows(builder, token_embd, invocation->input_tokens);
    const mercan_tensor_handle_v1 pos = graph->get_rows(builder, pos_embd, invocation->input_positions);
    if (!hidden || !pos) return fail("Anka embedding lookup failed");
    hidden = graph->add(builder, hidden, pos);
    if (!hidden) return fail("Anka token/position embedding add failed");

    const int64_t blocks = invocation->metadata && invocation->metadata->get_i64
        ? invocation->metadata->get_i64(invocation->metadata, "anka.block_count", 1) : 1;
    const float eps = invocation->metadata && invocation->metadata->get_f64
        ? static_cast<float>(invocation->metadata->get_f64(invocation->metadata, "anka.rms_norm_eps", 1e-5)) : 1e-5f;
    float attn_scale = invocation->metadata && invocation->metadata->get_f64
        ? static_cast<float>(invocation->metadata->get_f64(invocation->metadata, "anka.attention.scale", 0.0)) : 0.0f;

    for (int64_t il = 0; il < blocks; ++il) {
        const auto attn_norm = require(block_name(il, "attn_norm.weight"));
        const auto wq = require(block_name(il, "attn_q.weight"));
        const auto wk = require(block_name(il, "attn_k.weight"));
        const auto wv = require(block_name(il, "attn_v.weight"));
        const auto wo = require(block_name(il, "attn_output.weight"));
        const auto ffn_norm = require(block_name(il, "ffn_norm.weight"));
        const auto gate_w = require(block_name(il, "ffn_gate.weight"));
        const auto up_w = require(block_name(il, "ffn_up.weight"));
        const auto down_w = require(block_name(il, "ffn_down.weight"));
        if (!attn_norm || !wq || !wk || !wv || !wo || !ffn_norm || !gate_w || !up_w || !down_w)
            return fail("Anka transformer block weights are missing");

        const mercan_tensor_handle_v1 residual = hidden;
        mercan_tensor_handle_v1 x = graph->rms_norm(builder, hidden, eps);
        x = graph->mul(builder, x, attn_norm);
        const mercan_tensor_handle_v1 q = graph->matmul(builder, wq, x);
        const mercan_tensor_handle_v1 k = graph->matmul(builder, wk, x);
        const mercan_tensor_handle_v1 v = graph->matmul(builder, wv, x);
        if (!q || !k || !v) return fail("Anka Q/K/V projection failed");
        if (attn_scale == 0.0f) {
            const int64_t d = graph->dim(builder, q, 0);
            attn_scale = d > 0 ? 1.0f / std::sqrt(static_cast<float>(d)) : 1.0f;
        }
        const mercan_tensor_handle_v1 attn = graph->self_attention(
            builder, q, k, v, wo, MERCAN_TENSOR_NONE_V1, MERCAN_TENSOR_NONE_V1, attn_scale, static_cast<int32_t>(il));
        if (!attn) return fail("Anka runtime-owned self-attention failed");
        hidden = graph->add(builder, residual, attn);
        if (!hidden) return fail("Anka attention residual failed");

        const mercan_tensor_handle_v1 ffn_residual = hidden;
        x = graph->rms_norm(builder, hidden, eps);
        x = graph->mul(builder, x, ffn_norm);
        const mercan_tensor_handle_v1 gate = graph->matmul(builder, gate_w, x);
        const mercan_tensor_handle_v1 up = graph->matmul(builder, up_w, x);
        const mercan_tensor_handle_v1 activated = graph->swiglu_split(builder, gate, up);
        const mercan_tensor_handle_v1 down = graph->matmul(builder, down_w, activated);
        if (!gate || !up || !activated || !down) return fail("Anka FFN failed");
        hidden = graph->add(builder, ffn_residual, down);
        if (!hidden) return fail("Anka FFN residual failed");
    }

    mercan_tensor_handle_v1 normalized = graph->rms_norm(builder, hidden, eps);
    normalized = graph->mul(builder, normalized, output_norm);
    if (!normalized) return fail("Anka output norm failed");
    const mercan_tensor_handle_v1 logits = graph->matmul(builder, output, normalized);
    if (!logits) return fail("Anka output projection failed");
    if (graph->set_output(builder, MERCAN_GRAPH_OUTPUT_EMBEDDING_V1, normalized) != 0 ||
        graph->set_output(builder, MERCAN_GRAPH_OUTPUT_LOGITS_V1, logits) != 0 ||
        graph->finalize(builder, logits) != 0)
        return fail("Anka graph finalization failed");
    return 0;
}

const mercan_architecture_v1 ANKA_ARCH = {
    MERCAN_ARCH_ABI_VERSION,
    sizeof(mercan_architecture_v1),
    "anka",
    "Anka external transformer demo architecture",
    "anka-byte",
    MERCAN_ARCH_GRAPH_ABI_V1_PRIMITIVES | MERCAN_ARCH_TENSOR_ABI_V1 | MERCAN_ARCH_KV_ABI_V1 | MERCAN_ARCH_GRAPH_CALLBACK_V1,
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
    if (!host->register_tokenizer) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "Mercan host has no tokenizer registry");
        return -2;
    }
    const int trc = host->register_tokenizer(&ANKA_BYTE);
    if (trc != 0 && trc != 1) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "tokenizer registration failed: %d", trc);
        return -3;
    }
    const int rc = host->register_architecture(&ANKA_ARCH);
    if (rc != 0 && rc != 1) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "architecture registration failed: %d", rc);
        return -4;
    }
    return 0;
}

const mercan_plugin_v1 ANKA_PLUGIN = {
    MERCAN_PLUGIN_ABI_VERSION,
    sizeof(mercan_plugin_v1),
    "mercan-arch-anka",
    "0.2.0",
    0,
    anka_init,
};

} // namespace

extern "C" MERCAN_PLUGIN_EXPORT const mercan_plugin_v1 * mercan_plugin_entry_v1(void) {
    return &ANKA_PLUGIN;
}
