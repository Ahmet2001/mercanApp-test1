#include "models.h"
#include "mercan_graph_ggml.hpp"

#include <string>

void llama_model_nedolm::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);

    ml.get_key("nedolm.morph.layer_count", morph_layer_count);
    ml.get_key("nedolm.morph.shared_width", morph_shared_width);
    ml.get_key("nedolm.morph.root_width", morph_root_width);
    ml.get_key("nedolm.morph.suffix_width", morph_suffix_width);

    if (hparams.n_swa == 0) {
        throw std::runtime_error("NedoLM requires a positive sliding-window length");
    }
    if (morph_layer_count > hparams.n_layer()) {
        throw std::runtime_error("nedolm.morph.layer_count exceeds transformer block count");
    }
    if (morph_shared_width == 0 || morph_root_width == 0 || morph_suffix_width == 0) {
        throw std::runtime_error("NedoLM MorphFFN widths must be positive");
    }
    const uint64_t morph_total = static_cast<uint64_t>(morph_shared_width) +
                                 static_cast<uint64_t>(morph_root_width) +
                                 static_cast<uint64_t>(morph_suffix_width);
    if (morph_total != static_cast<uint64_t>(hparams.n_ff())) {
        throw std::runtime_error("NedoLM MorphFFN widths do not sum to feed_forward_length");
    }

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    // period=0 means every transformer block is sliding-window attention.
    hparams.set_swa_pattern(0);

    hparams.n_embd_head_k_swa = hparams.n_embd_head_k_full;
    hparams.n_embd_head_v_swa = hparams.n_embd_head_v_full;
    hparams.n_rot_swa = hparams.n_rot_full;
    hparams.rope_freq_base_train_swa = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;

    type = LLM_TYPE_1B;
}

void llama_model_nedolm::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), {n_embd, n_ff}, 0);
    }

    // One shared 2 x vocab lookup table: [is_root, is_suffix] for every token id.
    layers[0].ffn_gate_tid2eid = create_tensor(
        tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", 0), {2, n_vocab}, 0);
}

std::unique_ptr<llm_graph_context> llama_model_nedolm::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_nedolm::graph::graph(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    const auto & nedolm = static_cast<const llama_model_nedolm &>(model);
    const int64_t morph_shared = static_cast<int64_t>(nedolm.morph_shared_width);
    const int64_t morph_root = static_cast<int64_t>(nedolm.morph_root_width);
    const int64_t morph_suffix = static_cast<int64_t>(nedolm.morph_suffix_width);

    // Mercan Tensor ABI v1 maps stable .mercan tensor names to opaque handles.
    // The architecture-facing graph below resolves weights by name instead of
    // reaching into llama_model/layer fields directly.  Only this private
    // backend binding block knows the ggml pointers.
    mercan_ggml_tensor_catalog_v1 mercan_tensor_catalog;
    auto block_tensor_name = [](int il, const char * suffix) {
        return std::string("blk.") + std::to_string(il) + "." + suffix;
    };
    auto bind_tensor = [&](const std::string & name, ggml_tensor * tensor) {
        GGML_ASSERT(mercan_ggml_tensor_catalog_bind_v1(&mercan_tensor_catalog, name.c_str(), tensor) == 0);
    };

    bind_tensor("token_embd.weight", model.tok_embd);
    bind_tensor("output_norm.weight", model.output_norm);
    bind_tensor("output.weight", model.output);
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        bind_tensor(block_tensor_name(il, "attn_norm.weight"), layer.attn_norm);
        bind_tensor(block_tensor_name(il, "attn_q.weight"), layer.wq);
        bind_tensor(block_tensor_name(il, "attn_k.weight"), layer.wk);
        bind_tensor(block_tensor_name(il, "attn_v.weight"), layer.wv);
        bind_tensor(block_tensor_name(il, "attn_output.weight"), layer.wo);
        bind_tensor(block_tensor_name(il, "ffn_norm.weight"), layer.ffn_norm);
        bind_tensor(block_tensor_name(il, "ffn_gate.weight"), layer.ffn_gate);
        bind_tensor(block_tensor_name(il, "ffn_up.weight"), layer.ffn_up);
        bind_tensor(block_tensor_name(il, "ffn_down.weight"), layer.ffn_down);
    }
    bind_tensor("blk.0.ffn_gate_tid2eid.weight", model.layers[0].ffn_gate_tid2eid);

    mercan_tensor_resolver_v1 mercan_tensors = mercan_make_ggml_tensor_resolver_v1(&mercan_tensor_catalog);
    GGML_ASSERT(mercan_tensor_resolver_valid_v1(&mercan_tensors));
    const mercan_tensor_api_v1 & mt = *mercan_tensors.api;

    auto declare_required = [&](const std::string & name) {
        GGML_ASSERT(mt.declare_tensor(&mercan_tensors, name.c_str(),
            MERCAN_TENSOR_REQUIRED_V1 | MERCAN_TENSOR_WEIGHT_V1) == 0);
    };
    declare_required("token_embd.weight");
    declare_required("output_norm.weight");
    declare_required("output.weight");
    for (int il = 0; il < n_layer; ++il) {
        declare_required(block_tensor_name(il, "attn_norm.weight"));
        declare_required(block_tensor_name(il, "attn_q.weight"));
        declare_required(block_tensor_name(il, "attn_k.weight"));
        declare_required(block_tensor_name(il, "attn_v.weight"));
        declare_required(block_tensor_name(il, "attn_output.weight"));
        declare_required(block_tensor_name(il, "ffn_norm.weight"));
        declare_required(block_tensor_name(il, "ffn_gate.weight"));
        declare_required(block_tensor_name(il, "ffn_up.weight"));
        declare_required(block_tensor_name(il, "ffn_down.weight"));
    }
    declare_required("blk.0.ffn_gate_tid2eid.weight");

    auto require_tensor_h = [&](const std::string & name) -> mercan_tensor_handle_v1 {
        const mercan_tensor_handle_v1 handle = mt.require_tensor(&mercan_tensors, name.c_str());
        GGML_ASSERT(handle != MERCAN_TENSOR_NONE_V1);
        return handle;
    };
    auto require_tensor = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * tensor = mercan_ggml_tensor_from_handle_v1(require_tensor_h(name));
        GGML_ASSERT(tensor != nullptr);
        return tensor;
    };

    mercan_ggml_graph_userdata_v1 mercan_graph_userdata{ctx0};
    mercan_graph_builder_v1 mercan_graph = mercan_make_ggml_graph_builder_v1(&mercan_graph_userdata, &mercan_tensors);
    GGML_ASSERT(mercan_graph_builder_valid_v1(&mercan_graph));
    GGML_ASSERT(MERCAN_GRAPH_BUILDER_HAS_V1(&mercan_graph, tensors));
    GGML_ASSERT(mercan_graph.tensors == &mercan_tensors);
    const mercan_graph_api_v1 & mg = *mercan_graph.api;
    GGML_ASSERT(MERCAN_GRAPH_API_HAS_V1(&mg, rms_norm));
    GGML_ASSERT(MERCAN_GRAPH_API_HAS_V1(&mg, rope_ext));
    GGML_ASSERT(MERCAN_GRAPH_API_HAS_V1(&mg, self_attention));

    const mercan_rope_mode_v1 mercan_rope_mode = mercan_ggml_rope_mode_to_mercan_v1(rope_type);
    GGML_ASSERT(mercan_rope_mode != MERCAN_ROPE_MODE_UNSUPPORTED_V1);

    auto graph_rms_norm_weight = [&](ggml_tensor * src, ggml_tensor * weight, int il) -> ggml_tensor * {
        mercan_tensor_handle_v1 h = mg.rms_norm(
            &mercan_graph, mercan_ggml_tensor_to_handle_v1(src), hparams.f_norm_rms_eps);
        ggml_tensor * norm = mercan_ggml_tensor_from_handle_v1(h);
        GGML_ASSERT(norm != nullptr);
        cb(norm, "norm", il);
        h = mg.mul(&mercan_graph, h, mercan_ggml_tensor_to_handle_v1(weight));
        ggml_tensor * out = mercan_ggml_tensor_from_handle_v1(h);
        GGML_ASSERT(out != nullptr);
        return out;
    };

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);
    GGML_ASSERT(hparams.n_ff() == morph_shared + morph_root + morph_suffix);

    ggml_tensor * inpL = build_inp_embd(require_tensor("token_embd.weight"));
    ggml_tensor * inp_tokens = res->t_inp_tokens;
    GGML_ASSERT(inp_tokens != nullptr);

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv_iswa();

    mercan_ggml_kv_view_v1 mercan_kv_view{inp_attn, static_cast<int64_t>(hparams.n_swa), {}};
    mercan_kv_resolver_v1 mercan_kv = mercan_make_ggml_kv_resolver_v1(&mercan_kv_view);
    GGML_ASSERT(mercan_kv_resolver_valid_v1(&mercan_kv));
    mercan_graph.kv = &mercan_kv;
    GGML_ASSERT(MERCAN_GRAPH_BUILDER_HAS_V1(&mercan_graph, kv));
    GGML_ASSERT(mercan_graph.kv == &mercan_kv);
    const mercan_kv_api_v1 & mkv = *mercan_kv.api;
    const mercan_kv_cache_handle_v1 base_cache = mkv.cache_by_kind(&mercan_kv, MERCAN_KV_CACHE_KIND_BASE_V1);
    const mercan_kv_cache_handle_v1 swa_cache = mkv.cache_by_kind(&mercan_kv, MERCAN_KV_CACHE_KIND_SLIDING_WINDOW_V1);
    GGML_ASSERT(base_cache != MERCAN_KV_CACHE_NONE_V1);
    GGML_ASSERT(swa_cache != MERCAN_KV_CACHE_NONE_V1);
    GGML_ASSERT(mkv.window_size(&mercan_kv, swa_cache) == static_cast<int64_t>(hparams.n_swa));
    GGML_ASSERT(mkv.k_indices(&mercan_kv, base_cache) != MERCAN_TENSOR_NONE_V1);
    GGML_ASSERT(mkv.v_indices(&mercan_kv, base_cache) != MERCAN_TENSOR_NONE_V1);
    GGML_ASSERT(mkv.attention_mask(&mercan_kv, base_cache) != MERCAN_TENSOR_NONE_V1);
    GGML_ASSERT(mkv.k_indices(&mercan_kv, swa_cache) != MERCAN_TENSOR_NONE_V1);
    GGML_ASSERT(mkv.v_indices(&mercan_kv, swa_cache) != MERCAN_TENSOR_NONE_V1);
    GGML_ASSERT(mkv.attention_mask(&mercan_kv, swa_cache) != MERCAN_TENSOR_NONE_V1);

    struct nedolm_attention_bridge_v1 {
        llama_model_nedolm::graph * graph;
        llm_graph_input_attn_kv_iswa * input;
    };
    nedolm_attention_bridge_v1 attention_bridge{this, inp_attn};
    mercan_graph_userdata.attention_userdata = &attention_bridge;
    mercan_graph_userdata.self_attention = [](
            void * opaque,
            ggml_tensor * q,
            ggml_tensor * k,
            ggml_tensor * v,
            ggml_tensor * out_weight,
            ggml_tensor * out_bias,
            ggml_tensor * out_scale,
            float kq_scale,
            int32_t layer_index) -> ggml_tensor * {
        auto * bridge = static_cast<nedolm_attention_bridge_v1 *>(opaque);
        GGML_ASSERT(bridge && bridge->graph && bridge->input);
        return bridge->graph->build_attn(
            bridge->input,
            out_weight,
            out_bias,
            out_scale,
            q, k, v,
            nullptr, nullptr, nullptr,
            kq_scale,
            layer_index);
    };

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // [2, vocab] GET_ROWS token ids -> [2, n_tokens], through Graph ABI v1.
    mercan_tensor_handle_v1 role_mask_h = mg.get_rows(
        &mercan_graph,
        require_tensor_h("blk.0.ffn_gate_tid2eid.weight"),
        mercan_ggml_tensor_to_handle_v1(inp_tokens));
    role_mask_h = mg.cast_f32(&mercan_graph, role_mask_h);
    ggml_tensor * role_mask = mercan_ggml_tensor_from_handle_v1(role_mask_h);
    GGML_ASSERT(role_mask != nullptr);
    cb(role_mask, "nedolm_role_mask", -1);

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;
        ggml_tensor * cur = graph_rms_norm_weight(inpL, require_tensor(block_tensor_name(il, "attn_norm.weight")), il);
        cb(cur, "attn_norm", il);

        GGML_ASSERT(require_tensor(block_tensor_name(il, "attn_q.weight")) == model.layers[il].wq);
        GGML_ASSERT(require_tensor(block_tensor_name(il, "attn_k.weight")) == model.layers[il].wk);
        GGML_ASSERT(require_tensor(block_tensor_name(il, "attn_v.weight")) == model.layers[il].wv);
        auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                n_embd_head, n_head, n_head_kv, il);
        mercan_tensor_handle_v1 q_h = mg.rope_ext(
                &mercan_graph, mercan_ggml_tensor_to_handle_v1(Qcur),
                mercan_ggml_tensor_to_handle_v1(inp_pos), MERCAN_TENSOR_NONE_V1,
                n_rot, mercan_rope_mode, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        mercan_tensor_handle_v1 k_h = mg.rope_ext(
                &mercan_graph, mercan_ggml_tensor_to_handle_v1(Kcur),
                mercan_ggml_tensor_to_handle_v1(inp_pos), MERCAN_TENSOR_NONE_V1,
                n_rot, mercan_rope_mode, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        Qcur = mercan_ggml_tensor_from_handle_v1(q_h);
        Kcur = mercan_ggml_tensor_from_handle_v1(k_h);
        GGML_ASSERT(Qcur != nullptr && Kcur != nullptr);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        mercan_tensor_handle_v1 attn_h = mg.self_attention(
                &mercan_graph,
                mercan_ggml_tensor_to_handle_v1(Qcur),
                mercan_ggml_tensor_to_handle_v1(Kcur),
                mercan_ggml_tensor_to_handle_v1(Vcur),
                require_tensor_h(block_tensor_name(il, "attn_output.weight")),
                mercan_ggml_tensor_to_handle_v1(model.layers[il].wo_b),
                mercan_ggml_tensor_to_handle_v1(model.layers[il].wo_s),
                1.0f/sqrtf(float(n_embd_head)),
                il);
        cur = mercan_ggml_tensor_from_handle_v1(attn_h);
        GGML_ASSERT(cur != nullptr);

        if (il == n_layer - 1 && inp_out_ids) {
            mercan_tensor_handle_v1 cur_h = mg.get_rows(
                &mercan_graph, mercan_ggml_tensor_to_handle_v1(cur), mercan_ggml_tensor_to_handle_v1(inp_out_ids));
            mercan_tensor_handle_v1 residual_h = mg.get_rows(
                &mercan_graph, mercan_ggml_tensor_to_handle_v1(inpSA), mercan_ggml_tensor_to_handle_v1(inp_out_ids));
            cur = mercan_ggml_tensor_from_handle_v1(cur_h);
            inpSA = mercan_ggml_tensor_from_handle_v1(residual_h);
            GGML_ASSERT(cur != nullptr && inpSA != nullptr);
        }
        mercan_tensor_handle_v1 ffn_inp_h = mg.add(
            &mercan_graph, mercan_ggml_tensor_to_handle_v1(cur), mercan_ggml_tensor_to_handle_v1(inpSA));
        ggml_tensor * ffn_inp = mercan_ggml_tensor_from_handle_v1(ffn_inp_h);
        GGML_ASSERT(ffn_inp != nullptr);
        cur = graph_rms_norm_weight(ffn_inp, require_tensor(block_tensor_name(il, "ffn_norm.weight")), il);
        cb(cur, "ffn_norm", il);

        if (il < static_cast<int>(nedolm.morph_layer_count)) {
            ggml_tensor * up = build_lora_mm(require_tensor(block_tensor_name(il, "ffn_up.weight")), cur);
            ggml_tensor * gate = build_lora_mm(require_tensor(block_tensor_name(il, "ffn_gate.weight")), cur);

            mercan_tensor_handle_v1 z_h = mg.swiglu_split(
                &mercan_graph, mercan_ggml_tensor_to_handle_v1(gate), mercan_ggml_tensor_to_handle_v1(up));
            GGML_ASSERT(z_h != MERCAN_TENSOR_NONE_V1);
            ggml_tensor * z = mercan_ggml_tensor_from_handle_v1(z_h);
            cb(z, "nedolm_swiglu", il);

            const int64_t z_rows = mg.dim(&mercan_graph, z_h, 1);
            const size_t z_nb0 = mg.stride_bytes(&mercan_graph, z_h, 0);
            const size_t z_nb1 = mg.stride_bytes(&mercan_graph, z_h, 1);
            const int64_t role_rows = mg.dim(&mercan_graph, role_mask_h, 1);
            const size_t role_nb0 = mg.stride_bytes(&mercan_graph, role_mask_h, 0);
            const size_t role_nb1 = mg.stride_bytes(&mercan_graph, role_mask_h, 1);

            mercan_tensor_handle_v1 shared_h = mg.view_2d(&mercan_graph, z_h, morph_shared, z_rows, z_nb1, 0);
            mercan_tensor_handle_v1 root_h = mg.view_2d(&mercan_graph, z_h, morph_root, z_rows, z_nb1, morph_shared * z_nb0);
            mercan_tensor_handle_v1 suffix_h = mg.view_2d(&mercan_graph, z_h, morph_suffix, z_rows, z_nb1, (morph_shared + morph_root) * z_nb0);

            mercan_tensor_handle_v1 root_gate_h = mg.view_2d(&mercan_graph, role_mask_h, 1, role_rows, role_nb1, 0);
            mercan_tensor_handle_v1 suffix_gate_h = mg.view_2d(&mercan_graph, role_mask_h, 1, role_rows, role_nb1, role_nb0);
            root_h = mg.mul(&mercan_graph, root_h, root_gate_h);
            suffix_h = mg.mul(&mercan_graph, suffix_h, suffix_gate_h);

            z_h = mg.concat(&mercan_graph, shared_h, root_h, 0);
            z_h = mg.concat(&mercan_graph, z_h, suffix_h, 0);
            z = mercan_ggml_tensor_from_handle_v1(z_h);
            GGML_ASSERT(z != nullptr);
            cur = build_lora_mm(require_tensor(block_tensor_name(il, "ffn_down.weight")), z);
        } else {
            cur = build_ffn(cur,
                    require_tensor(block_tensor_name(il, "ffn_up.weight")), nullptr, nullptr,
                    require_tensor(block_tensor_name(il, "ffn_gate.weight")), nullptr, nullptr,
                    require_tensor(block_tensor_name(il, "ffn_down.weight")), nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        }
        cb(cur, "ffn_out", il);

        mercan_tensor_handle_v1 residual_out_h = mg.add(
            &mercan_graph, mercan_ggml_tensor_to_handle_v1(cur), mercan_ggml_tensor_to_handle_v1(ffn_inp));
        cur = mercan_ggml_tensor_from_handle_v1(residual_out_h);
        GGML_ASSERT(cur != nullptr);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        inpL = cur;
    }

    ggml_tensor * cur = graph_rms_norm_weight(inpL, require_tensor("output_norm.weight"), -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;
    cur = build_lora_mm(require_tensor("output.weight"), cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
