#include "models.h"

namespace {
constexpr int NEDOLM_MORPH_LAYERS = 18;
constexpr int64_t NEDOLM_SHARED = 4224;
constexpr int64_t NEDOLM_ROOT = 704;
constexpr int64_t NEDOLM_SUFFIX = 704;
}

void llama_model_nedolm::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);

    if (hparams.n_swa == 0) {
        throw std::runtime_error("NedoLM requires a positive sliding-window length");
    }
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    // period=0 means every transformer block is sliding-window attention.
    hparams.set_swa_pattern(0);

    hparams.n_embd_head_k_swa = hparams.n_embd_head_k_full;
    hparams.n_embd_head_v_swa = hparams.n_embd_head_v_full;
    hparams.n_rot_swa = hparams.n_rot_full;
    hparams.rope_freq_base_train_swa = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;

    if (hparams.n_layer() != 24 || hparams.n_embd != 1536 || hparams.n_ff() != 5632) {
        throw std::runtime_error("unexpected NedoLM-0.8B dimensions");
    }
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
    // Existing GET_ROWS tensor slot is reused so quantization/offload semantics are already correct.
    layers[0].ffn_gate_tid2eid = create_tensor(
        tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", 0), {2, n_vocab}, 0);
}

std::unique_ptr<llm_graph_context> llama_model_nedolm::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_nedolm::graph::graph(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);
    GGML_ASSERT(hparams.n_ff() == NEDOLM_SHARED + NEDOLM_ROOT + NEDOLM_SUFFIX);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_tokens = res->t_inp_tokens;
    GGML_ASSERT(inp_tokens != nullptr);

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv_iswa();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // [2, vocab] GET_ROWS token ids -> [2, n_tokens]
    ggml_tensor * role_mask = ggml_get_rows(ctx0, model.layers[0].ffn_gate_tid2eid, inp_tokens);
    role_mask = ggml_cast(ctx0, role_mask, GGML_TYPE_F32);
    cb(role_mask, "nedolm_role_mask", -1);

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;
        ggml_tensor * cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                n_embd_head, n_head, n_head_kv, il);
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        cur = build_attn(inp_attn,
                model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                1.0f/sqrtf(float(n_embd_head)), il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if (il < NEDOLM_MORPH_LAYERS) {
            ggml_tensor * up = build_lora_mm(model.layers[il].ffn_up, cur);
            ggml_tensor * gate = build_lora_mm(model.layers[il].ffn_gate, cur);
            ggml_tensor * z = ggml_swiglu_split(ctx0, gate, up);
            cb(z, "nedolm_swiglu", il);

            ggml_tensor * shared = ggml_view_2d(ctx0, z, NEDOLM_SHARED, z->ne[1], z->nb[1], 0);
            ggml_tensor * root = ggml_view_2d(ctx0, z, NEDOLM_ROOT, z->ne[1], z->nb[1], NEDOLM_SHARED * z->nb[0]);
            ggml_tensor * suffix = ggml_view_2d(ctx0, z, NEDOLM_SUFFIX, z->ne[1], z->nb[1], (NEDOLM_SHARED + NEDOLM_ROOT) * z->nb[0]);

            ggml_tensor * root_gate = ggml_view_2d(ctx0, role_mask, 1, role_mask->ne[1], role_mask->nb[1], 0);
            ggml_tensor * suffix_gate = ggml_view_2d(ctx0, role_mask, 1, role_mask->ne[1], role_mask->nb[1], role_mask->nb[0]);
            root = ggml_mul(ctx0, root, root_gate);
            suffix = ggml_mul(ctx0, suffix, suffix_gate);

            z = ggml_concat(ctx0, shared, root, 0);
            z = ggml_concat(ctx0, z, suffix, 0);
            cur = build_lora_mm(model.layers[il].ffn_down, z);
        } else {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up, nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        }
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;
    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
