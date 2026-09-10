#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one match, got {count}: {old[:100]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: upgrade_llama_patch.py /path/to/llama.cpp")
    root = Path(sys.argv[1]).resolve()

    vocab = root / "src" / "llama-vocab.cpp"
    models = root / "src" / "models" / "models.h"

    replace_once(
        vocab,
        '''extern "C" const uint16_t * nedo004_encode(const uint8_t * data, size_t len, size_t * out_len);\nextern "C" const char * nedo004_vocab_sha256();\n''',
        '''extern "C" const uint16_t * nedo004_encode(const uint8_t * data, size_t len, size_t * out_len);\nextern "C" const char * nedo004_vocab_sha256();\nextern "C" int nedo004_set_vocab(const uint8_t * data, size_t len, const char * expected_sha);\n''',
    )

    old_guard = '''                const int sha_keyidx = gguf_find_key(ctx, "nedolm.vocab_sha256");
                if (sha_keyidx < 0 || gguf_get_kv_type(ctx, sha_keyidx) != GGUF_TYPE_STRING) {
                    throw std::runtime_error("NedoLM GGUF is missing nedolm.vocab_sha256");
                }
                const char * gguf_sha = gguf_get_val_str(ctx, sha_keyidx);
                const char * runtime_sha = nedo004_vocab_sha256();
                if (runtime_sha == nullptr || gguf_sha == nullptr || std::string(gguf_sha) != runtime_sha) {
                    throw std::runtime_error(format(
                        "NDSRF004 tokenizer SHA mismatch: GGUF=%s runtime=%s",
                        gguf_sha ? gguf_sha : "<null>", runtime_sha ? runtime_sha : "<null>"));
                }
'''
    new_guard = '''                int sha_keyidx = gguf_find_key(ctx, "mercan.tokenizer.surface_vocab_sha256");
                if (sha_keyidx < 0) {
                    // Compatibility with pre-Mercan NedoLM GGUF files.
                    sha_keyidx = gguf_find_key(ctx, "nedolm.vocab_sha256");
                }
                if (sha_keyidx < 0 || gguf_get_kv_type(ctx, sha_keyidx) != GGUF_TYPE_STRING) {
                    throw std::runtime_error("NedoLM model is missing tokenizer SHA256 metadata");
                }
                const char * gguf_sha = gguf_get_val_str(ctx, sha_keyidx);

                const int vocab_keyidx = gguf_find_key(ctx, "mercan.tokenizer.surface_vocab");
                if (vocab_keyidx >= 0) {
                    if (gguf_get_kv_type(ctx, vocab_keyidx) != GGUF_TYPE_ARRAY ||
                        gguf_get_arr_type(ctx, vocab_keyidx) != GGUF_TYPE_UINT8) {
                        throw std::runtime_error("mercan.tokenizer.surface_vocab must be ARRAY<UINT8>");
                    }
                    const size_t vocab_n = gguf_get_arr_n(ctx, vocab_keyidx);
                    const auto * vocab_data = static_cast<const uint8_t *>(gguf_get_arr_data(ctx, vocab_keyidx));
                    if (vocab_data == nullptr || vocab_n == 0) {
                        throw std::runtime_error("Mercan tokenizer asset is empty");
                    }
                    const int set_rc = nedo004_set_vocab(vocab_data, vocab_n, gguf_sha);
                    if (set_rc != 0) {
                        throw std::runtime_error(format("failed to install Mercan tokenizer asset: code=%d", set_rc));
                    }
                }

                const char * runtime_sha = nedo004_vocab_sha256();
                if (runtime_sha == nullptr || gguf_sha == nullptr || std::string(gguf_sha) != runtime_sha) {
                    throw std::runtime_error(format(
                        "NDSRF004 tokenizer SHA mismatch: model=%s runtime=%s",
                        gguf_sha ? gguf_sha : "<null>", runtime_sha ? runtime_sha : "<null>"));
                }
'''
    replace_once(vocab, old_guard, new_guard)

    old_class = '''struct llama_model_nedolm : public llama_model_base {
    llama_model_nedolm(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
'''
    new_class = '''struct llama_model_nedolm : public llama_model_base {
    llama_model_nedolm(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    uint32_t morph_layer_count = 0;
    uint32_t morph_shared_width = 0;
    uint32_t morph_root_width = 0;
    uint32_t morph_suffix_width = 0;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
'''
    replace_once(models, old_class, new_class)

    print("UPGRADE_LLAMA_PATCH_OK")


if __name__ == "__main__":
    main()
