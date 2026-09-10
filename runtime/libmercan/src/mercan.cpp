#include "mercan.h"
#include "llama.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <string>
#include <thread>
#include <vector>

struct mercan_model {
    llama_model * impl = nullptr;
};

struct mercan_context {
    llama_context * impl = nullptr;
    mercan_model * model = nullptr;
};

static thread_local std::string g_last_error;

static void set_error(const std::string & s) {
    g_last_error = s;
}

extern "C" {

const char * mercan_version(void) {
    return "0.1.0";
}

const char * mercan_last_error(void) {
    return g_last_error.c_str();
}

void mercan_backend_init(void) {
    g_last_error.clear();
    llama_backend_init();
}

void mercan_backend_free(void) {
    llama_backend_free();
}

mercan_model_params mercan_model_default_params(void) {
    mercan_model_params p{};
    p.n_gpu_layers = 0;
    p.use_mmap = true;
    p.check_tensors = false;
    return p;
}

mercan_context_params mercan_context_default_params(void) {
    mercan_context_params p{};
    p.n_ctx = 4096;
    p.n_batch = 512;
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    p.n_threads = static_cast<int32_t>(hw);
    p.n_threads_batch = static_cast<int32_t>(hw);
    return p;
}

mercan_model * mercan_model_load(const char * path, mercan_model_params params) {
    g_last_error.clear();
    if (!path || !*path) {
        set_error("model path is empty");
        return nullptr;
    }
    try {
        llama_model_params lp = llama_model_default_params();
        lp.n_gpu_layers = params.n_gpu_layers;
        lp.load_mode = params.use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
        lp.check_tensors = params.check_tensors;

        llama_model * lm = llama_model_load_from_file(path, lp);
        if (!lm) {
            set_error(std::string("failed to load model: ") + path);
            return nullptr;
        }
        mercan_model * out = new mercan_model();
        out->impl = lm;
        return out;
    } catch (const std::exception & e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("unknown error while loading model");
        return nullptr;
    }
}

void mercan_model_free(mercan_model * model) {
    if (!model) return;
    if (model->impl) llama_model_free(model->impl);
    delete model;
}

mercan_context * mercan_context_create(mercan_model * model, mercan_context_params params) {
    g_last_error.clear();
    if (!model || !model->impl) {
        set_error("invalid model");
        return nullptr;
    }
    try {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = params.n_ctx;
        cp.n_batch = params.n_batch;
        cp.n_threads = params.n_threads;
        cp.n_threads_batch = params.n_threads_batch;
        llama_context * lc = llama_init_from_model(model->impl, cp);
        if (!lc) {
            set_error("failed to create llama context");
            return nullptr;
        }
        mercan_context * out = new mercan_context();
        out->impl = lc;
        out->model = model;
        return out;
    } catch (const std::exception & e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("unknown error while creating context");
        return nullptr;
    }
}

void mercan_context_free(mercan_context * ctx) {
    if (!ctx) return;
    if (ctx->impl) llama_free(ctx->impl);
    delete ctx;
}

int32_t mercan_tokenize(
    mercan_model * model,
    const char * text,
    size_t text_len,
    bool add_special,
    bool parse_special,
    mercan_token * out_tokens,
    int32_t capacity) {
    g_last_error.clear();
    if (!model || !model->impl || !text) {
        set_error("invalid tokenize arguments");
        return 0;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model->impl);
    if (!vocab) {
        set_error("model vocabulary is unavailable");
        return 0;
    }
    return llama_tokenize(
        vocab,
        text,
        static_cast<int32_t>(text_len),
        reinterpret_cast<llama_token *>(out_tokens),
        capacity,
        add_special,
        parse_special);
}

int32_t mercan_decode(mercan_context * ctx, const mercan_token * tokens, int32_t n_tokens) {
    g_last_error.clear();
    if (!ctx || !ctx->impl || !tokens || n_tokens <= 0) {
        set_error("invalid decode arguments");
        return -1;
    }
    llama_batch batch = llama_batch_get_one(
        const_cast<llama_token *>(reinterpret_cast<const llama_token *>(tokens)),
        n_tokens);
    const int rc = llama_decode(ctx->impl, batch);
    if (rc != 0) set_error("llama_decode failed with code " + std::to_string(rc));
    return rc;
}

const float * mercan_logits(mercan_context * ctx) {
    if (!ctx || !ctx->impl) return nullptr;
    return llama_get_logits_ith(ctx->impl, -1);
}

int32_t mercan_vocab_size(mercan_model * model) {
    if (!model || !model->impl) return 0;
    const llama_vocab * vocab = llama_model_get_vocab(model->impl);
    return vocab ? llama_vocab_n_tokens(vocab) : 0;
}

uint32_t mercan_context_size(mercan_context * ctx) {
    if (!ctx || !ctx->impl) return 0;
    return llama_n_ctx(ctx->impl);
}

mercan_token mercan_bos_token(mercan_model * model) {
    if (!model || !model->impl) return -1;
    const llama_vocab * vocab = llama_model_get_vocab(model->impl);
    return vocab ? llama_vocab_bos(vocab) : -1;
}

mercan_token mercan_eos_token(mercan_model * model) {
    if (!model || !model->impl) return -1;
    const llama_vocab * vocab = llama_model_get_vocab(model->impl);
    return vocab ? llama_vocab_eos(vocab) : -1;
}

mercan_token mercan_pad_token(mercan_model * model) {
    if (!model || !model->impl) return -1;
    const llama_vocab * vocab = llama_model_get_vocab(model->impl);
    return vocab ? llama_vocab_pad(vocab) : -1;
}

int32_t mercan_token_to_piece(
    mercan_model * model,
    mercan_token token,
    char * out,
    int32_t capacity,
    bool special) {
    if (!model || !model->impl) return 0;
    const llama_vocab * vocab = llama_model_get_vocab(model->impl);
    if (!vocab) return 0;
    return llama_token_to_piece(vocab, token, out, capacity, 0, special);
}

} // extern "C"
