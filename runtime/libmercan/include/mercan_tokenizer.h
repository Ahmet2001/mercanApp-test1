#ifndef MERCAN_TOKENIZER_H
#define MERCAN_TOKENIZER_H

#include "mercan_arch.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MERCAN_TOKENIZER_ABI_VERSION 1u

enum mercan_tokenizer_flags_v1 {
    /* Tokenization is implemented by the compiled inference backend. */
    MERCAN_TOKENIZER_BACKEND_MANAGED = 1ull << 0,
    MERCAN_TOKENIZER_BUILTIN = 1ull << 1,
};

typedef struct mercan_tokenizer_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    const char * name;
    const char * display_name;
    uint64_t flags;

    int (*probe)(const mercan_metadata_v1 * metadata);
    int (*validate)(const mercan_metadata_v1 * metadata, char * error, size_t error_capacity);

    /*
     * Optional external tokenizer lifecycle. If create/encode/decode_piece are NULL,
     * Mercan falls back to the compiled backend tokenizer.
     */
    void * (*create)(const mercan_metadata_v1 * metadata, char * error, size_t error_capacity);
    void (*destroy)(void * state);
    int32_t (*encode)(void * state, const char * text, size_t text_len,
                      bool add_special, bool parse_special,
                      mercan_token * out_tokens, int32_t capacity);
    int32_t (*decode_piece)(void * state, mercan_token token,
                            char * out, int32_t capacity, bool special);
} mercan_tokenizer_v1;

MERCAN_API int mercan_tokenizer_register_v1(const mercan_tokenizer_v1 * tokenizer);
MERCAN_API const mercan_tokenizer_v1 * mercan_tokenizer_find_v1(const char * name);
MERCAN_API size_t mercan_tokenizer_count_v1(void);
MERCAN_API const mercan_tokenizer_v1 * mercan_tokenizer_at_v1(size_t index);

#ifdef __cplusplus
}
#endif

#endif
