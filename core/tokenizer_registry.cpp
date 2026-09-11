#include "mercan_tokenizer.h"
#include "mercan_internal.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace {
std::mutex g_tokenizer_mutex;
std::vector<std::unique_ptr<mercan_tokenizer_v1>> g_tokenizers;

bool valid_tokenizer_descriptor(const mercan_tokenizer_v1 * tok) {
    return tok && tok->abi_version == MERCAN_TOKENIZER_ABI_VERSION &&
           tok->struct_size >= MERCAN_TOKENIZER_V1_BASE_SIZE &&
           tok->name && *tok->name;
}
}

extern "C" {

int mercan_tokenizer_register_v1(const mercan_tokenizer_v1 * tokenizer) {
    if (!valid_tokenizer_descriptor(tokenizer)) return -1;
    std::lock_guard<std::mutex> lock(g_tokenizer_mutex);
    for (const auto & item : g_tokenizers) {
        if (std::strcmp(item->name, tokenizer->name) == 0) return 1;
    }
    auto stored = std::make_unique<mercan_tokenizer_v1>();
    std::memset(stored.get(), 0, sizeof(*stored));
    const size_t copy_size = std::min<size_t>(tokenizer->struct_size, sizeof(*stored));
    std::memcpy(stored.get(), tokenizer, copy_size);
    stored->struct_size = static_cast<uint32_t>(copy_size);
    g_tokenizers.emplace_back(std::move(stored));
    return 0;
}

const mercan_tokenizer_v1 * mercan_tokenizer_find_v1(const char * name) {
    mercan_ensure_builtin_plugins();
    if (!name || !*name) return nullptr;
    std::lock_guard<std::mutex> lock(g_tokenizer_mutex);
    for (const auto & item : g_tokenizers) {
        if (std::strcmp(item->name, name) == 0) return item.get();
    }
    return nullptr;
}

size_t mercan_tokenizer_count_v1(void) {
    mercan_ensure_builtin_plugins();
    std::lock_guard<std::mutex> lock(g_tokenizer_mutex);
    return g_tokenizers.size();
}

const mercan_tokenizer_v1 * mercan_tokenizer_at_v1(size_t index) {
    mercan_ensure_builtin_plugins();
    std::lock_guard<std::mutex> lock(g_tokenizer_mutex);
    if (index >= g_tokenizers.size()) return nullptr;
    return g_tokenizers[index].get();
}

}
