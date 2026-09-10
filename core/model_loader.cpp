#include "mercan_internal.hpp"
#include "gguf.h"

#include <cstdint>
#include <string>

namespace {

int metadata_has_key(const mercan_metadata_v1 * metadata, const char * key) {
    if (!metadata || !metadata->userdata || !key) return 0;
    auto * ctx = static_cast<const gguf_context *>(metadata->userdata);
    return gguf_find_key(ctx, key) >= 0 ? 1 : 0;
}

const char * metadata_get_string(const mercan_metadata_v1 * metadata, const char * key) {
    if (!metadata || !metadata->userdata || !key) return nullptr;
    auto * ctx = static_cast<const gguf_context *>(metadata->userdata);
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) return nullptr;
    return gguf_get_val_str(ctx, id);
}

int64_t metadata_get_i64(const mercan_metadata_v1 * metadata, const char * key, int64_t fallback) {
    if (!metadata || !metadata->userdata || !key) return fallback;
    auto * ctx = static_cast<const gguf_context *>(metadata->userdata);
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return fallback;
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(ctx, id);
        case GGUF_TYPE_INT8:   return gguf_get_val_i8(ctx, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16(ctx, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT64: return static_cast<int64_t>(gguf_get_val_u64(ctx, id));
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(ctx, id);
        default: return fallback;
    }
}

double metadata_get_f64(const mercan_metadata_v1 * metadata, const char * key, double fallback) {
    if (!metadata || !metadata->userdata || !key) return fallback;
    auto * ctx = static_cast<const gguf_context *>(metadata->userdata);
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return fallback;
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(ctx, id);
        case GGUF_TYPE_FLOAT64: return gguf_get_val_f64(ctx, id);
        default: return static_cast<double>(metadata_get_i64(metadata, key, static_cast<int64_t>(fallback)));
    }
}

}

mercan_metadata_probe::mercan_metadata_probe() {
    view.abi_version = MERCAN_ARCH_ABI_VERSION;
    view.struct_size = sizeof(mercan_metadata_v1);
    view.userdata = nullptr;
    view.has_key = metadata_has_key;
    view.get_string = metadata_get_string;
    view.get_i64 = metadata_get_i64;
    view.get_f64 = metadata_get_f64;
}

mercan_metadata_probe::~mercan_metadata_probe() {
    if (ctx) gguf_free(ctx);
}

bool mercan_metadata_open(const char * path, mercan_metadata_probe & out, std::string & error) {
    if (!path || !*path) {
        error = "model path is empty";
        return false;
    }
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;
    out.ctx = gguf_init_from_file(path, params);
    if (!out.ctx) {
        error = std::string("failed to read .mercan/GGUF metadata: ") + path;
        return false;
    }
    out.view.userdata = out.ctx;
    return true;
}

std::string mercan_metadata_string(const mercan_metadata_v1 * metadata, const char * key) {
    if (!metadata || !metadata->get_string) return {};
    const char * value = metadata->get_string(metadata, key);
    return value ? value : "";
}
