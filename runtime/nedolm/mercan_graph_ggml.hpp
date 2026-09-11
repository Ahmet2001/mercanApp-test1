#pragma once

#include "mercan_graph.h"
#include "ggml.h"

#include <cstdint>

struct mercan_ggml_graph_userdata_v1 {
    ggml_context * ctx = nullptr;
};

static inline ggml_tensor * mercan_ggml_tensor_from_handle_v1(mercan_tensor_handle_v1 handle) {
    return reinterpret_cast<ggml_tensor *>(static_cast<uintptr_t>(handle));
}

static inline mercan_tensor_handle_v1 mercan_ggml_tensor_to_handle_v1(ggml_tensor * tensor) {
    return static_cast<mercan_tensor_handle_v1>(reinterpret_cast<uintptr_t>(tensor));
}

static inline mercan_ggml_graph_userdata_v1 * mercan_ggml_userdata_v1(mercan_graph_builder_v1 * builder) {
    return builder ? static_cast<mercan_ggml_graph_userdata_v1 *>(builder->userdata) : nullptr;
}

static inline mercan_rope_mode_v1 mercan_ggml_rope_mode_to_mercan_v1(int mode) {
    switch (mode) {
        case GGML_ROPE_TYPE_NORMAL: return MERCAN_ROPE_MODE_NORMAL_V1;
        case GGML_ROPE_TYPE_NEOX:   return MERCAN_ROPE_MODE_NEOX_V1;
        default:                    return MERCAN_ROPE_MODE_UNSUPPORTED_V1;
    }
}

static inline int mercan_ggml_rope_mode_from_mercan_v1(mercan_rope_mode_v1 mode) {
    switch (mode) {
        case MERCAN_ROPE_MODE_NORMAL_V1: return GGML_ROPE_TYPE_NORMAL;
        case MERCAN_ROPE_MODE_NEOX_V1:   return GGML_ROPE_TYPE_NEOX;
        default:                          return -1;
    }
}

static inline int64_t mercan_ggml_dim_v1(mercan_graph_builder_v1 *, mercan_tensor_handle_v1 tensor, uint32_t axis) {
    ggml_tensor * t = mercan_ggml_tensor_from_handle_v1(tensor);
    return t && axis < GGML_MAX_DIMS ? t->ne[axis] : 0;
}

static inline size_t mercan_ggml_stride_bytes_v1(mercan_graph_builder_v1 *, mercan_tensor_handle_v1 tensor, uint32_t axis) {
    ggml_tensor * t = mercan_ggml_tensor_from_handle_v1(tensor);
    return t && axis < GGML_MAX_DIMS ? t->nb[axis] : 0;
}

static inline mercan_tensor_handle_v1 mercan_ggml_get_rows_v1(
        mercan_graph_builder_v1 * builder,
        mercan_tensor_handle_v1 src,
        mercan_tensor_handle_v1 indices) {
    auto * ud = mercan_ggml_userdata_v1(builder);
    auto * a = mercan_ggml_tensor_from_handle_v1(src);
    auto * b = mercan_ggml_tensor_from_handle_v1(indices);
    return ud && ud->ctx && a && b ? mercan_ggml_tensor_to_handle_v1(ggml_get_rows(ud->ctx, a, b)) : MERCAN_TENSOR_NONE_V1;
}

static inline mercan_tensor_handle_v1 mercan_ggml_cast_f32_v1(
        mercan_graph_builder_v1 * builder,
        mercan_tensor_handle_v1 src) {
    auto * ud = mercan_ggml_userdata_v1(builder);
    auto * a = mercan_ggml_tensor_from_handle_v1(src);
    return ud && ud->ctx && a ? mercan_ggml_tensor_to_handle_v1(ggml_cast(ud->ctx, a, GGML_TYPE_F32)) : MERCAN_TENSOR_NONE_V1;
}

static inline mercan_tensor_handle_v1 mercan_ggml_swiglu_split_v1(
        mercan_graph_builder_v1 * builder,
        mercan_tensor_handle_v1 gate,
        mercan_tensor_handle_v1 up) {
    auto * ud = mercan_ggml_userdata_v1(builder);
    auto * a = mercan_ggml_tensor_from_handle_v1(gate);
    auto * b = mercan_ggml_tensor_from_handle_v1(up);
    return ud && ud->ctx && a && b ? mercan_ggml_tensor_to_handle_v1(ggml_swiglu_split(ud->ctx, a, b)) : MERCAN_TENSOR_NONE_V1;
}

static inline mercan_tensor_handle_v1 mercan_ggml_view_2d_v1(
        mercan_graph_builder_v1 * builder,
        mercan_tensor_handle_v1 src,
        int64_t ne0,
        int64_t ne1,
        size_t row_stride_bytes,
        size_t offset_bytes) {
    auto * ud = mercan_ggml_userdata_v1(builder);
    auto * a = mercan_ggml_tensor_from_handle_v1(src);
    return ud && ud->ctx && a ? mercan_ggml_tensor_to_handle_v1(
        ggml_view_2d(ud->ctx, a, ne0, ne1, row_stride_bytes, offset_bytes)) : MERCAN_TENSOR_NONE_V1;
}

static inline mercan_tensor_handle_v1 mercan_ggml_mul_v1(
        mercan_graph_builder_v1 * builder,
        mercan_tensor_handle_v1 lhs,
        mercan_tensor_handle_v1 rhs) {
    auto * ud = mercan_ggml_userdata_v1(builder);
    auto * a = mercan_ggml_tensor_from_handle_v1(lhs);
    auto * b = mercan_ggml_tensor_from_handle_v1(rhs);
    return ud && ud->ctx && a && b ? mercan_ggml_tensor_to_handle_v1(ggml_mul(ud->ctx, a, b)) : MERCAN_TENSOR_NONE_V1;
}

static inline mercan_tensor_handle_v1 mercan_ggml_add_v1(
        mercan_graph_builder_v1 * builder,
        mercan_tensor_handle_v1 lhs,
        mercan_tensor_handle_v1 rhs) {
    auto * ud = mercan_ggml_userdata_v1(builder);
    auto * a = mercan_ggml_tensor_from_handle_v1(lhs);
    auto * b = mercan_ggml_tensor_from_handle_v1(rhs);
    return ud && ud->ctx && a && b ? mercan_ggml_tensor_to_handle_v1(ggml_add(ud->ctx, a, b)) : MERCAN_TENSOR_NONE_V1;
}

static inline mercan_tensor_handle_v1 mercan_ggml_concat_v1(
        mercan_graph_builder_v1 * builder,
        mercan_tensor_handle_v1 lhs,
        mercan_tensor_handle_v1 rhs,
        int32_t dim) {
    auto * ud = mercan_ggml_userdata_v1(builder);
    auto * a = mercan_ggml_tensor_from_handle_v1(lhs);
    auto * b = mercan_ggml_tensor_from_handle_v1(rhs);
    return ud && ud->ctx && a && b ? mercan_ggml_tensor_to_handle_v1(ggml_concat(ud->ctx, a, b, dim)) : MERCAN_TENSOR_NONE_V1;
}

static inline mercan_tensor_handle_v1 mercan_ggml_matmul_v1(
        mercan_graph_builder_v1 * builder,
        mercan_tensor_handle_v1 weight,
        mercan_tensor_handle_v1 input) {
    auto * ud = mercan_ggml_userdata_v1(builder);
    auto * w = mercan_ggml_tensor_from_handle_v1(weight);
    auto * x = mercan_ggml_tensor_from_handle_v1(input);
    return ud && ud->ctx && w && x ? mercan_ggml_tensor_to_handle_v1(ggml_mul_mat(ud->ctx, w, x)) : MERCAN_TENSOR_NONE_V1;
}

static inline mercan_tensor_handle_v1 mercan_ggml_rms_norm_v1(
        mercan_graph_builder_v1 * builder,
        mercan_tensor_handle_v1 src,
        float eps) {
    auto * ud = mercan_ggml_userdata_v1(builder);
    auto * x = mercan_ggml_tensor_from_handle_v1(src);
    return ud && ud->ctx && x ? mercan_ggml_tensor_to_handle_v1(ggml_rms_norm(ud->ctx, x, eps)) : MERCAN_TENSOR_NONE_V1;
}

static inline mercan_tensor_handle_v1 mercan_ggml_rope_ext_v1(
        mercan_graph_builder_v1 * builder,
        mercan_tensor_handle_v1 src,
        mercan_tensor_handle_v1 positions,
        mercan_tensor_handle_v1 freq_factors,
        int32_t n_dims,
        mercan_rope_mode_v1 mode,
        int32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    auto * ud = mercan_ggml_userdata_v1(builder);
    auto * x = mercan_ggml_tensor_from_handle_v1(src);
    auto * pos = mercan_ggml_tensor_from_handle_v1(positions);
    auto * factors = mercan_ggml_tensor_from_handle_v1(freq_factors);
    const int ggml_mode = mercan_ggml_rope_mode_from_mercan_v1(mode);
    return ud && ud->ctx && x && pos && ggml_mode >= 0
        ? mercan_ggml_tensor_to_handle_v1(ggml_rope_ext(
              ud->ctx, x, pos, factors, n_dims, ggml_mode, n_ctx_orig,
              freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow))
        : MERCAN_TENSOR_NONE_V1;
}

static const mercan_graph_api_v1 MERCAN_GGML_GRAPH_API_V1 = {
    MERCAN_GRAPH_ABI_VERSION,
    sizeof(mercan_graph_api_v1),
    mercan_ggml_dim_v1,
    mercan_ggml_stride_bytes_v1,
    mercan_ggml_get_rows_v1,
    mercan_ggml_cast_f32_v1,
    mercan_ggml_swiglu_split_v1,
    mercan_ggml_view_2d_v1,
    mercan_ggml_mul_v1,
    mercan_ggml_add_v1,
    mercan_ggml_concat_v1,
    mercan_ggml_matmul_v1,
    mercan_ggml_rms_norm_v1,
    mercan_ggml_rope_ext_v1,
};

static inline mercan_graph_builder_v1 mercan_make_ggml_graph_builder_v1(mercan_ggml_graph_userdata_v1 * userdata) {
    mercan_graph_builder_v1 out{};
    out.abi_version = MERCAN_GRAPH_ABI_VERSION;
    out.struct_size = sizeof(mercan_graph_builder_v1);
    out.userdata = userdata;
    out.api = &MERCAN_GGML_GRAPH_API_V1;
    return out;
}
