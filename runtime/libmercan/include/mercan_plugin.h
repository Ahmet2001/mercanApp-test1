#ifndef MERCAN_PLUGIN_H
#define MERCAN_PLUGIN_H

#include "mercan_arch.h"
#include "mercan_tokenizer.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MERCAN_PLUGIN_ABI_VERSION 1u
#define MERCAN_PLUGIN_ENTRY_SYMBOL_V1 "mercan_plugin_entry_v1"

#if defined(_WIN32)
#  define MERCAN_PLUGIN_EXPORT __declspec(dllexport)
#else
#  define MERCAN_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

typedef struct mercan_plugin_host_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    int (*register_architecture)(const mercan_architecture_v1 * architecture);
    int (*register_tokenizer)(const mercan_tokenizer_v1 * tokenizer);
} mercan_plugin_host_v1;

typedef struct mercan_plugin_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    const char * name;
    const char * version;
    uint64_t flags;

    /* Called exactly once after dlopen/LoadLibrary. Return 0 on success. */
    int (*init)(const mercan_plugin_host_v1 * host, char * error, size_t error_capacity);
} mercan_plugin_v1;

typedef const mercan_plugin_v1 * (*mercan_plugin_entry_fn_v1)(void);

#define MERCAN_PLUGIN_HOST_V1_BASE_SIZE sizeof(mercan_plugin_host_v1)
#define MERCAN_PLUGIN_V1_BASE_SIZE \
    (offsetof(mercan_plugin_v1, init) + sizeof(((mercan_plugin_v1 *)0)->init))

/* Load once and retain the native library for process lifetime. 0=loaded, 1=already loaded. */
MERCAN_API int mercan_plugin_load_v1(const char * path);
MERCAN_API size_t mercan_plugin_count_v1(void);
MERCAN_API const char * mercan_plugin_name_v1(size_t index);
MERCAN_API const char * mercan_plugin_version_v1(size_t index);
MERCAN_API const char * mercan_plugin_path_v1(size_t index);
MERCAN_API const char * mercan_plugin_last_error_v1(void);

#ifdef __cplusplus
}
#endif

#endif
