#include "mercan_plugin.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace {

#if defined(_WIN32)
using native_handle = HMODULE;
static native_handle native_open(const char * path) { return LoadLibraryA(path); }
static void * native_symbol(native_handle h, const char * name) {
    return reinterpret_cast<void *>(GetProcAddress(h, name));
}
static std::string native_error() {
    const DWORD code = GetLastError();
    return "Windows loader error " + std::to_string(static_cast<unsigned long>(code));
}
#else
using native_handle = void *;
static native_handle native_open(const char * path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
static void * native_symbol(native_handle h, const char * name) { return dlsym(h, name); }
static std::string native_error() { const char * e = dlerror(); return e ? e : "unknown dynamic loader error"; }
#endif

struct loaded_plugin {
    std::string path;
    std::string name;
    std::string version;
    native_handle handle = nullptr;
};

std::mutex g_plugin_mutex;
std::vector<std::unique_ptr<loaded_plugin>> g_plugins;
thread_local std::string g_plugin_error;

bool valid_descriptor(const mercan_plugin_v1 * plugin) {
    return plugin && plugin->abi_version == MERCAN_PLUGIN_ABI_VERSION &&
           plugin->struct_size >= MERCAN_PLUGIN_V1_BASE_SIZE &&
           plugin->name && *plugin->name && plugin->init;
}

const mercan_plugin_host_v1 HOST_V1 = {
    MERCAN_PLUGIN_ABI_VERSION,
    sizeof(mercan_plugin_host_v1),
    mercan_arch_register_v1,
    mercan_tokenizer_register_v1,
};

} // namespace

extern "C" {

int mercan_plugin_load_v1(const char * path) {
    g_plugin_error.clear();
    if (!path || !*path) {
        g_plugin_error = "plugin path is empty";
        return -1;
    }

    std::lock_guard<std::mutex> lock(g_plugin_mutex);
    for (const auto & item : g_plugins) {
        if (item->path == path) return 1;
    }

    native_handle handle = native_open(path);
    if (!handle) {
        g_plugin_error = "could not load plugin '" + std::string(path) + "': " + native_error();
        return -2;
    }

    void * symbol = native_symbol(handle, MERCAN_PLUGIN_ENTRY_SYMBOL_V1);
    if (!symbol) {
        g_plugin_error = "plugin is missing " MERCAN_PLUGIN_ENTRY_SYMBOL_V1 ": " + native_error();
        // Keep the handle alive: constructors may have registered callback-backed descriptors.
        auto failed = std::make_unique<loaded_plugin>();
        failed->path = path;
        failed->name = "<invalid>";
        failed->handle = handle;
        g_plugins.emplace_back(std::move(failed));
        return -3;
    }

    auto entry = reinterpret_cast<mercan_plugin_entry_fn_v1>(symbol);
    const mercan_plugin_v1 * descriptor = entry();
    if (!valid_descriptor(descriptor)) {
        g_plugin_error = "plugin descriptor is incompatible with Mercan Plugin ABI v1";
        auto failed = std::make_unique<loaded_plugin>();
        failed->path = path;
        failed->name = "<incompatible>";
        failed->handle = handle;
        g_plugins.emplace_back(std::move(failed));
        return -4;
    }

    char error[512] = {};
    const int rc = descriptor->init(&HOST_V1, error, sizeof(error));
    if (rc != 0) {
        g_plugin_error = "plugin init failed";
        if (error[0]) g_plugin_error += ": " + std::string(error);
        // Keep the library loaded because init may have registered callback-backed descriptors.
        auto failed = std::make_unique<loaded_plugin>();
        failed->path = path;
        failed->name = descriptor->name;
        failed->version = descriptor->version ? descriptor->version : "";
        failed->handle = handle;
        g_plugins.emplace_back(std::move(failed));
        return -5;
    }

    auto item = std::make_unique<loaded_plugin>();
    item->path = path;
    item->name = descriptor->name;
    item->version = descriptor->version ? descriptor->version : "";
    item->handle = handle;
    g_plugins.emplace_back(std::move(item));
    return 0;
}

size_t mercan_plugin_count_v1(void) {
    std::lock_guard<std::mutex> lock(g_plugin_mutex);
    return g_plugins.size();
}

const char * mercan_plugin_name_v1(size_t index) {
    std::lock_guard<std::mutex> lock(g_plugin_mutex);
    return index < g_plugins.size() ? g_plugins[index]->name.c_str() : nullptr;
}

const char * mercan_plugin_version_v1(size_t index) {
    std::lock_guard<std::mutex> lock(g_plugin_mutex);
    return index < g_plugins.size() ? g_plugins[index]->version.c_str() : nullptr;
}

const char * mercan_plugin_path_v1(size_t index) {
    std::lock_guard<std::mutex> lock(g_plugin_mutex);
    return index < g_plugins.size() ? g_plugins[index]->path.c_str() : nullptr;
}

const char * mercan_plugin_last_error_v1(void) {
    return g_plugin_error.c_str();
}

} // extern "C"
