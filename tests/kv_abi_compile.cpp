#include "mercan_kv.h"
#include <cassert>

int main() {
    static_assert(MERCAN_KV_ABI_VERSION == 1u, "KV ABI changed unexpectedly");
    static_assert(MERCAN_KV_CACHE_NONE_V1 == 0, "zero must remain the invalid KV handle");
    mercan_kv_api_v1 api{};
    api.abi_version = MERCAN_KV_ABI_VERSION;
    api.struct_size = sizeof(api);
    mercan_kv_resolver_v1 resolver{};
    resolver.abi_version = MERCAN_KV_ABI_VERSION;
    resolver.struct_size = sizeof(resolver);
    resolver.api = &api;
    assert(mercan_kv_resolver_valid_v1(&resolver));
    return 0;
}
