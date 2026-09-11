#include "mercan_graph.h"
#include <cassert>
#include <cstddef>

int main() {
    static_assert(MERCAN_GRAPH_ABI_VERSION == 1u, "Graph ABI version changed unexpectedly");
    static_assert(offsetof(mercan_graph_api_v1, self_attention) > offsetof(mercan_graph_api_v1, rope_ext),
                  "self_attention must remain an append-only Graph ABI v1 extension");
    mercan_graph_api_v1 api{};
    api.abi_version = MERCAN_GRAPH_ABI_VERSION;
    api.struct_size = sizeof(api);
    assert(MERCAN_GRAPH_API_HAS_V1(&api, self_attention) == 0);
    return 0;
}
