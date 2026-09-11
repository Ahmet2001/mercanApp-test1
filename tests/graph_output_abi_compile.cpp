#include "mercan_graph.h"
#include <cstddef>

int main() {
    static_assert(offsetof(mercan_graph_api_v1, set_output) > offsetof(mercan_graph_api_v1, self_attention),
                  "set_output must be append-only");
    static_assert(offsetof(mercan_graph_api_v1, finalize) > offsetof(mercan_graph_api_v1, set_output),
                  "finalize must be append-only");
    static_assert(MERCAN_GRAPH_OUTPUT_EMBEDDING_V1 != MERCAN_GRAPH_OUTPUT_LOGITS_V1,
                  "output kinds must remain distinct");
    return 0;
}
