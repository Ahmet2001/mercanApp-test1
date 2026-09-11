#include "mercan_arch.h"
#include "mercan_plugin.h"

#include <cassert>
#include <cstring>
#include <iostream>

int main(int argc, char ** argv) {
    assert(argc == 2);
    const int rc = mercan_plugin_load_v1(argv[1]);
    if (rc != 0) {
        std::cerr << mercan_plugin_last_error_v1() << "\n";
        return 2;
    }
    assert(mercan_plugin_count_v1() >= 1);
    const mercan_architecture_v1 * anka = mercan_arch_find_v1("anka");
    assert(anka != nullptr);
    assert(std::strcmp(anka->display_name, "Anka external demo architecture") == 0);
    assert(mercan_plugin_load_v1(argv[1]) == 1); // idempotent path load
    std::cout << "PLUGIN_LOADER_OK " << mercan_plugin_name_v1(0) << "\n";
    return 0;
}
