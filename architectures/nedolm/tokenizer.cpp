#include "mercan_tokenizer.h"
#include "mercan_internal.hpp"

#include <cstdio>
#include <cstring>

namespace {

int ndsurf004_probe(const mercan_metadata_v1 * metadata) {
    if (!metadata || !metadata->get_string) return -1;
    const char * explicit_type = metadata->get_string(metadata, "mercan.tokenizer.type");
    if (explicit_type) {
        return (std::strcmp(explicit_type, "ndsurf004") == 0 || std::strcmp(explicit_type, "NDSRF004") == 0) ? 100 : 0;
    }
    const char * arch = metadata->get_string(metadata, "general.architecture");
    return arch && std::strcmp(arch, "nedolm") == 0 ? 50 : 0;
}

int ndsurf004_validate(const mercan_metadata_v1 * metadata, char * error, size_t error_capacity) {
    if (!metadata || !metadata->get_string) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "metadata view unavailable");
        return -1;
    }
    const char * sha = metadata->get_string(metadata, "mercan.tokenizer.surface_vocab_sha256");
    if (!sha) sha = metadata->get_string(metadata, "nedolm.vocab_sha256");
    if (!sha || !*sha) {
        if (error && error_capacity) std::snprintf(error, error_capacity, "NDSRF004 tokenizer SHA256 metadata is missing");
        return -2;
    }
    return 0;
}

const mercan_tokenizer_v1 NDSRF004 = {
    MERCAN_TOKENIZER_ABI_VERSION,
    sizeof(mercan_tokenizer_v1),
    "ndsurf004",
    "NDSRF004",
    MERCAN_TOKENIZER_BACKEND_MANAGED | MERCAN_TOKENIZER_BUILTIN,
    ndsurf004_probe,
    ndsurf004_validate,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}

void mercan_register_ndsurf004_tokenizer(void) {
    (void) mercan_tokenizer_register_v1(&NDSRF004);
}
