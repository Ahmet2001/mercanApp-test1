#include "mercan_arch.h"
#include "mercan_tokenizer.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>

static int never_probe(const mercan_metadata_v1 *) { return 99; }
static int never_validate(const mercan_metadata_v1 *, char *, size_t) { return 99; }
static int32_t never_encode(void *, const char *, size_t, bool, bool, mercan_token *, int32_t) { return 99; }

int main() {
    mercan_architecture_v1 arch{};
    arch.abi_version = MERCAN_ARCH_ABI_VERSION;
    arch.struct_size = static_cast<uint32_t>(MERCAN_ARCHITECTURE_V1_BASE_SIZE);
    arch.name = "prefix_test_arch";
    arch.display_name = "Prefix Test Architecture";
    arch.default_tokenizer = "prefix_test_tok";
    arch.flags = 0;
    // Deliberately populate fields outside struct_size. A compatible registry
    // must not copy or expose them.
    arch.probe = never_probe;
    arch.validate = never_validate;

    assert(mercan_arch_register_v1(&arch) == 0);
    const auto * got_arch = mercan_arch_find_v1("prefix_test_arch");
    assert(got_arch != nullptr);
    assert(got_arch->struct_size == MERCAN_ARCHITECTURE_V1_BASE_SIZE);
    assert(got_arch->probe == nullptr);
    assert(got_arch->validate == nullptr);

    mercan_tokenizer_v1 tok{};
    tok.abi_version = MERCAN_TOKENIZER_ABI_VERSION;
    tok.struct_size = static_cast<uint32_t>(MERCAN_TOKENIZER_V1_BASE_SIZE);
    tok.name = "prefix_test_tok";
    tok.display_name = "Prefix Test Tokenizer";
    tok.flags = 0;
    tok.probe = never_probe;
    tok.validate = never_validate;
    tok.encode = never_encode;

    assert(mercan_tokenizer_register_v1(&tok) == 0);
    const auto * got_tok = mercan_tokenizer_find_v1("prefix_test_tok");
    assert(got_tok != nullptr);
    assert(got_tok->struct_size == MERCAN_TOKENIZER_V1_BASE_SIZE);
    assert(got_tok->probe == nullptr);
    assert(got_tok->validate == nullptr);
    assert(got_tok->encode == nullptr);

    std::cout << "SDK_PREFIX_COMPAT_OK\n";
    return 0;
}
