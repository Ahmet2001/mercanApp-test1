# Anka external architecture plugin

This is a minimal out-of-tree-style Mercan Plugin ABI v1 example. It builds as `libmercan_arch_anka.so` on Linux and registers the `anka` architecture through host callbacks. It intentionally does not claim a compiled inference backend; its purpose is to prove dynamic discovery/registration without linking against Mercan internals.
