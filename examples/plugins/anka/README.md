# Anka external architecture plugin

`mercan_arch_anka` is a real loadable Mercan Plugin ABI v1 module. It registers the
`anka` architecture and provides a Graph Callback ABI v1 implementation using only
public opaque handles: token embedding lookup -> output projection -> logits/finalize.

This proves an external `.so` can build a graph through Mercan Tensor/Graph ABI without
linking to llama.cpp or touching Mercan core. The generic model/backend adapter that
feeds real `.mercan` tensors/inputs to arbitrary external architectures is the next layer;
Anka is intentionally a minimal graph callback proof, not a production language model.
