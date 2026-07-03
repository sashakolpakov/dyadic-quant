# TODO

- Build a native Qwen depth runner for Level 2. The current native path replaces
  Linear and Embedding modules inside the Transformers/PyTorch graph, so the
  wide AVX kernels are fast but the full network still pays repeated Python
  dispatch and tensor handoff costs across 24 layers. The next design target is
  a native block boundary that keeps hidden states, RMSNorm, QKV/O projections,
  attention intermediates, MLP projections, and residual adds inside C++ for one
  layer or a group of layers.
- Revisit convnet Level 2 separately from the LLM repro suite. The current
  ResNet path is slower for architectural reasons: many smaller convolution
  calls and full-model call granularity do not exercise the same AVX-friendly
  GEMM shapes as Qwen. Recover the last runner state that still included those
  steps from commit `acc3f13` when restarting that work.
