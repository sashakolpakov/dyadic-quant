# TODO

- Revisit convnet Level 2 separately from the LLM repro suite. The current
  ResNet path is slower for architectural reasons: many smaller convolution
  calls and full-model call granularity do not exercise the same AVX-friendly
  GEMM shapes as Qwen. Recover the last runner state that still included those
  steps from commit `acc3f13` when restarting that work.
