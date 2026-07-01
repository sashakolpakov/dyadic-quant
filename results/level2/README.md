# Level 2 Kernel Artifacts

Store native dyadic execution benchmark outputs here.

Existing top-level result CSV/JSON files are Level 1 quantization-quality
artifacts and should remain separate.

Generated smoke artifact:

- `native_dyop_smoke.json`: Level 2 native dyop forward comparison against the
  Level 1 materialized baseline for the tiny mixed-op smoke model.
- `native_dyop_prefix_sweep_results.csv`: multi-prefix Level 2 native dyop
  comparison rows emitted from a reloaded packed artifact.
- `native_dyop_prefix_sweep_metadata.json`: metadata for the multi-prefix
  validation run.
- `native_dyop_prefix_sweep.dyadic.pt`: packed Level 1 artifact consumed by the
  multi-prefix Level 2 validation.
