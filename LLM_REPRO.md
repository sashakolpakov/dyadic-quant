# LLM Repro Status

Current push-button replication is LLM-focused. The default runner exercises
Qwen Level 1 materialized evidence and Level 2 native CPU dyop evidence, with
results split under `results/level1/<run-id>/` and `results/level2/<run-id>/`.

## Remote Full Run

Remote host:

- GPU: NVIDIA A10
- CPU: 30 vCPUs, Intel Xeon Platinum 8358
- Torch: 2.6.0+cu124
- Run source: `results/level1/remote-llm-full-20260703/` and
  `results/level2/remote-llm-full-20260703/`
- Local audit: `results/level2/remote-llm-full-20260703/evidence_local/`

The local audit reproduced the remote audit. It is non-strict and reports one
issue: one Level 2 textual judge row is missing. Kernel and Qwen quality
evidence are present.

Qwen Level 2 native quality evidence:

| Bits | Compression vs source | Perplexity | PPL ratio | Agreement | ARC | Wikitext tok/s |
| ---: | --------------------: | ---------: | --------: | --------: | --: | -------------: |
| 4 | 7.98x | 101.942 | 3.708 | 0.443 | 0.30 | 293.1 |
| 5 | 6.39x | 34.724 | 1.263 | 0.717 | 0.43 | 274.1 |
| 6 | 5.33x | 29.482 | 1.072 | 0.838 | 0.49 | 296.9 |
| 8 | 4.00x | 28.102 | 1.022 | 0.923 | 0.51 | 333.2 |

Qwen native kernel evidence:

| Shape | Torch materialized ms | Native dyop ms | Speedup |
| --- | ---: | ---: | ---: |
| `gemv_qwen_mlp` | 0.565 | 0.096 | 5.88x |
| `gemm_qwen_seq` | 0.715 | 0.134 | 5.32x |
| `output_projection_gemv` | 33.480 | 2.508 | 13.35x |
| `output_projection` | 108.869 | 3.776 | 28.83x |
| `qwen_vocab_width` | 0.076 | 0.058 | 1.33x |

Level 1 materialized Qwen evidence for comparison:

| Bits | Perplexity | Agreement | ARC | Wikitext tok/s |
| ---: | ---------: | --------: | --: | -------------: |
| 4 | 102.961 | 0.445 | 0.34 | 11759.2 |
| 5 | 34.856 | 0.716 | 0.43 | 11846.3 |
| 6 | 29.581 | 0.839 | 0.49 | 11121.0 |
| 8 | 28.144 | 0.916 | 0.52 | 11679.2 |

## Depth Throughput

The kernel table proves the wide AVX dyop kernels on representative Qwen
shapes. It does not prove the current full-network execution path is optimal.
Today Level 2 still runs inside the Transformers/PyTorch module graph, so every
layer crosses Python/PyTorch boundaries around native Linear and Embedding
calls. That is the open depth problem.

The runner now writes `results/level2/<run-id>/depth/qwen_depth_profile.csv`.
Use that file to separate:

- wide kernel time, from `results/level2/<run-id>/kernels/qwen_native_kernels.csv`;
- full forward and per-module depth time, from `depth/qwen_depth_profile.csv`;
- quality evidence, from `qwen_native/qwen25_level2_native_cpu_results.csv`.

The next Level 2 optimization target is a native Qwen block runner that keeps
hidden-state tensors and per-layer intermediates inside C++ across RMSNorm,
attention projections, attention reductions, MLP projections, and residual
adds.

## Command

Use the Docker runner on the remote:

```bash
docker run --gpus all --ipc=host --rm \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/results:/workspace/results" \
  -v dyadic-ollama:/root/.ollama \
  dyadic-experiments --level all --threads 30
```

For a fast smoke:

```bash
docker run --gpus all --ipc=host --rm \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/results:/workspace/results" \
  -v dyadic-ollama:/root/.ollama \
  dyadic-experiments --level 2 --quick --threads 30
```
