# LLM Repro Status

Current push-button replication is LLM-focused. The default runner exercises
Qwen Level 1 materialized evidence and Level 2 native CPU dyop evidence, with
results split under `results/level1/<run-id>/` and `results/level2/<run-id>/`.

## Remote Smoke

Remote host:

- GPU: NVIDIA A10
- CPU: 30 vCPUs, Intel Xeon Platinum 8358
- Torch: 2.6.0+cu124
- Run source: `results/level2/remote-level2-llm-repro-smoke/`

Qwen Level 2 native quality evidence:

| Bits | Compression vs source | Perplexity | PPL ratio | Agreement | ARC | Wikitext tok/s |
| ---: | --------------------: | ---------: | --------: | --------: | --: | -------------: |
| 4 | 7.98x | 114.824 | 3.698 | 0.441 | 0.45 | 115.4 |
| 5 | 6.39x | 41.223 | 1.328 | 0.732 | 0.50 | 129.1 |
| 6 | 5.33x | 34.316 | 1.105 | 0.841 | 0.50 | 138.4 |
| 8 | 4.00x | 31.922 | 1.028 | 0.916 | 0.45 | 148.6 |

Qwen native kernel evidence:

| Shape | Torch materialized ms | Native dyop ms | Speedup |
| --- | ---: | ---: | ---: |
| `gemv_qwen_mlp` | 0.568 | 0.168 | 3.39x |
| `gemm_qwen_seq` | 0.706 | 0.172 | 4.11x |
| `output_projection_gemv` | 32.494 | 2.263 | 14.36x |
| `output_projection` | 100.159 | 4.251 | 23.56x |
| `qwen_vocab_width` | 0.121 | 0.083 | 1.47x |

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
