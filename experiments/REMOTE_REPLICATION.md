# Remote LLM Replication Runner

Use this on the A10 host to run the Qwen Level 1 materialized baseline and
Level 2 native CPU dyop LLM evidence with one command. Outputs remain separated:

- `results/level1/<run-id>/`
- `results/level2/<run-id>/`

Each run writes per-step logs plus CSV/JSON outputs, then creates a single
`dyadic-experiments-<run-id>.tar.gz` bundle for fetching.

For Level 2 runs, inspect:

```text
results/level2/<run-id>/evidence/native_evidence_audit.json
results/level2/<run-id>/evidence/native_evidence_audit.md
results/level2/<run-id>/evidence/qwen_native_evidence.csv
results/level2/<run-id>/evidence/qwen_kernel_evidence.csv
```

The audit summarizes Qwen memory, perplexity, next-token agreement, ARC
accuracy, textual cosine/judge metrics when enabled, and native kernel speed
rows. The audit is informational by default, so the bundle is still created when
a metric is weak or missing.

To make the audit fail the Docker run when a quality threshold is missed, add
`--strict-audit` with explicit cutoffs:

```bash
docker run --gpus all --ipc=host --rm \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/results:/workspace/results" \
  -v dyadic-ollama:/root/.ollama \
  dyadic-experiments --level all --threads 30 --strict-audit \
    --min-qwen-agreement 0.80 \
    --max-qwen-perplexity-ratio 1.20
```

## Build

```bash
docker build -t dyadic-experiments .
```

The image includes Ollama. The entrypoint starts `ollama serve`, pulls the
models needed by the requested command, and then runs `dyadic-experiments.sh`.
Mount `/root/.ollama` as a Docker volume so model pulls are reused.

The runner prepares the Qwen2.5-0.5B-Instruct checkpoint from Hugging Face when
it is missing. Keep `data/` mounted read-write for this. Add
`--no-prepare-data` when the remote data directory is already complete and
should not be modified. The small LLM eval files `wikitext2_test.txt` and
`arc_easy.json` are still expected under `data/llm_eval` or the directory passed
with `--data-dir`.

## One-Command Remote Run

From your local checkout, this does the full remote orchestration:

```bash
./dyadic-remote-experiments.sh ubuntu@A10_HOST --level all --threads 30
```

It will:

- rsync the repo to `/home/ubuntu/dyadic-quant`;
- rsync `data/llm_eval/`;
- build the `dyadic-experiments` Docker image on the remote;
- run the requested Level 1/Level 2 suite in Docker;
- fetch `results/level1/<run-id>/` and `results/level2/<run-id>/` locally;
- rerun the Level 2 audit locally into `results/level2/<run-id>/evidence_local/`.

Useful variants:

```bash
./dyadic-remote-experiments.sh ubuntu@A10_HOST --level 2 --quick --threads 30
./dyadic-remote-experiments.sh ubuntu@A10_HOST --level all --run-id paper-rerun-001
./dyadic-remote-experiments.sh ubuntu@A10_HOST --level all --no-build --threads 30
```

## Run Level 1

```bash
docker run --gpus all --ipc=host --rm \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/results:/workspace/results" \
  -v dyadic-ollama:/root/.ollama \
  dyadic-experiments --level 1 --threads 30
```

## Run Level 2

```bash
docker run --gpus all --ipc=host --rm \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/results:/workspace/results" \
  -v dyadic-ollama:/root/.ollama \
  dyadic-experiments --level 2 --threads 30
```

## Run Both

```bash
docker run --gpus all --ipc=host --rm \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/results:/workspace/results" \
  -v dyadic-ollama:/root/.ollama \
  dyadic-experiments --level all --threads 30
```

## Ollama Baseline

By default the container starts its own Ollama server and pulls:

- `qwen2.5:0.5b` for the ARC baseline;
- `nomic-embed-text` for cosine embeddings;
- `gemma3:4b` as the default Ollama judge.

Set extra or replacement pulls with `DYADIC_OLLAMA_MODELS`, or set
`DYADIC_JUDGE_MODEL` / `--judge-model` to use a different local judge.

```bash
docker run --gpus all --ipc=host --rm \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/results:/workspace/results" \
  -v dyadic-ollama:/root/.ollama \
  -e DYADIC_OLLAMA_MODELS="qwen2.5:0.5b nomic-embed-text llama3.1:8b" \
  dyadic-experiments --level all --threads 30 --judge-model llama3.1:8b
```

To use an Ollama daemon already running on the host, disable the container
server and use host networking:

```bash
docker run --gpus all --ipc=host --network host --rm \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/results:/workspace/results" \
  -e DYADIC_START_OLLAMA=0 \
  dyadic-experiments --level all --threads 30 --require-ollama
```

## Fetch

The script prints the bundle path at the end. From your laptop:

```bash
scp user@A10_HOST:/path/to/dyadic-quant/results/level2/<run-id>/dyadic-experiments-<run-id>.tar.gz .
```

For a Level 1-only run, the bundle is under `results/level1/<run-id>/`.

## Cheap Local Checks

These only validate wiring and command construction:

```bash
bash dyadic-experiments.sh --level 1 --quick --skip-ollama --dry-run
bash dyadic-experiments.sh --level 2 --quick --skip-ollama --dry-run
```
