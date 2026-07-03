#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

LEVEL="all"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULTS_ROOT="results"
MODEL_DIR="data/checkpoints/Qwen2.5-0.5B-Instruct"
DATA_DIR="data/llm_eval"
BITS=(4 5 6 8)
THREADS="${DYOP_CPU_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
PREPARE_DATA=1
OLLAMA_MODEL="qwen2.5:0.5b"
SKIP_OLLAMA=0
REQUIRE_OLLAMA=0
SKIP_TEXTUAL=0
REQUIRE_TEXTUAL=0
NO_JUDGE=0
EMBED_MODEL="nomic-embed-text"
JUDGE_BACKEND="ollama"
JUDGE_MODEL="${DYADIC_JUDGE_MODEL:-gemma3:4b}"
JUDGE_TIMEOUT=180
JUDGE_BATCH_SIZE=1
LEVEL1_GENERATIONS=""
QWEN_MLP_BACKEND="native-cpu-plan"
QWEN_NORM_BACKEND="native-cpu"
AUDIT_STRICT=0
MIN_QWEN_AGREEMENT=""
MAX_QWEN_PERPLEXITY_RATIO=""
QUICK=0
DRY_RUN=0
PYTHON_BIN="${PYTHON:-python3}"

usage() {
  cat <<'EOF'
Usage: ./dyadic-experiments.sh --level 1|2|all [options]

Remote-first replication runner. It writes separated outputs under:
  results/level1/<run-id>/
  results/level2/<run-id>/

Options:
  --level 1|2|all              Experiment level to run.
  --run-id NAME                Stable result directory name.
  --results-root DIR           Root containing level1/ and level2/ (default: results).
  --model-dir DIR              Local Qwen checkpoint directory.
  --data-dir DIR               Local LLM eval data directory.
  --bits "4 5 6 8"             Bit depths to evaluate.
  --threads N                  Native dyop worker threads.
  --no-prepare-data            Do not download missing public checkpoints/datasets.
  --ollama-model NAME          Ollama model name (default: qwen2.5:0.5b).
  --skip-ollama                Do not run the Ollama ARC baseline.
  --require-ollama             Fail if the Ollama API is not reachable.
  --skip-textual               Do not run textual cosine/judge evidence.
  --require-textual            Fail if textual evidence cannot run.
  --embed-model NAME           Ollama embedding model (default: nomic-embed-text).
  --judge-backend claude|ollama
                               LLM judge backend (default: ollama).
  --judge-model NAME           LLM judge model (default: gemma3:4b for Ollama).
  --judge-timeout SECONDS      Per-prompt judge timeout.
  --judge-batch-size N         Variants per judge call.
  --no-judge                   Compute lexical/cosine metrics without judge calls.
  --level1-generations FILE    Level 1 generations file used to seed Level 2 text.
  --qwen-mlp-backend torch|native-cpu-plan
                               Level 2 Qwen MLP backend (default: native-cpu-plan).
  --qwen-norm-backend torch|native-cpu
                               Level 2 Qwen RMSNorm backend (default: native-cpu).
  --strict-audit               Exit nonzero when the native evidence audit has issues.
  --min-qwen-agreement VALUE   Audit threshold for next-token agreement.
  --max-qwen-perplexity-ratio VALUE
                               Audit threshold relative to source perplexity.
  --quick                      Small smoke-sized run.
  --dry-run                    Print commands without executing them.
  -h, --help                   Show this help.

Docker A10 example:
  docker run --gpus all --ipc=host --rm \
    -v "$PWD/data:/workspace/data" \
    -v "$PWD/results:/workspace/results" \
    -v dyadic-ollama:/root/.ollama \
    dyadic-experiments --level all --threads 30
EOF
}

parse_bits() {
  local raw="$1"
  raw="${raw//,/ }"
  read -r -a BITS <<<"$raw"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --level)
      LEVEL="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --bits)
      parse_bits "$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --no-prepare-data)
      PREPARE_DATA=0
      shift
      ;;
    --ollama-model)
      OLLAMA_MODEL="$2"
      shift 2
      ;;
    --skip-ollama)
      SKIP_OLLAMA=1
      shift
      ;;
    --require-ollama)
      REQUIRE_OLLAMA=1
      shift
      ;;
    --skip-textual)
      SKIP_TEXTUAL=1
      shift
      ;;
    --require-textual)
      REQUIRE_TEXTUAL=1
      shift
      ;;
    --embed-model)
      EMBED_MODEL="$2"
      shift 2
      ;;
    --judge-backend)
      JUDGE_BACKEND="$2"
      shift 2
      ;;
    --judge-model)
      JUDGE_MODEL="$2"
      shift 2
      ;;
    --judge-timeout)
      JUDGE_TIMEOUT="$2"
      shift 2
      ;;
    --judge-batch-size)
      JUDGE_BATCH_SIZE="$2"
      shift 2
      ;;
    --no-judge)
      NO_JUDGE=1
      shift
      ;;
    --level1-generations)
      LEVEL1_GENERATIONS="$2"
      shift 2
      ;;
    --qwen-mlp-backend)
      QWEN_MLP_BACKEND="$2"
      shift 2
      ;;
    --qwen-norm-backend)
      QWEN_NORM_BACKEND="$2"
      shift 2
      ;;
    --strict-audit)
      AUDIT_STRICT=1
      shift
      ;;
    --min-qwen-agreement)
      MIN_QWEN_AGREEMENT="$2"
      shift 2
      ;;
    --max-qwen-perplexity-ratio)
      MAX_QWEN_PERPLEXITY_RATIO="$2"
      shift 2
      ;;
    --quick)
      QUICK=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$LEVEL" in
  1|2|all) ;;
  *)
    echo "--level must be 1, 2, or all" >&2
    exit 2
    ;;
esac
case "$QWEN_MLP_BACKEND" in
  torch|native-cpu-plan) ;;
  *)
    echo "--qwen-mlp-backend must be torch or native-cpu-plan" >&2
    exit 2
    ;;
esac
case "$QWEN_NORM_BACKEND" in
  torch|native-cpu) ;;
  *)
    echo "--qwen-norm-backend must be torch or native-cpu" >&2
    exit 2
    ;;
esac

LEVEL1_DIR="$RESULTS_ROOT/level1/$RUN_ID"
LEVEL2_DIR="$RESULTS_ROOT/level2/$RUN_ID"
LOG_ROOT="$LEVEL2_DIR/logs"
if [[ "$LEVEL" == "1" ]]; then
  LOG_ROOT="$LEVEL1_DIR/logs"
fi
if [[ "$LEVEL" == "1" || "$LEVEL" == "all" ]]; then
  mkdir -p "$LEVEL1_DIR"
fi
if [[ "$LEVEL" == "2" || "$LEVEL" == "all" ]]; then
  mkdir -p "$LEVEL2_DIR"
fi
mkdir -p "$LOG_ROOT"

QWEN_MAX_TOKENS=8192
QWEN_SEQUENCE_LENGTH=256
QWEN_ARC_LIMIT=100
QWEN_WARMUP_REPEATS=2
KERNEL_REPEATS=50
TEXT_ARC_COUNT=20
TEXT_WIKITEXT_COUNT=10
TEXT_MAX_NEW_TOKENS=128
DEPTH_PROFILE_REPEATS=2
DEPTH_PROFILE_SEQUENCE_LENGTHS=(1 8 64 256)

if [[ "$QUICK" == "1" ]]; then
  QWEN_MAX_TOKENS=1024
  QWEN_SEQUENCE_LENGTH=128
  QWEN_ARC_LIMIT=20
  QWEN_WARMUP_REPEATS=0
  KERNEL_REPEATS=5
  TEXT_ARC_COUNT=1
  TEXT_WIKITEXT_COUNT=1
  TEXT_MAX_NEW_TOKENS=16
  DEPTH_PROFILE_REPEATS=1
  DEPTH_PROFILE_SEQUENCE_LENGTHS=(1 8)
fi

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_step() {
  local name="$1"
  shift
  local log="$LOG_ROOT/${name}.log"
  echo "[$(date -u +%H:%M:%S)] $name"
  print_command "$@" | tee "$log"
  if [[ "$DRY_RUN" == "0" ]]; then
    "$@" 2>&1 | tee -a "$log"
  fi
}

prepare_data() {
  if [[ "$PREPARE_DATA" == "0" ]]; then
    echo "Data preparation disabled by request."
    return
  fi
  if [[ ! -f "$MODEL_DIR/model.safetensors" ]]; then
    run_step prepare_qwen_checkpoint \
      "$PYTHON_BIN" -c 'from huggingface_hub import snapshot_download; import sys; snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", local_dir=sys.argv[1], allow_patterns=["*.json", "*.safetensors", "tokenizer.*", "merges.txt", "vocab.json"])' \
        "$MODEL_DIR"
  fi
  if [[ ! -f "$DATA_DIR/wikitext2_test.txt" || ! -f "$DATA_DIR/arc_easy.json" ]]; then
    echo "Missing LLM eval files in $DATA_DIR; copy wikitext2_test.txt and arc_easy.json or pass --data-dir." >&2
    exit 1
  fi
}

record_environment() {
  local log="$LOG_ROOT/environment.log"
  {
    date -u
    uname -a
    echo "run_id=$RUN_ID"
    echo "level=$LEVEL"
    echo "threads=$THREADS"
    echo "bits=${BITS[*]}"
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
    command -v lscpu >/dev/null 2>&1 && lscpu || true
    "$PYTHON_BIN" - <<'PY' || true
import platform
import torch
print("python", platform.python_version())
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_name", torch.cuda.get_device_name(0))
print("mps_available", bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()))
PY
  } >"$log" 2>&1
  cat "$log"
}

ollama_available() {
  command -v curl >/dev/null 2>&1 &&
    curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1
}

textual_available() {
  if [[ "$SKIP_TEXTUAL" == "1" ]]; then
    echo "Skipping textual evidence by request."
    return 1
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  if ! ollama_available; then
    if [[ "$REQUIRE_TEXTUAL" == "1" ]]; then
      echo "Textual evidence needs Ollama at http://127.0.0.1:11434 for embeddings." >&2
      exit 1
    fi
    echo "Ollama API is not reachable; continuing without textual evidence."
    return 1
  fi
  return 0
}

judge_model_arg() {
  if [[ -n "$JUDGE_MODEL" ]]; then
    echo "$JUDGE_MODEL"
  elif [[ "$JUDGE_BACKEND" == "ollama" ]]; then
    echo "$OLLAMA_MODEL"
  else
    echo ""
  fi
}

maybe_run_ollama() {
  local output_dir="$1"
  local prefix="$2"
  if [[ "$SKIP_OLLAMA" == "1" ]]; then
    echo "Skipping Ollama baseline by request."
    return
  fi
  if ! ollama_available; then
    if [[ "$REQUIRE_OLLAMA" == "1" ]]; then
      echo "Ollama API is not reachable at http://127.0.0.1:11434" >&2
      exit 1
    fi
    echo "Ollama API is not reachable; continuing without the Ollama baseline."
    return
  fi
  mkdir -p "$output_dir/ollama"
  run_step "${prefix}_ollama_arc" \
    "$PYTHON_BIN" experiments/run_ollama_llm.py \
      --model "$OLLAMA_MODEL" \
      --data-dir "$DATA_DIR" \
      --arc-limit "$QWEN_ARC_LIMIT" \
      --result-prefix "${prefix}_ollama_${OLLAMA_MODEL//[^A-Za-z0-9_]/_}" \
      --output-dir "$output_dir/ollama"
}

run_level1_textual() {
  local out="$LEVEL1_DIR/textual"
  local generations="$out/qwen25_generations.json"
  local judge_model
  judge_model="$(judge_model_arg)"
  mkdir -p "$out"
  run_step level1_qwen_textual_generation \
    "$PYTHON_BIN" experiments/level1/run_textual_generation.py \
      --model-dir "$MODEL_DIR" \
      --data-dir "$DATA_DIR" \
      --variant bf16_source \
      --dyadic-prefix dyadic \
      --bits "${BITS[@]}" \
      --arc-count "$TEXT_ARC_COUNT" \
      --wikitext-count "$TEXT_WIKITEXT_COUNT" \
      --max-new-tokens "$TEXT_MAX_NEW_TOKENS" \
      --generations-file "$generations"
  local compare=(
    "$PYTHON_BIN" experiments/level1/compare_generations.py
    --generations-file "$generations"
    --source-variant bf16_source
    --embed-model "$EMBED_MODEL"
    --judge-backend "$JUDGE_BACKEND"
    --judge-model "$judge_model"
    --judge-timeout "$JUDGE_TIMEOUT"
    --judge-batch-size "$JUDGE_BATCH_SIZE"
    --output-dir "$out"
  )
  if [[ "$NO_JUDGE" == "1" ]]; then
    compare+=(--no-judge)
  fi
  run_step level1_qwen_textual_compare "${compare[@]}"
}

run_level2_textual() {
  local out="$LEVEL2_DIR/textual"
  local generations="$out/qwen25_dyop_generations.json"
  local source_variant="float32_source"
  local source_generations="$LEVEL1_GENERATIONS"
  local judge_model
  judge_model="$(judge_model_arg)"
  mkdir -p "$out"
  if [[ -z "$source_generations" && -f "$LEVEL1_DIR/textual/qwen25_generations.json" ]]; then
    source_generations="$LEVEL1_DIR/textual/qwen25_generations.json"
  fi
  if [[ -z "$source_generations" && "$LEVEL" == "all" ]]; then
    source_generations="$LEVEL1_DIR/textual/qwen25_generations.json"
  fi
  if [[ -n "$source_generations" && ( -f "$source_generations" || "$DRY_RUN" == "1" ) ]]; then
    source_variant="bf16_source"
    run_step level2_qwen_textual_seed_source \
      "$PYTHON_BIN" experiments/level2/seed_qwen_textual_reference.py \
        --source-generations "$source_generations" \
        --source-variant "$source_variant" \
        --output "$generations"
    run_step level2_qwen_textual_generation \
      env DYOP_CPU_THREADS="$THREADS" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      "$PYTHON_BIN" experiments/level2/run_qwen_textual_generation.py \
        --model-dir "$MODEL_DIR" \
        --data-dir "$DATA_DIR" \
        --bits "${BITS[@]}" \
        --arc-count "$TEXT_ARC_COUNT" \
        --wikitext-count "$TEXT_WIKITEXT_COUNT" \
        --max-new-tokens "$TEXT_MAX_NEW_TOKENS" \
        --source-variant "$source_variant" \
        --dyop-prefix dyop_native \
        --skip-source-generation \
        --load-dyadic "$LEVEL2_DIR/qwen_native/qwen25_level2_native_cpu.dyadic.pt" \
        --generations-file "$generations" \
        --qwen-mlp-backend "$QWEN_MLP_BACKEND" \
        --qwen-norm-backend "$QWEN_NORM_BACKEND" \
        --skip-speed-gate-check
  else
    run_step level2_qwen_textual_generation \
      env DYOP_CPU_THREADS="$THREADS" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      "$PYTHON_BIN" experiments/level2/run_qwen_textual_generation.py \
        --model-dir "$MODEL_DIR" \
        --data-dir "$DATA_DIR" \
        --bits "${BITS[@]}" \
        --arc-count "$TEXT_ARC_COUNT" \
        --wikitext-count "$TEXT_WIKITEXT_COUNT" \
        --max-new-tokens "$TEXT_MAX_NEW_TOKENS" \
        --source-variant "$source_variant" \
        --dyop-prefix dyop_native \
        --load-dyadic "$LEVEL2_DIR/qwen_native/qwen25_level2_native_cpu.dyadic.pt" \
        --generations-file "$generations" \
        --qwen-mlp-backend "$QWEN_MLP_BACKEND" \
        --qwen-norm-backend "$QWEN_NORM_BACKEND" \
        --skip-speed-gate-check
  fi
  local compare=(
    "$PYTHON_BIN" experiments/level1/compare_generations.py
    --generations-file "$generations"
    --source-variant "$source_variant"
    --embed-model "$EMBED_MODEL"
    --judge-backend "$JUDGE_BACKEND"
    --judge-model "$judge_model"
    --judge-timeout "$JUDGE_TIMEOUT"
    --judge-batch-size "$JUDGE_BATCH_SIZE"
    --output-dir "$out"
  )
  if [[ "$NO_JUDGE" == "1" ]]; then
    compare+=(--no-judge)
  fi
  run_step level2_qwen_textual_compare "${compare[@]}"
  if [[ "$source_variant" == "bf16_source" && ( -f "$source_generations" || "$DRY_RUN" == "1" ) ]]; then
    run_step level2_qwen_textual_level1_audit \
      "$PYTHON_BIN" experiments/level2/compare_qwen_level1_level2_textual.py \
        --level1-generations "$source_generations" \
        --level2-generations "$generations" \
        --level1-summary "$(dirname "$source_generations")/qwen25_textual_summary.csv" \
        --level1-comparison "$(dirname "$source_generations")/qwen25_textual_comparison.csv" \
        --level1-metadata "$(dirname "$source_generations")/qwen25_textual_metadata.json" \
        --level2-summary "$out/qwen25_textual_summary.csv" \
        --level2-comparison "$out/qwen25_textual_comparison.csv" \
        --level2-metadata "$out/qwen25_textual_metadata.json" \
        --source-variant "$source_variant" \
        --output-dir "$LEVEL2_DIR/textual_audit"
  fi
}

run_level1() {
  local out="$LEVEL1_DIR"
  mkdir -p "$out/qwen"
  maybe_run_ollama "$out" "level1"
  run_step level1_qwen_materialized_gpu \
    "$PYTHON_BIN" experiments/level1/run_qwen_dyadic.py \
      --model-dir "$MODEL_DIR" \
      --data-dir "$DATA_DIR" \
      --bits "${BITS[@]}" \
      --max-tokens "$QWEN_MAX_TOKENS" \
      --sequence-length "$QWEN_SEQUENCE_LENGTH" \
      --arc-limit "$QWEN_ARC_LIMIT" \
      --warmup-repeats "$QWEN_WARMUP_REPEATS" \
      --skip-generation \
      --variant-name qwen25_level1_materialized_gpu \
      --output-dir "$out/qwen"
  if textual_available; then
    run_level1_textual
  fi
}

run_level2() {
  local out="$LEVEL2_DIR"
  mkdir -p "$out/qwen_native" "$out/kernels" "$out/depth"
  maybe_run_ollama "$out" "level2"
  run_step level2_build_native_cpu \
    "$PYTHON_BIN" experiments/level2/build_native_cpu.py --force
  run_step level2_qwen_native_kernels \
    env DYOP_CPU_THREADS="$THREADS" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON_BIN" experiments/level2/benchmark_native_kernels.py \
      --bits 6 \
      --repeats "$KERNEL_REPEATS" \
      --torch-threads 1 \
      --dyop-threads "$THREADS" \
      --embedding-dyop-threads 1 \
      --ops linear embedding \
      --output "$out/kernels/qwen_native_kernels.csv"
  run_step level2_qwen_native_cpu \
    env DYOP_CPU_THREADS="$THREADS" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON_BIN" experiments/level1/run_qwen_dyadic.py \
      --model-dir "$MODEL_DIR" \
      --data-dir "$DATA_DIR" \
      --bits "${BITS[@]}" \
      --max-tokens "$QWEN_MAX_TOKENS" \
      --sequence-length "$QWEN_SEQUENCE_LENGTH" \
      --arc-limit "$QWEN_ARC_LIMIT" \
      --warmup-repeats 0 \
      --skip-generation \
      --execution-backend level2-native \
      --level2-linear-backend native-cpu \
      --level2-embedding-backend native-cpu \
      --qwen-mlp-backend "$QWEN_MLP_BACKEND" \
      --qwen-norm-backend "$QWEN_NORM_BACKEND" \
      --skip-speed-gate-check \
      --variant-name qwen25_level2_native_cpu \
      --save-dyadic "$out/qwen_native/qwen25_level2_native_cpu.dyadic.pt" \
      --output-dir "$out/qwen_native"
  run_step level2_qwen_depth_profile \
    env DYOP_CPU_THREADS="$THREADS" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON_BIN" experiments/level2/profile_qwen_depth.py \
      --model-dir "$MODEL_DIR" \
      --bits 6 \
      --sequence-lengths "${DEPTH_PROFILE_SEQUENCE_LENGTHS[@]}" \
      --repeats "$DEPTH_PROFILE_REPEATS" \
      --threads "$THREADS" \
      --qwen-mlp-backend "$QWEN_MLP_BACKEND" \
      --qwen-norm-backend "$QWEN_NORM_BACKEND" \
      --load-dyadic "$out/qwen_native/qwen25_level2_native_cpu.dyadic.pt" \
      --output "$out/depth/qwen_depth_profile.csv"
  run_step level2_qwen_depth_backend_sweep \
    env DYOP_CPU_THREADS="$THREADS" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON_BIN" experiments/level2/sweep_qwen_depth_backends.py \
      --model-dir "$MODEL_DIR" \
      --bits 6 \
      --sequence-lengths "${DEPTH_PROFILE_SEQUENCE_LENGTHS[@]}" \
      --batch-sizes 1 \
      --repeats "$DEPTH_PROFILE_REPEATS" \
      --threads "$THREADS" \
      --load-dyadic "$out/qwen_native/qwen25_level2_native_cpu.dyadic.pt" \
      --output "$out/depth/qwen_depth_backend_sweep.csv" \
      --keep-raw-dir "$out/depth/backend_sweep_raw"
  if textual_available; then
    run_level2_textual
  fi
  local audit=(
    "$PYTHON_BIN" experiments/level2/audit_native_evidence.py
    --level2-dir "$out"
    --level1-dir "$LEVEL1_DIR"
    --output-dir "$out/evidence"
  )
  if [[ "$AUDIT_STRICT" == "1" ]]; then
    audit+=(--strict)
  fi
  if [[ -n "$MIN_QWEN_AGREEMENT" ]]; then
    audit+=(--min-qwen-agreement "$MIN_QWEN_AGREEMENT")
  fi
  if [[ -n "$MAX_QWEN_PERPLEXITY_RATIO" ]]; then
    audit+=(--max-qwen-perplexity-ratio "$MAX_QWEN_PERPLEXITY_RATIO")
  fi
  run_step level2_native_evidence_audit "${audit[@]}"
}

write_manifest() {
  local path="$LOG_ROOT/run_manifest.json"
  "$PYTHON_BIN" - "$path" <<PY
import json
import pathlib
payload = {
    "run_id": "$RUN_ID",
    "level": "$LEVEL",
    "quick": bool(int("$QUICK")),
    "threads": int("$THREADS"),
    "bits": [int(x) for x in "${BITS[*]}".split()],
    "prepare_data": bool(int("$PREPARE_DATA")),
    "level1_dir": "$LEVEL1_DIR",
    "level2_dir": "$LEVEL2_DIR",
    "model_dir": "$MODEL_DIR",
    "data_dir": "$DATA_DIR",
    "qwen_mlp_backend": "$QWEN_MLP_BACKEND",
    "qwen_norm_backend": "$QWEN_NORM_BACKEND",
    "audit_strict": bool(int("$AUDIT_STRICT")),
    "audit_thresholds": {
        "min_qwen_agreement": "$MIN_QWEN_AGREEMENT" or None,
        "max_qwen_perplexity_ratio": "$MAX_QWEN_PERPLEXITY_RATIO" or None,
    },
    "skip_textual": bool(int("$SKIP_TEXTUAL")),
    "require_textual": bool(int("$REQUIRE_TEXTUAL")),
    "embed_model": "$EMBED_MODEL",
    "judge_backend": "$JUDGE_BACKEND",
    "judge_model": "$(judge_model_arg)",
    "no_judge": bool(int("$NO_JUDGE")),
}
path = pathlib.Path("$path")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n")
print(path)
PY
}

create_bundle() {
  local bundle_dir="$LEVEL2_DIR"
  local include=()
  if [[ "$LEVEL" == "1" ]]; then
    bundle_dir="$LEVEL1_DIR"
    include=("$LEVEL1_DIR")
  elif [[ "$LEVEL" == "2" ]]; then
    include=("$LEVEL2_DIR")
  else
    include=("$LEVEL1_DIR" "$LEVEL2_DIR")
  fi
  local tmp="${TMPDIR:-/tmp}/dyadic-experiments-${RUN_ID}.tar.gz"
  local final="$bundle_dir/dyadic-experiments-${RUN_ID}.tar.gz"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "Would create bundle: $final"
    return
  fi
  tar -czf "$tmp" "${include[@]}"
  mv "$tmp" "$final"
  echo "Bundle: $final"
}

record_environment
prepare_data
write_manifest

case "$LEVEL" in
  1)
    run_level1
    ;;
  2)
    run_level2
    ;;
  all)
    run_level1
    run_level2
    ;;
esac

create_bundle
if [[ "$LEVEL" == "1" || "$LEVEL" == "all" ]]; then
  echo "Level 1 output: $LEVEL1_DIR"
fi
if [[ "$LEVEL" == "2" || "$LEVEL" == "all" ]]; then
  echo "Level 2 output: $LEVEL2_DIR"
fi
