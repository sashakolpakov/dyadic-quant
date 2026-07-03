#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: ./dyadic-remote-experiments.sh user@host [options passed to dyadic-experiments]

Build and run the LLM-only dyadic repro suite on a remote Docker host, fetch
the result directories back, and run a local Level 2 evidence audit.

Outer options:
  --remote-dir DIR      Remote checkout directory (default: /home/ubuntu/dyadic-quant).
  --image NAME          Docker image tag to build/run (default: dyadic-experiments).
  --no-sync             Do not rsync the repo or data/llm_eval.
  --no-build            Do not build the Docker image.
  --no-fetch            Do not fetch results back.
  --no-local-audit      Do not rerun the Level 2 audit locally after fetching.
  -h, --help            Show this help.

Common repro options:
  --level 1|2|all       Default: all.
  --run-id NAME         Default: remote-llm-<UTC timestamp>.
  --threads N           Default: 30.
  --quick               Passed through to dyadic-experiments.
  --strict-audit        Passed through to dyadic-experiments.

Example:
  ./dyadic-remote-experiments.sh ubuntu@A10_HOST --level all --threads 30
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage >&2
  exit 0
fi

REMOTE="$1"
shift

REMOTE_DIR="/home/ubuntu/dyadic-quant"
IMAGE="dyadic-experiments"
RUN_ID="remote-llm-$(date -u +%Y%m%dT%H%M%SZ)"
LEVEL="all"
THREADS="30"
SYNC=1
BUILD=1
FETCH=1
LOCAL_AUDIT=1
runner_extra=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote-dir)
      REMOTE_DIR="$2"
      shift 2
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --no-sync)
      SYNC=0
      shift
      ;;
    --no-build)
      BUILD=0
      shift
      ;;
    --no-fetch)
      FETCH=0
      shift
      ;;
    --no-local-audit)
      LOCAL_AUDIT=0
      shift
      ;;
    --level)
      LEVEL="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      runner_extra+=("$1")
      shift
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

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
ssh_remote=(ssh "${SSH_OPTS[@]}" "$REMOTE")
rsync_ssh="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

run_remote() {
  "${ssh_remote[@]}" "$@"
}

need_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required local file: $1" >&2
    exit 1
  fi
}

need_file data/llm_eval/wikitext2_test.txt
need_file data/llm_eval/arc_easy.json

echo "remote=$REMOTE"
echo "remote_dir=$REMOTE_DIR"
echo "run_id=$RUN_ID"
echo "level=$LEVEL"
echo "threads=$THREADS"
echo "image=$IMAGE"

if [[ "$SYNC" == "1" ]]; then
  echo "[remote-repro] creating remote directories"
  run_remote "mkdir -p '$REMOTE_DIR/data/llm_eval' '$REMOTE_DIR/results'"

  echo "[remote-repro] syncing repository"
  rsync -az --delete -e "$rsync_ssh" \
    --exclude .git \
    --exclude data \
    --exclude results \
    --exclude .pytest_cache \
    --exclude __pycache__ \
    --exclude dyadic_quant/level2/native/_build \
    ./ "$REMOTE:$REMOTE_DIR/"

  echo "[remote-repro] syncing LLM eval data"
  rsync -az -e "$rsync_ssh" data/llm_eval/ "$REMOTE:$REMOTE_DIR/data/llm_eval/"
fi

if [[ "$BUILD" == "1" ]]; then
  echo "[remote-repro] building Docker image"
  run_remote "cd '$REMOTE_DIR' && sudo docker build -t '$IMAGE' ."
fi

runner_args=(--level "$LEVEL" --threads "$THREADS" --run-id "$RUN_ID" "${runner_extra[@]}")

echo "[remote-repro] running dyadic-experiments: ${runner_args[*]}"
run_remote "cd '$REMOTE_DIR' && sudo docker run --rm --gpus all --ipc=host \
  -v '$REMOTE_DIR/data:/workspace/data' \
  -v '$REMOTE_DIR/results:/workspace/results' \
  -v dyadic-ollama:/root/.ollama \
  --workdir /workspace \
  '$IMAGE' ${runner_args[*]@Q}"

if [[ "$FETCH" == "1" ]]; then
  echo "[remote-repro] fetching results"
  if [[ "$LEVEL" == "1" || "$LEVEL" == "all" ]]; then
    mkdir -p results/level1
    rsync -az -e "$rsync_ssh" "$REMOTE:$REMOTE_DIR/results/level1/$RUN_ID/" "results/level1/$RUN_ID/"
  fi
  if [[ "$LEVEL" == "2" || "$LEVEL" == "all" ]]; then
    mkdir -p results/level2
    rsync -az -e "$rsync_ssh" "$REMOTE:$REMOTE_DIR/results/level2/$RUN_ID/" "results/level2/$RUN_ID/"
  fi
fi

if [[ "$LOCAL_AUDIT" == "1" && ( "$LEVEL" == "2" || "$LEVEL" == "all" ) ]]; then
  echo "[remote-repro] running local Level 2 audit"
  audit_args=(
    python3 experiments/level2/audit_native_evidence.py
    --level2-dir "results/level2/$RUN_ID"
    --level1-dir "results/level1/$RUN_ID"
    --output-dir "results/level2/$RUN_ID/evidence_local"
  )
  "${audit_args[@]}"
fi

echo "[remote-repro] complete"
if [[ "$LEVEL" == "1" || "$LEVEL" == "all" ]]; then
  echo "Level 1: results/level1/$RUN_ID"
fi
if [[ "$LEVEL" == "2" || "$LEVEL" == "all" ]]; then
  echo "Level 2: results/level2/$RUN_ID"
fi
