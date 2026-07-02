#!/usr/bin/env bash
set -euo pipefail

cd /workspace

model_args=()
next_is_model=0
next_is_embed=0
next_is_judge=0
for arg in "$@"; do
  if [[ "$next_is_model" == "1" ]]; then
    model_args+=("$arg")
    next_is_model=0
    continue
  fi
  if [[ "$next_is_embed" == "1" ]]; then
    model_args+=("$arg")
    next_is_embed=0
    continue
  fi
  if [[ "$next_is_judge" == "1" ]]; then
    [[ -n "$arg" ]] && model_args+=("$arg")
    next_is_judge=0
    continue
  fi
  case "$arg" in
    --ollama-model)
      next_is_model=1
      ;;
    --embed-model)
      next_is_embed=1
      ;;
    --judge-model)
      next_is_judge=1
      ;;
  esac
done

if [[ "${DYADIC_START_OLLAMA:-1}" == "1" ]]; then
  mkdir -p /workspace/results
  export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11434}"
  ollama serve > /workspace/results/ollama-serve.log 2>&1 &
  server_pid="$!"
  trap 'kill "$server_pid" 2>/dev/null || true' EXIT

  for _ in $(seq 1 120); do
    if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  curl -fsS http://127.0.0.1:11434/api/tags >/dev/null

  models=()
  if [[ -n "${DYADIC_OLLAMA_MODELS:-}" ]]; then
    read -r -a models <<<"${DYADIC_OLLAMA_MODELS}"
  else
    models=(qwen2.5:0.5b nomic-embed-text "${DYADIC_JUDGE_MODEL:-gemma3:4b}")
  fi
  models+=("${model_args[@]}")

  seen=" "
  for model in "${models[@]}"; do
    [[ -z "$model" ]] && continue
    if [[ "$seen" == *" $model "* ]]; then
      continue
    fi
    seen="$seen$model "
    echo "Pulling Ollama model: $model"
    ollama pull "$model"
  done
fi

exec /workspace/dyadic-experiments.sh "$@"
