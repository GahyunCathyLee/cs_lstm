#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/gahyun/miniconda3/envs/tf/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

WARMUP="${WARMUP:-100}"
ITERS="${ITERS:-1000}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-logs/latency}"
mkdir -p "$LOG_DIR"

cases=(
  "exiD-baseline|configs/exiD0-5.yaml|ckpts/exiD0-5/best.pt"
  "exiD-+I|configs/exiDI-5.yaml|ckpts/exiDI-5/best.pt"
  "highD-baseline|configs/highD0-4.yaml|ckpts/highD0-4/best.pt"
  "highD-+I|configs/highDI-3.yaml|ckpts/highDI-3/best.pt"
)

for row in "${cases[@]}"; do
  IFS='|' read -r name config ckpt <<< "$row"
  log_path="${LOG_DIR}/${name}.log"

  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP] ${name}: missing ${ckpt}"
    continue
  fi

  cmd=(
    "$PYTHON_BIN" evaluate.py
    --config "$config"
    --measure_time
    --latency_warmup "$WARMUP"
    --latency_iters "$ITERS"
  )

  echo "[RUN] ${name}"
  printf '  %q' "${cmd[@]}"
  echo
  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi
  "${cmd[@]}" 2>&1 | tee "$log_path"
done
