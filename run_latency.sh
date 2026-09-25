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

# Optional data overrides:
#   DATA_ROOT=/path/holding/data_dirs ./run_latency.sh
#   EXID_MMAP_DIR=/path/to/exiD/dimI EXID_SPLIT_DIR=/path/to/exiD/splits ./run_latency.sh
# Optional checkpoint overrides:
#   CKPT_ROOT=/path/to/cslstm_ckpts ./run_latency.sh
#   EXID_BASE_CKPT=/path/to/exiD0-5.pt EXID_I_CKPT=/path/to/exiDI-5.pt ./run_latency.sh

cases=(
  "exiD-baseline|configs/exiD0-5.yaml|ckpts/exiD0-5/best.pt"
  "exiD-+I|configs/exiDI-5.yaml|ckpts/exiDI-5/best.pt"
)

for row in "${cases[@]}"; do
  IFS='|' read -r name config ckpt <<< "$row"
  dataset="${name%%-*}"
  condition="${name#*-}"
  log_path="${LOG_DIR}/${name}.log"

  ckpt_key=""
  if [[ "$condition" == "baseline" ]]; then
    ckpt_key="${EXID_BASE_CKPT:-}"
  else
    ckpt_key="${EXID_I_CKPT:-}"
  fi
  if [[ -n "$ckpt_key" ]]; then
    ckpt="$ckpt_key"
  elif [[ -n "${CKPT_ROOT:-}" ]]; then
    ckpt="${CKPT_ROOT}/${ckpt#ckpts/}"
  fi

  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP] ${name}: missing ${ckpt}"
    continue
  fi

  cmd=(
    "$PYTHON_BIN" evaluate.py
    --config "$config"
    --ckpt "$ckpt"
    --measure_time
    --latency_warmup "$WARMUP"
    --latency_iters "$ITERS"
  )

  mmap_dir=""
  split_dir=""
  mmap_dir="${EXID_MMAP_DIR:-}"
  split_dir="${EXID_SPLIT_DIR:-}"
  if [[ -z "$mmap_dir" && -n "${DATA_ROOT:-}" ]]; then
    mmap_dir="${DATA_ROOT}/${dataset}/dimI"
  fi
  if [[ -z "$split_dir" && -n "${DATA_ROOT:-}" ]]; then
    split_dir="${DATA_ROOT}/${dataset}/splits"
  fi
  if [[ -n "$mmap_dir" ]]; then
    cmd+=(--mmap_dir "$mmap_dir")
  fi
  if [[ -n "$split_dir" ]]; then
    cmd+=(--split_dir "$split_dir")
  fi

  echo "[RUN] ${name}"
  printf '  %q' "${cmd[@]}"
  echo
  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi
  "${cmd[@]}" 2>&1 | tee "$log_path"
done
