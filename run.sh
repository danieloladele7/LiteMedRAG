#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-help}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-config.json}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export LITEMEDRAG_TORCH_THREADS="${LITEMEDRAG_TORCH_THREADS:-1}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

case "$MODE" in
  check)
    "$PYTHON_BIN" -c 'import numpy,pandas,torch; import litemedrag; print("LiteMedRAG imports OK")'
    ;;
  smoke)
    "$PYTHON_BIN" scripts/run.py --synthetic --encoder random --device cpu --datasets slake \
      --variants base text_rag mm_rag litemedrag_acc litemedrag_ground \
      --limit-per-split 32 --max-epochs 2 --patience 1 --skip-figures --artifact-root artifacts_smoke
    ;;
  slake)
    "$PYTHON_BIN" scripts/run.py --config "$CONFIG" --artifact-root "$ARTIFACT_ROOT" --datasets slake
    ;;
  imageclef)
    "$PYTHON_BIN" scripts/run.py --config "$CONFIG" --artifact-root "$ARTIFACT_ROOT" --datasets imageclef_vqa_med_2019
    ;;
  all)
    "$PYTHON_BIN" scripts/run.py --config "$CONFIG" --artifact-root "$ARTIFACT_ROOT" --datasets slake imageclef_vqa_med_2019
    ;;
  *)
    cat <<'EOF'
Usage: ./run.sh {check|smoke|slake|imageclef|all}

First install:
  python -m venv .venv
  source .venv/bin/activate
  pip install -e .

Then run:
  ./run.sh smoke
  ./run.sh all
EOF
    ;;
esac
