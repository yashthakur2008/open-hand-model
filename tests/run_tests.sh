#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
python -m pytest -q
if [[ "${RUN_MODEL_SMOKE:-0}" == "1" ]]; then
  python -m pytest -q tests/test_model_smoke.py
fi
