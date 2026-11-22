#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "${ARESIM_PYTHON:-engine/.venv/bin/python}" scripts/run_llm_dm_ablation.py "$@"
