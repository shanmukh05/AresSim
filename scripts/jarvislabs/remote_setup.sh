#!/usr/bin/env bash
# Install Python 3.12 and aresim[rllib] on a JarvisLabs instance.
#
# Lives next to cloud_train.py and is uploaded to the instance. Training still
# uses the same aresim-rl CLI as local runs; this script only prepares the venv.
# Re-run is safe: an existing .venv is reused and dependencies are re-synced.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv python install 3.12
if [[ ! -x .venv/bin/python ]]; then
  uv venv .venv --python 3.12 --seed
fi

# Same extra as local training. Do not point uv at download.pytorch.org first:
# that index currently stops at torch 2.11+cu128, which cannot satisfy
# engine[rllib]'s torch>=2.13 pin. PyPI linux wheels for 2.13 include CUDA.
uv pip install --python .venv/bin/python -e './engine[rllib]'

.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if command -v nvidia-smi >/dev/null 2>&1; then
  .venv/bin/python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
fi
.venv/bin/aresim-rl --help >/dev/null
echo "remote setup ok"
