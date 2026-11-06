#!/usr/bin/env bash
# Shared cloud dependency installation; engine/pyproject.toml owns version pins.
set -euo pipefail

EXTRA="${1:-rllib}"
case "$EXTRA" in
  rllib|muzero|world_models) ;;
  *) echo "unsupported training extra: $EXTRA" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"
uv python install 3.12
if [[ ! -x .venv/bin/python ]]; then
  uv venv .venv --python 3.12 --seed
fi
# The PyPI CUDA wheels satisfy the engine pins; alternative Torch indexes may lag.
uv pip install --python .venv/bin/python -e "./engine[$EXTRA]"
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if command -v nvidia-smi >/dev/null 2>&1; then
  .venv/bin/python -c "import torch; assert torch.cuda.is_available(); print((torch.ones(16, device='cuda') @ torch.ones(16, device='cuda')).item())"
fi
.venv/bin/aresim-rl --help >/dev/null
echo "remote setup ok"
