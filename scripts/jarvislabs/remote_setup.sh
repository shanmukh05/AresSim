#!/usr/bin/env bash
# Retain the documented JarvisLabs entrypoint; both providers share setup.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec bash "$ROOT/scripts/cloud/remote_setup.sh" "${1:-rllib}"
