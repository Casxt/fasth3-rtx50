#!/usr/bin/env bash
# Thin launcher; setup logic lives in the shared Python entry point.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${FASTH3_PYTHON:-python3.12}" "$root/scripts/bootstrap.py" "$@"
