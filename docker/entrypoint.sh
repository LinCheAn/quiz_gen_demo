#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/app/models/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

mkdir -p \
  /app/models/adapters \
  /app/models/cache/huggingface \
  /app/data/uploads \
  /app/artifacts

exec python /app/app.py --port "${APP_PORT:-7860}"
