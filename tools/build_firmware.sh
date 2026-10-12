#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v idf.py >/dev/null; then
    echo 'Activate ESP-IDF 5.5.4 first: source /path/to/esp-idf/export.sh' >&2
    exit 1
fi
idf.py -C firmware \
    -D MINIRUN_TILE_BYTES="${MINIRUN_TILE_BYTES:-32768}" \
    -D MINIRUN_MODEL_ALIGNED="${MINIRUN_MODEL_ALIGNED:-ON}" \
    -D MINIRUN_CONTEXT="${MINIRUN_CONTEXT:-1024}" \
    -D MINIRUN_KV_BITS="${MINIRUN_KV_BITS:-32}" \
    -D MINIRUN_KV_FILE="${MINIRUN_KV_FILE:-ON}" \
    -D MINIRUN_KV_SHARED="${MINIRUN_KV_SHARED:-ON}" build
