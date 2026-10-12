#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo "$@"
cmake --build build --parallel "${MINIRUN_JOBS:-4}"
ctest --test-dir build --output-on-failure
