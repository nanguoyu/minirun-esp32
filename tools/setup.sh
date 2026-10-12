#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for tool in python3 git cmake; do
    command -v "$tool" >/dev/null || { echo "Install $tool before running setup." >&2; exit 1; }
done
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
revision=956973c76466b6c791d7bdbe6eed3aa3235b2dc1
llama_dir=deps/llama.cpp
if [[ ! -e "$llama_dir" ]]; then
    mkdir -p "$llama_dir"
    git -C "$llama_dir" init -q
    git -C "$llama_dir" remote add origin https://github.com/ggml-org/llama.cpp.git
    git -C "$llama_dir" fetch --depth 1 origin "$revision"
    git -C "$llama_dir" checkout --detach FETCH_HEAD
fi
if [[ "$(git -C "$llama_dir" rev-parse HEAD)" != "$revision" ]] || [[ -n "$(git -C "$llama_dir" status --porcelain)" ]]; then
    echo "deps/llama.cpp must be an unchanged checkout of $revision." >&2
    exit 1
fi
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel "${MINIRUN_JOBS:-4}"
ctest --test-dir build --output-on-failure
printf '\nReady. Activate Python with: source .venv/bin/activate\n'
