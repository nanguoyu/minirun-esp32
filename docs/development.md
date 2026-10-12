# Working on Minirun

The repository has two programs: the ESP32 firmware and the computer-side text client.

| Directory | What is there |
|---|---|
| `src` | Model operations, quantized row decoding, conversation cache |
| `firmware/main` | SD card, LCD, USB commands, two-core scheduling |
| `tools` | Chat, model preparation, host reference, device probes |
| `tests` | C and Python tests |
| `models` | Model download information and fixed tensor layout |

## Build and test

After `bash tools/setup.sh`:

```sh
source .venv/bin/activate
python -m unittest discover -s tests -v
ctest --test-dir build --output-on-failure
```

The short C tests cover quantized decoding, FP16 conversion, cache paging, cancellation, short reads, write failures, and the attention seen by every query in a batch. They need no model download or device.

For address and undefined-behavior checks:

```sh
bash tools/build_host.sh -DMINIRUN_SANITIZE=ON
```

For tests that read the model, first download and prepare it as described in Setup:

```sh
python tools/generate_index.py
python tools/test_tokenizer_capacity.py
bash tools/build_host.sh -DMINIRUN_MODEL_TESTS=ON
python tools/parity.py
```

Model tests take longer. The scalar C implementation and llama.cpp use different arithmetic in some operations; the comparison checks the highest-scoring token and the full output vector. Cache and batch comparisons within the C implementation require identical bytes.

GitHub Actions builds the host tools on Linux and macOS and compiles the ESP32 firmware. Device timing and screen checks need a connected board.

## USB commands

The chat client sends these over USB at 115200 baud. Use one serial client at a time.

| Command | Purpose |
|---|---|
| `INFO` | Board configuration and capacity |
| `OPEN` | Verify the model and start an empty session |
| `RESET` | Start fresh using an already-open, verified model |
| `PREFILL <head> <ids...>` | Process 1–8 tokens; `head=1` requests the next-token scores |
| `STEP <id> <head>` | Process one token |
| `LOGITS` | Read the latest 65,536 FP32 scores |
| `KVINFO` | Conversation-cache size and I/O counters |
| `KVCLOSE` | Delete the current owned cache and invalidate the session |
| `!` | Cancel at the next read boundary; no newline needed |

The firmware also provides `FRAME`, `TEXT`, and `PHASE` for the display. `tools/device.py` has upload and raw-token helpers; `tools/probe_sd_kv.py` runs longer device checks with host reference files you supply.

## Places to experiment

The board has spare PSRAM. Caching frequently read weights or older attention pages may help. Prompt processing already shares weights across eight positions; trying other batching choices is another small experiment. Supporting another LCD is a useful first hardware contribution.

Keep the original output as a comparison when changing arithmetic or scheduling. When reporting a timing, include the board, SD wiring and clock, model, build settings, prompt length, and the part you timed. Do not add I/O and compute times together when they overlap.

## Model and dependency versions

The text helper uses llama.cpp commit `956973c76466b6c791d7bdbe6eed3aa3235b2dc1`. Setup fetches it directly from the public project. The firmware uses ESP-IDF 5.5.4. Model revision and hashes live in `models/lfm230-artifact.json` and `models/lfm230-aligned-artifact.json`.

The C runtime originated in the private development snapshot `aad1438fcd565f27972c5e47269b90912a9d31ee`. Its model and firmware C files are unchanged in the initial public import `a8b5340`; later commits show the release work. Historical device measurements refer to that runtime. The private repository's logs and device identifiers are not required to build or use this project.
