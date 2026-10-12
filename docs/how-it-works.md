# How the model fits

The model file is about 149 MB. The board has 8 MiB of PSRAM. Minirun keeps the model on SD and reads the weights needed for the current calculation in small chunks.

```text
SD card             ESP32-S3                         Screen
149 MB model  →  two 32 KiB read buffers  →  layers  →  reply
                  read and compute overlap
```

## A layer at a time

LFM2.5-230M has 14 layers: eight short-convolution layers and six attention layers. Minirun implements these in C. Most weight matrices use Q4_0. The shared embedding/output table uses Q6_K, while several smaller tensors use FP32.

For a matrix multiplication, the board reads a tile of rows, decodes them, and multiplies them by the current input. It then reuses that memory for the next tile. Two buffers let one CPU core work while the other reads. The reading core also helps with arithmetic between reads.

A prompt is processed in groups of up to eight tokens. The same weights serve all tokens in that group. Generating a reply is slower per token because the next token depends on the one just chosen.

## Keeping the conversation

Attention needs keys and values from earlier positions. At 1024 positions, those arrays take 24 MiB in FP32. Minirun stores completed pages on SD and keeps the active pages in PSRAM. The cache workspace is 721,264 bytes, about 704 KiB.

When several prompt tokens are processed together, historical pages are read once for the group. Each token can attend only to its own past. The convolution state stays in RAM.

The 1024-token setting is a capacity. A 20-token exchange does not process 1004 empty positions. Each invocation of the chat client starts fresh, and the program stops if the capacity is exceeded.

## Why it takes its time

A generated position requests about 146.7 MB of model data. Reading that through a one-bit SD connection accounts for most of the delay. The board still has to visit nearly all weights for a short answer.

| Measurement | Time |
|---|---:|
| A new token with a short context | About 36.7 s |
| One position at the end of a 1024-token context | 43.14 s |
| First complete model verification | About 34 s |

These measurements used an ESP32-S3 EYE at 240 MHz, an SP32G card over 1-bit 40 MHz SDMMC, ESP-IDF 5.5.4, 32 KiB tiles, the aligned official QAD model, and FP32 cache. The full-capacity run was recorded on September 29, 2026, with chip temperatures of 30–33°C. The card's internal cache and temperature were not controlled. Other cards and wiring can change the timings.

The September run processed all 1024 positions of a fixed input sequence and checked outputs against the host implementation. The greeting in the README was generated separately on the board. Long-context capacity and the quality of a particular answer are different things.

## Memory you can account for

| Allocation | Bytes | Location |
|---|---:|---|
| Two model read buffers | 65,536 | Internal SRAM |
| Batch input staging | 81,920 | Internal SRAM |
| Model file buffer | 32,768 | Internal SRAM |
| Display pixels | 9,600 | Internal SRAM |
| Main and worker task stacks | 49,152 + 24,576 | Internal SRAM |
| Model state and scratch | 389,224 | PSRAM |
| Prompt batch workspace | 327,680 | PSRAM |
| SD conversation workspace | 721,264 | PSRAM |

The SDK and filesystem also allocate memory. During the full 1024-position run, about 6.56 MiB of PSRAM remained free. The runtime and firmware C sources in the first public release match the September implementation; the release changes installation, defaults, and the host chat interface.

## Try a different configuration

The default build uses 1024 positions with FP32 cache on SD. To keep a shorter conversation entirely in PSRAM:

```sh
MINIRUN_CONTEXT=128 MINIRUN_KV_FILE=OFF MINIRUN_KV_SHARED=OFF \
  bash tools/build_firmware.sh
```

That uses 3 MiB for the conversation cache. Experimental FP16 storage can fit 256 positions into the same amount of RAM:

```sh
MINIRUN_CONTEXT=256 MINIRUN_KV_BITS=16 MINIRUN_KV_FILE=OFF MINIRUN_KV_SHARED=OFF \
  bash tools/build_firmware.sh
```

FP16 changes the stored values. It has host tests and a short device demonstration; the public default stays FP32. Build settings are described in `firmware/main/CMakeLists.txt`.
