# Your first reply

These steps use a Mac and the ESP32-S3 EYE described in [Hardware](hardware.md). Allow about 1 GB of disk space for dependencies and the two model files. The first build takes a few minutes.

## 1. Set up the computer

Install Apple's command-line tools with `xcode-select --install` if needed. You also need Git, Python 3.12 or later, and CMake. With Homebrew, `brew install python cmake` supplies the latter two.

```sh
git clone https://github.com/nanguoyu/minirun-esp32.git
cd minirun-esp32
bash tools/setup.sh
source .venv/bin/activate
```

Setup installs pyserial into `.venv`, fetches a fixed version of llama.cpp into `deps`, and builds the text helper and host tests. Nothing is installed into system Python.

The full text display uses macOS Cocoa fonts. On Linux, the chat client uses the board's built-in ASCII font. Windows setup is not covered yet.

## 2. Get the model

```sh
python tools/fetch_lfm.py
python tools/repack_alignment.py
```

The first command downloads the official 149 MB model and checks its SHA-256. The second creates `artifacts/model-aligned.gguf`, with tensor data placed at 4 KiB boundaries for SD reads. Both files stay on your computer. Their weight values are identical.

Only this model and quantization are supported. Other GGUF files will not work with this firmware.

## 3. Build and flash

Install [ESP-IDF 5.5.4 for ESP32-S3](https://docs.espressif.com/projects/esp-idf/en/v5.5.4/esp32s3/get-started/index.html). Activate it in a **second terminal**, using the `export.sh` from your installation, then return to the project directory:

```sh
source /path/to/esp-idf/export.sh
cd /path/to/minirun-esp32
bash tools/build_firmware.sh
```

The default is 1024-token context with FP32 conversation cache on SD. The firmware is for an 8 MiB flash / 8 MiB PSRAM board with the pins listed in [Hardware](hardware.md).

Find the board's port:

```sh
python -m serial.tools.list_ports -v
```

Set `MINIRUN_PORT` to that port. On macOS it usually starts with `/dev/cu.usbmodem`; on Linux, `/dev/ttyACM`. Select the Espressif USB JTAG/serial device, not another USB peripheral.

```sh
export MINIRUN_PORT=/dev/cu.usbmodemYOUR_PORT
mkdir -p backups
python -m esptool --chip esp32s3 --port "$MINIRUN_PORT" read_flash 0 0x800000 backups/before-minirun.bin
idf.py -C firmware -p "$MINIRUN_PORT" flash
```

The backup command is for an **8 MiB flash** board and uses esptool 4.x, included with ESP-IDF 5.5.4. Keep the backup if you want to restore the previous program. Flashing replaces the board's program. It does not format the SD card.

## 4. Put the model on the card

Use a FAT32 card with at least 200 MB free. You have two choices:

- **Card reader:** copy `artifacts/model-aligned.gguf` to the card's root directory and name it **`LFM4K.GGF`**. Eject the card, insert it into the powered-off board, then connect USB.
- **USB upload:** insert the card before starting the board, then run the command below from your first terminal. This is slower than a card reader. The firmware refuses to overwrite an existing model file.

```sh
source .venv/bin/activate
python tools/device.py --aligned upload
```

The firmware checks the complete model file before running it. It writes temporary conversation files called `KV0000.TMP`, `KV0001.TMP`, and so on. Normal chat completion removes its own cache file. After an interrupted power supply, old cache files can remain. With the board powered off, you can remove those `KV*.TMP` files using a card reader.

## 5. Say hello

```sh
python tools/chat.py --prompt 'Hello!' --new-tokens 16
```

The screen first shows `Thinking...`, then the reply. Model verification takes about 34 seconds on the tested card. Processing the short greeting and producing its first token takes about 84 seconds more. Subsequent tokens take about 37 seconds each.

When the model finishes, the screen says `DONE`. If it reaches your chosen output limit first, it says `OUTPUT LIMIT`. Run the command again for a new message.

Use `--port PORT` if more than one board is connected, and `--verbose` when diagnosing a problem. You can also set `MINIRUN_PORT` in this terminal.

## If something gets stuck

| What you see | What to check |
|---|---|
| No ESP32-S3 found | Use a data cable and the board's native USB port. List ports again. |
| `sd_mount` | Card inserted before boot, FAT32 filesystem, and GPIO39/38/40 wiring. |
| `model_missing` | `LFM4K.GGF` must be in the card's root directory. |
| `verification_failed` | Copy the aligned file again; check that the copy completed before ejecting. |
| `destination_exists` during upload | A model or partial upload already exists. Inspect the card with a reader; keep the good model. |
| Blank or wrong-color LCD | Check the [display pins and settings](hardware.md). |
| Message exceeds context | Shorten the input or lower `--new-tokens`. Input and output share the limit. |
| Serial port busy | Close serial monitors and other chat processes. |

The board does not accept a second inference command while it is working. Ctrl+C requests cancellation at a read boundary. If the connection is lost, reconnect the board before starting again.
