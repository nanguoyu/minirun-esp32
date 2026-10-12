# The board and its screen

The tested board is a **YD-ESP32-S3-EYE-MB v2.2**, an ESP32-S3 EYE variant with an ST7789 display and a microSD slot.

| Part | Configuration |
|---|---|
| CPU | ESP32-S3, two cores at 240 MHz |
| PSRAM | 8 MiB, octal, 80 MHz |
| Flash | 8 MiB |
| LCD | ST7789, 240 × 240, RGB, inverted colors |
| SD card | FAT32; tested with a 32 GB SP32G |
| SD interface | SDMMC, 1 data bit, 40 MHz |
| USB | Native USB Serial/JTAG |

## Connections

| SD signal | GPIO |
|---|---:|
| CLK | 39 |
| CMD | 38 |
| D0 | 40 |

D1–D3 are not connected on the tested board. Four-bit SD operation needs different wiring.

| LCD signal | GPIO |
|---|---:|
| MOSI | 47 |
| SCLK | 21 |
| CS | 44 |
| DC | 43 |
| Backlight | 48, active low |
| Reset | Not driven by this firmware |

The display uses SPI2 at 40 MHz. The driver selects RGB order and enables color inversion. These settings produced the correct orientation and colors on the tested board.

## Another ESP32-S3 board?

Start with 8 MiB PSRAM and 8 MiB flash, then compare the SD and LCD pins above. The SD configuration is in `firmware/main/main.c`; the display configuration is in `firmware/main/display.c`. Update those files for your wiring before flashing.

The program currently expects the EYE display and SD setup. There is no board-selection menu. A board with the same processor but different peripherals needs a small port.

Use stable USB power and insert or remove the SD card with power disconnected. Model files are opened read-only during inference; temporary conversation files are written beside them.
