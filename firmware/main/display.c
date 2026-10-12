#include "display.h"
#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_heap_caps.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_ops.h"
#include "esp_lcd_panel_vendor.h"
#include "font8x8.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include <stdio.h>
#include <string.h>
static esp_lcd_panel_handle_t panel;
static SemaphoreHandle_t done;
static uint16_t *pixels;
static bool ready;
static char phase[20];
static bool transferred(esp_lcd_panel_io_handle_t io, esp_lcd_panel_io_event_data_t *ev,
                        void *ctx) {
    (void)io;
    (void)ev;
    (void)ctx;
    BaseType_t wake = pdFALSE;
    xSemaphoreGiveFromISR(done, &wake);
    return wake == pdTRUE;
}
static uint16_t wire(uint16_t c) { return (c >> 8) | (c << 8); }
static bool flush(int y, int h) {
    if (!ready)
        return false;
    if (esp_lcd_panel_draw_bitmap(panel, 0, y, 240, y + h, pixels) != ESP_OK ||
        xSemaphoreTake(done, pdMS_TO_TICKS(1000)) != pdTRUE) {
        ready = false;
        return false;
    }
    return true;
}
static void strip(int y, int h, uint16_t bg, const char *text, uint16_t fg, int scale) {
    if (!ready)
        return;
    for (int i = 0; i < 240 * h; i++)
        pixels[i] = wire(bg);
    if (text)
        for (int n = 0; text[n] && 8 + n * 8 * scale + 8 * scale <= 240; n++) {
            unsigned char c = text[n];
            if (c >= 128)
                c = '?';
            for (int yy = 0; yy < 8; yy++)
                for (int xx = 0; xx < 8; xx++)
                    if (font8x8_basic[c][yy] & (1 << xx))
                        for (int dy = 0; dy < scale; dy++)
                            for (int dx = 0; dx < scale; dx++) {
                                int row = 2 + yy * scale + dy,
                                    col = 8 + n * 8 * scale + xx * scale + dx;
                                if (row < h)
                                    pixels[row * 240 + col] = wire(fg);
                            }
        }
    (void)flush(y, h);
}
int display_init(void) {
    gpio_config_t bl = {.pin_bit_mask = 1ULL << 48, .mode = GPIO_MODE_OUTPUT};
    if (gpio_config(&bl) != ESP_OK)
        return -1;
    gpio_set_level(48, 1);
    pixels = heap_caps_malloc(240 * 20 * 2, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    done = xSemaphoreCreateBinary();
    if (!pixels || !done)
        return -1;
    spi_bus_config_t bus = {.sclk_io_num = 21,
                            .mosi_io_num = 47,
                            .miso_io_num = -1,
                            .quadwp_io_num = -1,
                            .quadhd_io_num = -1,
                            .max_transfer_sz = 240 * 20 * 2};
    if (spi_bus_initialize(SPI2_HOST, &bus, SPI_DMA_CH_AUTO) != ESP_OK)
        return -1;
    esp_lcd_panel_io_spi_config_t io = {.dc_gpio_num = 43,
                                        .cs_gpio_num = 44,
                                        .pclk_hz = 40000000,
                                        .lcd_cmd_bits = 8,
                                        .lcd_param_bits = 8,
                                        .spi_mode = 0,
                                        .trans_queue_depth = 1,
                                        .on_color_trans_done = transferred};
    esp_lcd_panel_io_handle_t handle;
    if (esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)SPI2_HOST, &io, &handle) != ESP_OK)
        return -1;
    esp_lcd_panel_dev_config_t cfg = {
        .reset_gpio_num = -1, .rgb_ele_order = LCD_RGB_ELEMENT_ORDER_RGB, .bits_per_pixel = 16};
    if (esp_lcd_new_panel_st7789(handle, &cfg, &panel) != ESP_OK)
        return -1;
    if (esp_lcd_panel_reset(panel) != ESP_OK || esp_lcd_panel_init(panel) != ESP_OK ||
        esp_lcd_panel_invert_color(panel, true) != ESP_OK ||
        esp_lcd_panel_disp_on_off(panel, true) != ESP_OK)
        return -1;
    ready = true;
    for (int y = 0; y < 240; y += 20)
        strip(y, 20, 0x0841, NULL, 0, 1);
    strip(4, 20, 0x0841, "MINIRUN", 0x07ff, 2);
    strip(28, 12, 0x0841, "LFM2.5-230M / SD", 0xffff, 1);
    display_text("Screen ready.\nWaiting for a prompt.");
    display_status("READY", 0);
    gpio_set_level(48, 0);
    return ready ? 0 : -1;
}
void display_status(const char *text, int percent) {
    strip(198, 14, 0x0841, text, 0x07ff, 1);
    if (percent < 0)
        percent = 0;
    if (percent > 100)
        percent = 100;
    if (!ready)
        return;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 240; x++)
            pixels[y * 240 + x] = wire(x < 240 * percent / 100 ? 0x07e0 : 0x2104);
    (void)flush(216, 4);
}
void display_text(const char *text) {
    char rows[12][29] = {{0}};
    int line = 0, col = 0;
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        if ((*p & 0xc0) == 0x80)
            continue;
        if (*p == '\n' || col == 28) {
            line++;
            col = 0;
            if (line == 12) {
                memmove(rows, rows + 1, 11 * 29);
                memset(rows[11], 0, 29);
                line = 11;
            }
            if (*p == '\n')
                continue;
        }
        rows[line][col++] = *p < 128 ? *p : '?';
    }
    for (int l = 0; l < 12; l++)
        strip(48 + l * 12, 12, 0x0841, rows[l], 0xffff, 1);
}
void display_metrics(int position, uint64_t us) {
    char s[30];
    if (phase[0])
        snprintf(s, sizeof(s), "%s %.1fs", phase, (double)us / 1000000);
    else
        snprintf(s, sizeof(s), "POS %d %.1fs", position, (double)us / 1000000);
    strip(224, 12, 0x0841, s, 0xffe0, 1);
}

void display_frame(const uint8_t *gray4) {
    if (!ready)
        return;
    for (int y = 0; y < 144; y += 18) {
        for (int j = 0; j < 18; j++)
            for (int x = 0; x < 240; x++) {
                unsigned v = (gray4[(y + j) * 120 + x / 2] >> (x % 2 ? 0 : 4)) & 15;
                unsigned c =
                    ((1 + 30 * v / 15) << 11) | ((2 + 61 * v / 15) << 5) | (1 + 30 * v / 15);
                pixels[j * 240 + x] = wire(c);
            }
        if (!flush(48 + y, 18))
            return;
    }
}

void display_span(int first, int last, uint64_t us) {
    char s[30];
    if (phase[0])
        snprintf(s, sizeof(s), "%s %.1fs", phase, (double)us / 1000000);
    else if (first == last)
        snprintf(s, sizeof(s), "POS %d  %.1fs", first, (double)us / 1000000);
    else
        snprintf(s, sizeof(s), "POS %d-%d  %.1fs", first, last, (double)us / 1000000);
    strip(224, 12, 0x0841, s, 0xffe0, 1);
}

void display_set_phase(const char *text) {
    snprintf(phase, sizeof(phase), "%s", text);
    strip(224, 12, 0x0841, phase, 0xffe0, 1);
}
