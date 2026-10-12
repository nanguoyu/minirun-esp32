#include "display.h"
#include "driver/sdmmc_host.h"
#include "driver/temperature_sensor.h"
#include "driver/usb_serial_jtag.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_vfs_fat.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lfm.h"
#include "kv_file.h"
#ifndef LFM_KV_FILE
#define LFM_KV_FILE 0
#endif
#include "mbedtls/sha256.h"
#include "model_layout.h"
#include "pipeline.h"
#include "sdmmc_cmd.h"
#include <dirent.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#define MODEL MODEL_PATH
#define PART MODEL_PART_PATH
static uint8_t *buffer;
static char model_stdio[32768];
static bool logits_ready;
static FILE *model;
static bool verified_handle;
static struct stat verified_stat;
static sdmmc_card_t *mounted_card;
static Lfm *runner;
static LfmBatch *batch_workspace;
static int active_count = 1;
#if LFM_KV_FILE
static LfmKVFile *kv_file;
#define KV_BACKEND_NAME "FILE"
#define KV_RESIDENT_BYTES sizeof(LfmKVFile)
#else
static LfmKV *kv;
#define KV_BACKEND_NAME "RAM"
#define KV_RESIDENT_BYTES LFM_KV_BYTES
#endif
static int stop;
static temperature_sensor_handle_t temperature;
static bool use_pipeline = true;
static int64_t step_started, last_draw;
/* Each ringbuffer item must fit the 4096-byte TX ring. */
static int send_bytes(const void *data, size_t count) {
    const uint8_t *p = data;
    while (count) {
        size_t chunk = count < 1024 ? count : 1024;
        if (usb_serial_jtag_write_bytes(p, chunk, pdMS_TO_TICKS(3000)) != (int)chunk)
            return -1;
        p += chunk;
        count -= chunk;
    }
    return 0;
}
static void reply(const char *s) { (void)send_bytes(s, strlen(s)); }
static int receive(void *dst, size_t n) {
    size_t got = 0;
    while (got < n) {
        int r = usb_serial_jtag_read_bytes((uint8_t *)dst + got, n - got, pdMS_TO_TICKS(15000));
        if (r <= 0)
            return -1;
        got += r;
    }
    return 0;
}
static void digest_string(unsigned char *hash, char *hex) {
    for (int i = 0; i < 32; i++)
        sprintf(hex + 2 * i, "%02x", hash[i]);
}
static int verify(FILE *f) {
    mbedtls_sha256_context sha;
    unsigned char hash[32];
    char hex[65];
    size_t total = 0;
    int64_t started = esp_timer_get_time();
    mbedtls_sha256_init(&sha);
    mbedtls_sha256_starts(&sha, 0);
    rewind(f);
    while (total < MODEL_BYTES) {
        size_t n = MODEL_BYTES - total;
        if (n > LFM_IO)
            n = LFM_IO;
        if (fread(buffer, 1, n, f) != n) {
            mbedtls_sha256_free(&sha);
            return -1;
        }
        mbedtls_sha256_update(&sha, buffer, n);
        total += n;
        if (total == LFM_IO || total % (8 * 1024 * 1024) == 0) {
            char status[120];
            snprintf(status, sizeof(status), "VERIFY bytes=%u us=%lld\n", (unsigned)total,
                     (long long)(esp_timer_get_time() - started));
            reply(status);
            display_status("VERIFYING MODEL", (int)(100ULL * total / MODEL_BYTES));
        }
    }
    int trailing = fgetc(f);
    mbedtls_sha256_finish(&sha, hash);
    mbedtls_sha256_free(&sha);
    digest_string(hash, hex);
    char line[110];
    snprintf(line, sizeof(line), "SHA256 %s\n", hex);
    reply(line);
    return trailing == EOF && !ferror(f) && !strcmp(hex, MODEL_SHA256) ? 0 : -1;
}
static void upload(void) {
    struct stat st;
    if (!stat(MODEL, &st) || !stat(PART, &st)) {
        reply("ERROR destination_exists\n");
        return;
    }
    FILE *f = fopen(PART, "wb");
    if (!f) {
        reply("ERROR create_failed\n");
        return;
    }
    setvbuf(f, NULL, _IONBF, 0);
    reply("READY 16384\n");
    mbedtls_sha256_context sha;
    unsigned char hash[32];
    char hex[65];
    mbedtls_sha256_init(&sha);
    mbedtls_sha256_starts(&sha, 0);
    int bad = 0;
    for (size_t done = 0; done < MODEL_BYTES;) {
        size_t n = MODEL_BYTES - done;
        if (n > 16384)
            n = 16384;
        if (receive(buffer, n) || fwrite(buffer, 1, n, f) != n) {
            bad = 1;
            break;
        }
        mbedtls_sha256_update(&sha, buffer, n);
        done += n;
        reply("ACK\n");
    }
    if (fflush(f))
        bad = 1;
    if (fclose(f))
        bad = 1;
    mbedtls_sha256_finish(&sha, hash);
    mbedtls_sha256_free(&sha);
    digest_string(hash, hex);
    if (bad || strcmp(hex, MODEL_SHA256)) {
        reply("ERROR upload_incomplete_part_retained\n");
        return;
    }
    /* The next OPEN performs a full read-back hash before any execution. */
    if (rename(PART, MODEL))
        reply("ERROR rename_failed\n");
    else
        reply("UPLOADED\n");
}
static void tensor_progress(const Tensor *t, uint32_t rows, uint32_t total, void *unused) {
    (void)unused;
    int64_t now = esp_timer_get_time();
    if (rows && rows < total && now - last_draw < 250000)
        return;
    last_draw = now;
    char text[32];
    int layer = 0;
    const char *op = "WEIGHTS";
    if (!strcmp(t->name, "token_embd.weight"))
        op = "OUTPUT HEAD";
    else {
        sscanf(t->name, "blk.%d.", &layer);
        layer++;
        if (strstr(t->name, "ffn_gate"))
            op = "GATE";
        else if (strstr(t->name, "ffn_up"))
            op = "FFN UP";
        else if (strstr(t->name, "ffn_down"))
            op = "FFN DOWN";
        else if (strstr(t->name, "in_proj"))
            op = "CONV IN";
        else if (strstr(t->name, "out_proj"))
            op = "CONV OUT";
        else if (strstr(t->name, "attn_output"))
            op = "ATTN OUT";
        else
            op = "ATTENTION";
    }
    int percent = total ? (int)(100ULL * rows / total) : 0;
    if (layer)
        snprintf(text, sizeof(text), "L%02d READ %s %d%%", layer, op, percent);
    else
        snprintf(text, sizeof(text), "READ %s %d%%", op, percent);
    display_status(text, percent);
    display_span(runner->position + 1, runner->position + active_count, now - step_started);
}
static uint64_t now_us(void) { return esp_timer_get_time(); }
static void progress(int layer, void *unused) {
    (void)unused;
    char line[140];
    snprintf(line, sizeof(line), "LAYER %d bytes=%llu heap_internal=%u heap_psram=%u\n", layer,
             (unsigned long long)runner->read_bytes,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    reply(line);
    char status[30];
    snprintf(status, sizeof(status), "LAYER %d / 14", layer + 1);
    display_status(status, (layer + 1) * 100 / 14);
    vTaskDelay(1);
}
static int cancelled(void *unused) {
    (void)unused;
    char c;
    while (usb_serial_jtag_read_bytes(&c, 1, 0) > 0)
        if (c == '!')
            stop = 1;
    return stop;
}
static void benchmark_io(void) {
    if (!model || runner->failed) {
        reply("ERROR model_not_ready\n");
        return;
    }
    const uint32_t head = 2387936, total = 16 * 1024 * 1024;
    for (int arm = 0; arm < 4; arm++) {
        bool aligned = (arm == 1 || arm == 2);
        uint32_t off = aligned ? (head + 511) / 512 * 512 : head;
        int64_t start = esp_timer_get_time();
        for (uint32_t n = 0; n < total; n += 32768)
            if (lfm_read(runner, off + n, buffer, 32768)) {
                reply("ERROR benchmark_read\n");
                return;
            }
        char line[150];
        snprintf(line, sizeof(line), "IOBENCH arm=%d aligned=%d offset=%u bytes=%u us=%lld\n", arm,
                 aligned, (unsigned)off, (unsigned)total,
                 (long long)(esp_timer_get_time() - start));
        reply(line);
        display_status(aligned ? "ALIGNED READ TEST" : "UNALIGNED READ TEST", (arm + 1) * 25);
    }
    reply("IOBENCH_DONE\n");
}
static int prepare_runner(void) {
#if LFM_KV_FILE
    if (lfm_kv_file_close(kv_file) || lfm_kv_file_open(kv_file, "/sdcard", buffer)) {
        reply("ERROR kv_file_open\n");
        return -1;
    }
    if (lfm_init_external(runner, model, buffer, lfm_kv_file_attention, kv_file)) {
        lfm_kv_file_close(kv_file);
        return -1;
    }
#if LFM_KV_SHARED
    runner->attention_many_hook = lfm_kv_file_attention_many;
#endif
#else
    if (lfm_init(runner, model, buffer, kv)) return -1;
#endif
    stop = 0;
    logits_ready = false;
    runner->cancelled = cancelled;
    runner->progress = progress;
    runner->clock_us = now_us;
    runner->tensor_progress = tensor_progress;
    runner->matvec_hook = use_pipeline ? pipeline_matvec : NULL;
    runner->many_hook = use_pipeline ? pipeline_many : NULL;
    display_status("MODEL READY", 100);
    return 0;
}
static void reset_verified_model(void) {
    struct stat now;
    if (!verified_handle || !model || runner->failed || runner->file != model || !mounted_card ||
        sdmmc_get_status(mounted_card) != ESP_OK || fstat(fileno(model), &now) ||
        now.st_dev != verified_stat.st_dev || now.st_ino != verified_stat.st_ino ||
        now.st_size != verified_stat.st_size || now.st_mtime != verified_stat.st_mtime ||
        now.st_ctime != verified_stat.st_ctime) {
        verified_handle = false;
        runner->failed = 1;
        logits_ready = false;
        reply("RESET_NEEDS_OPEN\n");
        return;
    }
    int64_t start = esp_timer_get_time();
    if (prepare_runner()) {
        verified_handle = false;
        runner->failed = 1;
        logits_ready = false;
        reply("RESET_NEEDS_OPEN\n");
        return;
    }
    char line[100];
    snprintf(line, sizeof(line), "RESET_OK reused_verified_handle us=%lld\n",
             (long long)(esp_timer_get_time() - start));
    reply(line);
}
static void open_model(void) {
    verified_handle = false;
    logits_ready = false;
    if (model) {
        fclose(model);
        model = NULL;
    }
    model = fopen(MODEL, "rb");
    if (!model) {
        reply("ERROR model_missing\n");
        return;
    }
    /* Explicit bounded stdio cache, separate from the DMA destination. */
    if (setvbuf(model, model_stdio, _IOFBF, sizeof(model_stdio))) {
        fclose(model);
        model = NULL;
        reply("ERROR stdio_buffer\n");
        return;
    }
    reply("VERIFYING\n");
    display_status("VERIFYING MODEL", 0);
    if (verify(model) || fstat(fileno(model), &verified_stat) || prepare_runner()) {
        fclose(model);
        model = NULL;
        reply("ERROR verification_failed\n");
        return;
    }
    verified_handle = true;
    char opened[80];
    snprintf(opened, sizeof(opened), "OPENED context=%d\n", LFM_CONTEXT);
    reply(opened);
}
void app_main(void) {
    usb_serial_jtag_driver_config_t usb = {.tx_buffer_size = 4096, .rx_buffer_size = 32768};
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&usb));
    if (pipeline_init()) {
        reply("ERROR pipeline_alloc\n");
        return;
    }
    buffer = pipeline_buffer();

    reply(display_init() == 0 ? "LCD initialized\n" : "LCD error\n");
    temperature_sensor_config_t tc = TEMPERATURE_SENSOR_CONFIG_DEFAULT(10, 80);
    if (temperature_sensor_install(&tc, &temperature) == ESP_OK)
        temperature_sensor_enable(temperature);
    runner = heap_caps_calloc(1, sizeof(Lfm), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
#if LFM_KV_FILE
    kv_file = heap_caps_calloc(1, sizeof(LfmKVFile), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    bool kv_ok = kv_file != NULL;
#else
    kv = heap_caps_calloc(LFM_KV_COUNT, sizeof(LfmKV), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    bool kv_ok = kv != NULL;
#endif
    batch_workspace = heap_caps_calloc(1, sizeof(LfmBatch), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!buffer || !runner || !kv_ok || !batch_workspace) {
        char error[160];
        snprintf(error, sizeof(error), "ERROR allocation mask=%u internal8=%u dma_largest=%u\n",
                 (!buffer) | ((!runner) << 1) | ((!kv_ok) << 2) | ((!batch_workspace) << 3),
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
                 (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_DMA));
        reply(error);
        display_status("MEMORY BUDGET ERROR", 0);
        return;
    }
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.max_freq_khz = SDMMC_FREQ_HIGHSPEED;
    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
    slot.width = 1;
    slot.clk = 39;
    slot.cmd = 38;
    slot.d0 = 40;
    slot.d1 = GPIO_NUM_NC;
    slot.d2 = GPIO_NUM_NC;
    slot.d3 = GPIO_NUM_NC;
    slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;
    esp_vfs_fat_sdmmc_mount_config_t cfg = {
        .format_if_mount_failed = false, .max_files = 3, .allocation_unit_size = 16 * 1024};
    sdmmc_card_t *card = NULL;
    esp_err_t e = esp_vfs_fat_sdmmc_mount("/sdcard", &host, &slot, &cfg, &card);
    if (e != ESP_OK) {
        char s[90];
        snprintf(s, sizeof(s), "ERROR sd_mount %s\n", esp_err_to_name(e));
        reply(s);
        return;
    }
    mounted_card = card;
    char info[256];
    snprintf(info, sizeof(info),
             "READY LFM230 card=%.5s sectors=%u clock_khz=%u internal=%u psram=%u tile_bytes=%d context=%d kv_bytes=%u kv_bits=%d kv_backend=%s kv_ram_bytes=%u kv_shared=%d\n",
             card->cid.name, (unsigned)card->csd.capacity, (unsigned)card->real_freq_khz,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM), LFM_IO, LFM_CONTEXT,
             (unsigned)(LFM_KV_BYTES), LFM_KV_BITS, KV_BACKEND_NAME, (unsigned)KV_RESIDENT_BYTES, LFM_KV_SHARED);
    reply(info);
    for (;;) {
        char line[96];
        size_t n = 0;
        char c;
        while (n < sizeof(line) - 1) {
            if (receive(&c, 1)) {
                n = 0;
                continue;
            }
            if (c == '\n')
                break;
            if (c != '\r')
                line[n++] = c;
        }
        line[n] = 0;
        if (!strcmp(line, "KVCLOSE")) {
#if LFM_KV_FILE
            runner->failed = 1;
            logits_ready = false;
            verified_handle = false;
            reply(lfm_kv_file_close(kv_file) ? "ERROR kv_close\n" : "KV_CLOSED\n");
#else
            reply("KV_RAM\n");
#endif
        } else if (!strcmp(line, "KVINFO")) {
            char status[256];
#if LFM_KV_FILE
            struct stat st;
            long size = kv_file->opened && !fstat(kv_file->fd, &st) ? (long)st.st_size : -1;
            snprintf(status, sizeof(status), "KV backend=FILE resident=%u file_bytes=%ld failed=%d read_bytes=%llu write_bytes=%llu read_us=%llu write_us=%llu\n",
                     (unsigned)sizeof(*kv_file), size, kv_file->failed,
                     (unsigned long long)kv_file->read_bytes, (unsigned long long)kv_file->write_bytes,
                     (unsigned long long)kv_file->read_us, (unsigned long long)kv_file->write_us);
#else
            snprintf(status, sizeof(status), "KV backend=RAM resident=%u\n", (unsigned)LFM_KV_BYTES);
#endif
            reply(status);
        } else if (!strncmp(line, "PHASE ", 6)) {
            display_set_phase(line + 6);
            reply("PHASE_OK\n");
        } else if (!strcmp(line, "FRAME")) {
            reply("FRAME_READY\n");
            if (receive(buffer, 17280)) {
                reply("ERROR frame_short\n");
                continue;
            }
            display_frame(buffer);
            reply("FRAME_OK\n");
        } else if (!strncmp(line, "TEXT ", 5)) {
            char *end;
            unsigned long n = strtoul(line + 5, &end, 10);
            if (*end || n > 512) {
                reply("ERROR text_size\n");
                continue;
            }
            char text[513];
            reply("TEXT_READY\n");
            if (receive(text, n)) {
                reply("ERROR text_short\n");
                continue;
            }
            text[n] = 0;
            display_text(text);
            reply("TEXT_OK\n");
        } else if (!strncmp(line, "MODE ", 5)) {
            if (!strcmp(line + 5, "PIPELINE"))
                use_pipeline = true;
            else if (!strcmp(line + 5, "SERIAL"))
                use_pipeline = false;
            else {
                reply("ERROR mode\n");
                continue;
            }
            reply("MODE_OK next_OPEN\n");
        } else if (!strcmp(line, "INFO"))
            reply(info);
        else if (!strcmp(line, "LIST")) {
            DIR *d = opendir("/sdcard");
            if (d) {
                struct dirent *ent;
                while ((ent = readdir(d))) {
                    reply(ent->d_name);
                    reply("\n");
                }
                closedir(d);
            }
            reply("LIST_END\n");
        } else if (!strcmp(line, "TEST_TX")) {
            reply("TEST_TX 262144\n");
            for (size_t i = 0; i < 32768; i++)
                buffer[i] = (uint8_t)(i % 251);
            int bad = 0;
            for (int i = 0; i < 8; i++)
                if (send_bytes(buffer, 32768)) {
                    bad = 1;
                    break;
                }
            reply(bad ? "ERROR short_tx\n" : "\nTEST_TX_END\n");
        } else if (!strcmp(line, "PUT"))
            upload();
        else if (!strcmp(line, "IOBENCH"))
            benchmark_io();
        else if (!strcmp(line, "RESET"))
            reset_verified_model();
        else if (!strcmp(line, "OPEN"))
            open_model();
        else if (!strncmp(line, "STEP ", 5) || !strncmp(line, "PREFILL ", 8)) {
            int tokens[LFM_BATCH], count = 0, logits = -1;
            bool prefill = !strncmp(line, "PREFILL ", 8);
            if (prefill) {
                char *cursor = line + 8, *end;
                long flag = strtol(cursor, &end, 10);
                if (end == cursor || flag < 0 || flag > 1) {
                    reply("ERROR command\n");
                    continue;
                }
                logits = flag;
                cursor = end;
                while (*cursor && count < LFM_BATCH) {
                    long token = strtol(cursor, &end, 10);
                    if (end == cursor || token < 0 || token >= LFM_VOCAB) {
                        count = 0;
                        break;
                    }
                    tokens[count++] = token;
                    cursor = end;
                }
                if (*cursor || count < 1) {
                    reply("ERROR command\n");
                    continue;
                }
            } else {
                char extra;
                if (sscanf(line + 5, "%d %d %c", tokens, &logits, &extra) != 2) {
                    reply("ERROR command\n");
                    continue;
                }
                count = 1;
            }
            if (!model || logits < 0 || logits > 1) {
                reply("ERROR command\n");
                continue;
            }
            active_count = count;
            int64_t t = esp_timer_get_time();
            step_started = t;
            last_draw = 0;
            display_status("STARTING TOKEN", 0);
            uint64_t before = runner->read_bytes;
            uint64_t io_before = runner->io_us, mm_before = runner->matmul_us;
#if LFM_KV_FILE
            uint64_t kr = kv_file->read_bytes, kw = kv_file->write_bytes;
            uint64_t krt = kv_file->read_us, kwt = kv_file->write_us;
#endif
            if (prefill ? lfm_prefill(runner, batch_workspace, tokens, count, logits)
                        : lfm_step(runner, tokens[0], logits)) {
                reply("ERROR step\n");
                display_status("STOPPED / ERROR", 0);
                continue;
            }
            logits_ready = logits;
            int best = logits ? lfm_argmax(runner) : -1;
            char out[220];
            snprintf(out, sizeof(out),
                     "TOKEN position=%d argmax=%d logit=%.9g us=%lld bytes=%llu\n",
                     runner->position, best, logits ? runner->logits[best] : 0,
                     (long long)(esp_timer_get_time() - t),
                     (unsigned long long)(runner->read_bytes - before));
            reply(out);
            char profile[180];
            snprintf(profile, sizeof(profile),
                     "PROFILE io_us=%llu matmul_us=%llu stack=%u worker_stack=%u\n",
                     (unsigned long long)(runner->io_us - io_before),
                     (unsigned long long)(runner->matmul_us - mm_before),
                     (unsigned)uxTaskGetStackHighWaterMark(NULL), pipeline_stack_free());
            reply(profile);
#if LFM_KV_FILE
            snprintf(profile, sizeof(profile), "KVPROFILE read_bytes=%llu write_bytes=%llu read_us=%llu write_us=%llu\n",
                     (unsigned long long)(kv_file->read_bytes - kr), (unsigned long long)(kv_file->write_bytes - kw),
                     (unsigned long long)(kv_file->read_us - krt), (unsigned long long)(kv_file->write_us - kwt));
            reply(profile);
#endif
            float celsius;
            if (temperature && temperature_sensor_get_celsius(temperature, &celsius) == ESP_OK) {
                char thermal[40];
                snprintf(thermal, sizeof(thermal), "THERMAL chip_c=%.1f\n", (double)celsius);
                reply(thermal);
            }
            display_status(logits ? "TOKEN READY" : "PREFILL", 100);
            display_metrics(runner->position, esp_timer_get_time() - t);
        } else if (!strcmp(line, "LOGITS") && model && runner->position > 0 && !runner->failed &&
                   logits_ready) {
            reply("LOGITS 262144\n");
            if (send_bytes(runner->logits, sizeof(runner->logits))) {
                reply("ERROR short_tx\n");
                continue;
            }
            reply("\nLOGITS_END\n");
        } else
            reply("ERROR unknown_command\n");
    }
}
