#include "pipeline.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include <string.h>
typedef struct {
    Lfm *m;
    const uint8_t *data;
    float *out;
    uint32_t type, cols, rows;
    int slot, count;
    size_t out_stride;
} Job;
static QueueHandle_t queue;
static SemaphoreHandle_t finished[2];
static TaskHandle_t worker;
static uint8_t *arena, *second;
static float *input;
static int errors[2];
static uint64_t elapsed[2];
static void work(void *unused) {
    (void)unused;
    Job job;
    for (;;)
        if (xQueueReceive(queue, &job, portMAX_DELAY) == pdTRUE) {
            int64_t started = esp_timer_get_time();
            errors[job.slot] =
                lfm_compute_rows_many(job.type, job.data, job.cols, job.rows, job.count, input,
                                      job.cols, job.out, job.out_stride);
            elapsed[job.slot] = esp_timer_get_time() - started;
            xSemaphoreGive(finished[job.slot]);
        }
}
int pipeline_init(void) {
    arena = heap_caps_malloc(2 * LFM_IO, MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
    second = arena ? arena + LFM_IO : NULL;
    input =
        heap_caps_malloc(LFM_BATCH * 2560 * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    queue = xQueueCreate(1, sizeof(Job));
    finished[0] = xSemaphoreCreateBinary();
    finished[1] = xSemaphoreCreateBinary();
    if (!second || !input || !queue || !finished[0] || !finished[1])
        return -1;
    return xTaskCreatePinnedToCore(work, "lfm-matvec", 24576, NULL, 4, &worker, 1) == pdPASS ? 0
                                                                                             : -1;
}
uint8_t *pipeline_buffer(void) { return arena; }
unsigned pipeline_stack_free(void) { return worker ? uxTaskGetStackHighWaterMark(worker) : 0; }
int pipeline_matvec(Lfm *m, const Tensor *t, const float *x, float *out) {
    return pipeline_many(m, t, 1, x, t->cols, out, t->rows);
}
int pipeline_many(Lfm *m, const Tensor *t, int count, const float *x, size_t xs, float *out,
                  size_t os) {
    if (!t || t->cols > 2560 || count < 1 || count > LFM_BATCH)
        return -1;
    size_t rb = lfm_row_bytes(t);
    if (!rb || rb > LFM_IO)
        return -1;
    for (int b = 0; b < count; b++)
        memcpy(input + b * t->cols, x + b * xs, t->cols * sizeof(float));
    bool busy[2] = {false, false};
    uint32_t work_rows[2] = {0, 0};
    uint64_t row_cost = 0;
    int result = 0, slot = 0;
    for (size_t row = 0; row < t->rows;) {
        if (busy[slot]) {
            xSemaphoreTake(finished[slot], portMAX_DELAY);
            busy[slot] = false;
            m->matmul_us += elapsed[slot];
            uint64_t sample = work_rows[slot] ? elapsed[slot] * 1024 / work_rows[slot] : 0;
            if (sample)
                row_cost = row_cost ? (row_cost * 3 + sample) / 4 : sample;
            if (errors[slot]) {
                result = -1;
                break;
            }
        }
        uint8_t *bytes = slot ? second : m->io;
        size_t n = LFM_IO / rb;
        if (n > t->rows - row)
            n = t->rows - row;
        uint64_t io_before = m->io_us;
        if (lfm_read(m, t->offset + row * rb, bytes, n * rb)) {
            result = -1;
            break;
        }
        /* Balance IO + producer compute against worker compute. This only assigns
         * independent output rows; every row retains the same accumulation order. */
        uint64_t io_scaled = (m->io_us - io_before) * 1024;
        size_t main_rows = 0;
        if (count > 1 && row_cost && n * row_cost > io_scaled)
            main_rows = (n * row_cost - io_scaled) / (2 * row_cost);
        if (main_rows > n / 2)
            main_rows = n / 2;
        work_rows[slot] = n - main_rows;
        Job j = {.m = m,
                 .data = bytes + main_rows * rb,
                 .out = out + row + main_rows,
                 .type = t->type,
                 .cols = t->cols,
                 .rows = n - main_rows,
                 .slot = slot,
                 .count = count,
                 .out_stride = os};
        xQueueSend(queue, &j, portMAX_DELAY);
        busy[slot] = true;
        if (main_rows) {
            int64_t started = esp_timer_get_time();
            int rc = lfm_compute_rows_many(t->type, bytes, t->cols, main_rows, count, input,
                                           t->cols, out + row, os);
            m->matmul_us += esp_timer_get_time() - started;
            if (rc) {
                result = -1;
                break;
            }
        }
        row += n;
        if (m->tensor_progress)
            m->tensor_progress(t, row, t->rows, m->opaque);
        slot ^= 1;
    }
    /* Drain on success, error and cancellation before caller may reuse input/output. */
    for (int i = 0; i < 2; i++)
        if (busy[i]) {
            xSemaphoreTake(finished[i], portMAX_DELAY);
            m->matmul_us += elapsed[i];
            if (errors[i])
                result = -1;
        }
    return result;
}
