#ifndef LFM_H
#define LFM_H
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#ifndef LFM_CONTEXT
#define LFM_CONTEXT 128
#endif
#if LFM_CONTEXT != 64 && LFM_CONTEXT != 128 && LFM_CONTEXT != 256 && LFM_CONTEXT != 512 && LFM_CONTEXT != 1024
#error "Supported context capacities: 64, 128, 256, 512, 1024"
#endif
#ifndef LFM_KV_BITS
#define LFM_KV_BITS 32
#endif
#if LFM_KV_BITS == 16
typedef uint16_t LfmKV;
#elif LFM_KV_BITS == 32
typedef float LfmKV;
#else
#error "KV storage must be 16 or 32 bits"
#endif
#define LFM_KV_COUNT (6u * 2u * LFM_CONTEXT * 512u)
#define LFM_KV_BYTES (LFM_KV_COUNT * sizeof(LfmKV))
#define LFM_DIM 1024
#define LFM_VOCAB 65536
#ifndef LFM_IO
#define LFM_IO 32768
#endif
#define LFM_BATCH 8
typedef struct {
    float x[LFM_BATCH][1024], a[LFM_BATCH][3072], b[LFM_BATCH][3072], c[LFM_BATCH][3072];
} LfmBatch;
/* Caller must verify the pinned snapshot SHA-256 before init. Init checks size/magic.
 * Buffers are caller owned. This adapter supports only the pinned 230M QAD geometry. */
typedef struct {
    const char *name;
    uint32_t offset, cols, rows, type;
} Tensor;
struct Lfm;
typedef int (*LfmAttention)(struct Lfm *, int, int, const float *, const float *, const float *, float *);
typedef int (*LfmAttentionMany)(struct Lfm *, int, int, int, const float *, size_t, const float *, const float *, size_t, float *, size_t);
typedef struct Lfm {
    FILE *file;
    unsigned char *io;
    LfmAttention attention_hook;
    LfmAttentionMany attention_many_hook;
    void *attention_opaque;
    LfmKV *kv; /* 6 * 2 * context * 512 */
    float conv[8][2][1024];
    float x[1024], a[3072], b[3072], c[3072], row[3072], norm[1024];
    float scores[LFM_CONTEXT], logits[65536];
    uint64_t read_bytes, reads, io_us, matmul_us;
    uint64_t (*clock_us)(void);
    int position, failed;
    uint32_t cursor, norm_offset;
    int norm_count;
    int (*matvec_hook)(struct Lfm *, const Tensor *, const float *, float *);
    int (*many_hook)(struct Lfm *, const Tensor *, int, const float *, size_t, float *, size_t);
    int (*cancelled)(void *);
    void *opaque;
    void (*progress)(int layer, void *);
    void (*tensor_progress)(const Tensor *, uint32_t, uint32_t, void *);
} Lfm;
int lfm_init(Lfm *m, FILE *file, unsigned char *io, LfmKV *kv);
int lfm_init_external(Lfm *m, FILE *f, unsigned char *io, LfmAttention hook, void *opaque);
int lfm_step(Lfm *m, int token, int need_logits);
int lfm_argmax(const Lfm *m);
int lfm_read(Lfm *m, uint32_t offset, void *dst, size_t bytes);
int lfm_decode_row(uint32_t type, const unsigned char *src, float *dst, size_t cols);

size_t lfm_row_bytes(const Tensor *t);
int lfm_compute_rows(uint32_t type, const unsigned char *data, uint32_t cols, uint32_t rows,
                     const float *input, float *out);

int lfm_compute_rows_many(uint32_t type, const unsigned char *data, uint32_t cols, uint32_t rows,
                          int count, const float *input, size_t xs, float *out, size_t os);
int lfm_prefill(Lfm *m, LfmBatch *w, const int *tokens, int count, int need_logits);
#endif
