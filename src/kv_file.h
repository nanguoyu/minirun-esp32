#ifndef LFM_KV_FILE_H
#define LFM_KV_FILE_H
#include "lfm.h"
#ifndef LFM_KV_SHARED
#define LFM_KV_SHARED 0
#endif
#define LFM_KV_QUERIES (LFM_KV_SHARED ? LFM_BATCH : 1)
#define LFM_KV_PAGE_BYTES 32768u
_Static_assert(LFM_IO >= LFM_KV_PAGE_BYTES, "KV page must fit the borrowed IO buffer");
#define LFM_KV_PAGE_POS (LFM_KV_PAGE_BYTES / (2u * 512u * sizeof(LfmKV)))
/* Ephemeral, exclusively owned cache. Zero-initialize before first open.
 * No existing path is overwritten or reused; close deletes only this session's file.
 * io is borrowed from the drained model pipeline, not concurrently accessed. */
typedef struct {
    int fd, opened;
    unsigned char *io;
    char path[256];
    int owned, failed, next[6], resident_page[6];
    LfmKV active[6][2][LFM_KV_PAGE_POS][512];
    float scores[LFM_KV_QUERIES][16][LFM_CONTEXT];
    uint64_t read_bytes, write_bytes, read_us, write_us, extent;
} LfmKVFile;
int lfm_kv_file_open(LfmKVFile *k, const char *directory, unsigned char *io);
int lfm_kv_file_close(LfmKVFile *k);
int lfm_kv_file_attention_many(Lfm *, int, int, int, const float *, size_t, const float *, const float *, size_t, float *, size_t);
int lfm_kv_file_attention(Lfm *m, int layer, int pos, const float *q,
                          const float *key, const float *value, float *out);
#endif
