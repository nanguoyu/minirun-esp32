#include "kv_file.h"
#include "half.h"
#include <assert.h>
#include <fcntl.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static uint32_t rng = 42;
static float random_value(void) {
    rng = rng * 1664525u + 1013904223u;
    return ((int)(rng >> 16) - 32768) / 8192.0f;
}
static LfmKV pack(float x) {
#if LFM_KV_BITS == 16
    return lfm_half_pack(x);
#else
    return x;
#endif
}
static float unpack(LfmKV x) {
#if LFM_KV_BITS == 16
    return lfm_half_expand(x);
#else
    return x;
#endif
}
static int cancel(void *unused) { (void)unused; return 1; }
int main(void) {
    char directory[] = "/tmp/lfm-kv-unit-XXXXXX"; assert(mkdtemp(directory));
    char sentinel[256]; snprintf(sentinel, sizeof(sentinel), "%s/KV0000.TMP", directory);
    FILE *f = fopen(sentinel, "wb"); assert(f); assert(fwrite("keep", 1, 4, f) == 4); fclose(f);
    LfmKVFile *k = calloc(1, sizeof(*k)); Lfm *m = calloc(1, sizeof(*m));
    LfmKV *memory = calloc(LFM_KV_COUNT, sizeof(LfmKV));
    unsigned char *io = malloc(LFM_IO); assert(k && m && memory && io);
    assert(!lfm_kv_file_open(k, directory, io)); assert(strcmp(k->path, sentinel));
    m->attention_opaque = k;
    float q[1024], key[512], value[512], actual[1024], expected[1024], scores[LFM_CONTEXT];
    for (int pos = 0; pos < LFM_CONTEXT; pos++) {
        for (int layer = 0; layer < 6; layer++) {
            for (int d = 0; d < 1024; d++) q[d] = random_value();
            for (int d = 0; d < 512; d++) { key[d] = random_value(); value[d] = random_value(); }
            LfmKV *keys = memory + layer * 2 * LFM_CONTEXT * 512, *vals = keys + LFM_CONTEXT * 512;
            for (int d = 0; d < 512; d++) { keys[pos * 512 + d] = pack(key[d]); vals[pos * 512 + d] = pack(value[d]); }
            assert(!lfm_kv_file_attention(m, layer, pos, q, key, value, actual));
            for (int h = 0; h < 16; h++) {
                float maximum = -INFINITY, total = 0;
                for (int t = 0; t <= pos; t++) {
                    float sum = 0;
                    for (int d = 0; d < 64; d++) sum += q[h * 64 + d] * unpack(keys[t * 512 + (h / 2) * 64 + d]);
                    scores[t] = sum * 0.125f;
                    if (scores[t] > maximum) maximum = scores[t];
                }
                for (int t = 0; t <= pos; t++) { scores[t] = expf(scores[t] - maximum); total += scores[t]; }
                for (int d = 0; d < 64; d++) {
                    float sum = 0;
                    for (int t = 0; t <= pos; t++) sum += scores[t] / total * unpack(vals[t * 512 + (h / 2) * 64 + d]);
                    expected[h * 64 + d] = sum;
                }
            }
            assert(!memcmp(actual, expected, sizeof(actual)));
        }
    }
    assert(k->write_bytes == LFM_KV_BYTES && k->read_bytes > 0);
    printf("PASS bits=%d positions=%d all attention outputs bitwise; workspace=%zu read=%llu write=%llu\n", LFM_KV_BITS, LFM_CONTEXT, sizeof(*k), (unsigned long long)k->read_bytes, (unsigned long long)k->write_bytes);
    assert(!lfm_kv_file_close(k));
    /* A short historical read must fail and stay failed. */
    assert(!lfm_kv_file_open(k, directory, io)); m->failed = 0;
    for (int pos = 0; pos < (int)LFM_KV_PAGE_POS; pos++) assert(!lfm_kv_file_attention(m, 0, pos, q, key, value, actual));
    assert(!ftruncate(k->fd, 0));
    assert(lfm_kv_file_attention(m, 0, LFM_KV_PAGE_POS, q, key, value, actual) == -1 && m->failed && k->failed);
    uint64_t reads = k->read_bytes, writes = k->write_bytes;
    assert(lfm_kv_file_attention(m, 0, LFM_KV_PAGE_POS + 1, q, key, value, actual) == -1);
    assert(k->read_bytes == reads && k->write_bytes == writes); assert(!lfm_kv_file_close(k));
    /* Reject writes through a read-only stream; never fall back to a RAM history. */
    assert(!lfm_kv_file_open(k, directory, io)); m->failed = 0;
    int readonly = open(k->path, O_RDONLY); assert(readonly >= 0); close(k->fd); k->fd = readonly;
    for (int pos = 0; pos < (int)LFM_KV_PAGE_POS - 1; pos++) assert(!lfm_kv_file_attention(m, 0, pos, q, key, value, actual));
    assert(lfm_kv_file_attention(m, 0, LFM_KV_PAGE_POS - 1, q, key, value, actual) == -1 && m->failed);
    assert(!lfm_kv_file_close(k));
    assert(!lfm_kv_file_open(k, directory, io)); m->failed = 0; m->cancelled = cancel;
    assert(lfm_kv_file_attention(m, 0, 0, q, key, value, actual) == -1 && k->write_bytes == 0);
    assert(!lfm_kv_file_close(k));
    f = fopen(sentinel, "rb"); char check[4]; assert(f && fread(check, 1, 4, f) == 4 && !memcmp(check, "keep", 4)); fclose(f);
    unlink(sentinel); assert(!rmdir(directory)); free(k); free(m); free(memory); free(io);
    puts("PASS short read, write failure, cancellation and unrelated-file preservation");
}
