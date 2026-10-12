#include "kv_file.h"
#include "half.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

int lfm_kv_file_close(LfmKVFile *k) {
    int result = 0;
    if (k->opened && close(k->fd)) result = -1;
    k->opened = 0;
    k->fd = -1;
    if (k->owned) {
        if (unlink(k->path) && errno != ENOENT) result = -1;
        else k->owned = 0;
    }
    k->failed = 1;
    return result;
}
int lfm_kv_file_open(LfmKVFile *k, const char *directory, unsigned char *io) {
    if (!k || !directory || !io || k->opened || k->owned) return -1;
    memset(k, 0, sizeof(*k));
    k->io = io;
    k->fd = -1;
    for (int i = 0; i < 6; i++) k->resident_page[i] = -1;
    for (unsigned i = 0; i < 100; i++) {
        int n = snprintf(k->path, sizeof(k->path), "%s/KV%04u.TMP", directory, i);
        if (n < 0 || n >= (int)sizeof(k->path)) break;
        int fd = open(k->path, O_CREAT | O_EXCL | O_RDWR, 0600);
        if (fd < 0) {
            if (errno == EEXIST) continue;
            break;
        }
        k->owned = 1;
        k->opened = 1;
        k->fd = fd;
        return 0;
    }
    k->failed = 1;
    return -1;
}
static float load(LfmKV v) {
#if LFM_KV_BITS == 16
    return lfm_half_expand(v);
#else
    return v;
#endif
}
static int stopped(Lfm *m, LfmKVFile *k) {
    if (k->failed || !k->opened || (m->cancelled && m->cancelled(m->opaque))) {
        k->failed = m->failed = 1;
        return 1;
    }
    return 0;
}
static long offset(int layer, int page, int kind) {
    return (long)(page * 6 + layer) * LFM_KV_PAGE_BYTES + kind * (LFM_KV_PAGE_BYTES / 2);
}
static int append(Lfm *m, LfmKVFile *k, int layer, int pos, const float *key, const float *value) {
    if (stopped(m, k) || layer < 0 || layer >= 6 || pos < 0 || pos >= LFM_CONTEXT ||
        pos != k->next[layer]) return -1;
    int page = pos / LFM_KV_PAGE_POS, slot = pos % LFM_KV_PAGE_POS;
    if (page != k->resident_page[layer]) {
        memset(k->active[layer], 0, LFM_KV_PAGE_BYTES);
        k->resident_page[layer] = page;
    }
    for (int kind = 0; kind < 2; kind++) {
        const float *src = kind ? value : key;
#if LFM_KV_BITS == 16
        for (int d = 0; d < 512; d++) {
            uint16_t h = lfm_half_pack(src[d]);
            if ((h & 0x7c00) == 0x7c00) return -1;
            k->active[layer][kind][slot][d] = h;
        }
#else
        memcpy(k->active[layer][kind][slot], src, 512 * sizeof(float));
#endif
    }
    if (slot == (int)LFM_KV_PAGE_POS - 1) {
        uint64_t start = m->clock_us ? m->clock_us() : 0;
        struct stat st;
        if (fstat(k->fd, &st) || st.st_size < 0 || (uint64_t)st.st_size != k->extent) return -1;
        memcpy(k->io, k->active[layer], LFM_KV_PAGE_BYTES);
        long off = offset(layer, page, 0);
        if (lseek(k->fd, off, SEEK_SET) != off ||
            write(k->fd, k->io, LFM_KV_PAGE_BYTES) != LFM_KV_PAGE_BYTES) return -1;
        uint64_t end = (uint64_t)off + LFM_KV_PAGE_BYTES;
        if (end > k->extent) k->extent = end;
        k->write_bytes += LFM_KV_PAGE_BYTES;
        if (m->clock_us) k->write_us += m->clock_us() - start;
    }
    k->next[layer]++;
    return 0;
}
static const LfmKV *read_page(Lfm *m, LfmKVFile *k, int layer, int page, int kind) {
    if (stopped(m, k)) return NULL;
    if (page == k->resident_page[layer]) return &k->active[layer][kind][0][0];
    if (page < 0 || page > k->resident_page[layer]) return NULL;
    uint64_t start = m->clock_us ? m->clock_us() : 0;
    const size_t n = LFM_KV_PAGE_BYTES / 2;
    long off = offset(layer, page, kind);
    if (lseek(k->fd, off, SEEK_SET) != off || read(k->fd, k->io, n) != (ssize_t)n)
        return NULL;
    k->read_bytes += n;
    if (m->clock_us) k->read_us += m->clock_us() - start;
    return (const LfmKV *)k->io;
}
int lfm_kv_file_attention_many(Lfm *m, int layer, int first, int count,
        const float *q, size_t qs, const float *key, const float *value, size_t ks,
        float *out, size_t os) {
    LfmKVFile *k = m->attention_opaque;
    if (count < 1 || count > LFM_KV_QUERIES || first < 0 || first + count > LFM_CONTEXT)
        goto fail;
    for (int b = 0; b < count; b++)
        if (append(m, k, layer, first + b, key + b*ks, value + b*ks)) goto fail;
    int last = first + count - 1;
    float maxima[LFM_KV_QUERIES][16], totals[LFM_KV_QUERIES][16] = {{0}};
    for (int b = 0; b < count; b++)
        for (int h = 0; h < 16; h++) maxima[b][h] = -INFINITY;
    /* One read per page, shared by all queries. Each query sees only its causal prefix. */
    for (int base = 0; base <= last; base += LFM_KV_PAGE_POS) {
        const LfmKV *keys = read_page(m, k, layer, base / LFM_KV_PAGE_POS, 0);
        if (!keys) goto fail;
        for (int b = 0; b < count; b++) {
            int n = first + b - base + 1;
            if (n > (int)LFM_KV_PAGE_POS) n = LFM_KV_PAGE_POS;
            for (int h = 0; h < 16; h++)
                for (int t = 0; t < n; t++) {
                    float sum = 0;
                    for (int d = 0; d < 64; d++)
                        sum += q[b*qs + h*64 + d] * load(keys[t*512 + (h/2)*64 + d]);
                    sum *= 0.125f;
                    k->scores[b][h][base+t] = sum;
                    if (sum > maxima[b][h]) maxima[b][h] = sum;
                }
        }
    }
    for (int b = 0; b < count; b++) {
        for (int h = 0; h < 16; h++)
            for (int t = 0; t <= first+b; t++) {
                k->scores[b][h][t] = expf(k->scores[b][h][t] - maxima[b][h]);
                totals[b][h] += k->scores[b][h][t];
            }
        /* The same FP32 division result is reused for every value component. */
        for (int h = 0; h < 16; h++)
            for (int t = 0; t <= first+b; t++)
                k->scores[b][h][t] /= totals[b][h];
        memset(out + b*os, 0, 1024*sizeof(float));
    }
    for (int base = 0; base <= last; base += LFM_KV_PAGE_POS) {
        const LfmKV *values = read_page(m, k, layer, base / LFM_KV_PAGE_POS, 1);
        if (!values) goto fail;
        for (int b = 0; b < count; b++) {
            int n = first + b - base + 1;
            if (n > (int)LFM_KV_PAGE_POS) n = LFM_KV_PAGE_POS;
            for (int h = 0; h < 16; h++)
                for (int d = 0; d < 64; d++) {
                    float sum = out[b*os + h*64 + d];
                    for (int t = 0; t < n; t++)
                        sum += k->scores[b][h][base+t] *
                               load(values[t*512 + (h/2)*64 + d]);
                    out[b*os + h*64 + d] = sum;
                }
        }
    }
    return 0;
fail:
    k->failed = m->failed = 1;
    return -1;
}
int lfm_kv_file_attention(Lfm *m, int layer, int pos, const float *q,
                          const float *key, const float *value, float *out) {
    return lfm_kv_file_attention_many(m, layer, pos, 1, q, 0, key, value, 0, out, 0);
}
