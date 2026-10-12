#include "lfm.h"
#include <stdlib.h>
#include <time.h>
int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: host model.gguf logits.bin token_ids...\n");
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    Lfm *m = calloc(1, sizeof(*m));
    unsigned char *io = malloc(LFM_IO);
    LfmKV *kv = calloc(LFM_KV_COUNT, sizeof(LfmKV));
    if (!f || !m || !io || !kv || lfm_init(m, f, io, kv))
        return 2;
    int batch = getenv("LFM_PREFILL_BATCH") ? atoi(getenv("LFM_PREFILL_BATCH")) : 0;
    if (batch) {
        if (batch < 1 || batch > LFM_BATCH)
            return 2;
        LfmBatch *w = calloc(1, sizeof(*w));
        if (!w)
            return 2;
        for (int i = 3; i < argc;) {
            int n = argc - i;
            if (n > batch)
                n = batch;
            int ids[LFM_BATCH];
            for (int j = 0; j < n; j++)
                ids[j] = atoi(argv[i + j]);
            if (lfm_prefill(m, w, ids, n, i + n == argc))
                return 3;
            i += n;
        }
        free(w);
    } else {
        for (int i = 3; i < argc; i++) {
            clock_t t = clock();
            if (lfm_step(m, atoi(argv[i]), i == argc - 1))
                return 3;
            fprintf(stderr, "position=%d cpu_seconds=%.3f bytes=%llu\n", m->position,
                    (double)(clock() - t) / CLOCKS_PER_SEC, (unsigned long long)m->read_bytes);
        }
    }
    printf("argmax=%d logit=%.9g\n", lfm_argmax(m), m->logits[lfm_argmax(m)]);
    FILE *o = fopen(argv[2], "wb");
    if (!o || fwrite(m->logits, 4, 65536, o) != 65536)
        return 4;
    fclose(o);
    int extra = getenv("LFM_GREEDY_TOKENS") ? atoi(getenv("LFM_GREEDY_TOKENS")) : 0;
    for (int step = 0; step < extra; step++) {
        int best = lfm_argmax(m);
        printf("GENERATED %d\n", best);
        if (best == 2 || best == 7)
            break;
        if (lfm_step(m, best, 1))
            return 3;
    }
    fclose(f);
    free(kv);
    free(io);
    free(m);
    return 0;
}
