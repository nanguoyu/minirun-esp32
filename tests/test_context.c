#include "lfm.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
/* Sequential versus causal microbatch through the last allocated position. */
int main(int argc, char **argv) {
    assert(argc == 3);
    const size_t bytes = LFM_KV_BYTES;
    Lfm *a = calloc(1, sizeof(*a)), *b = calloc(1, sizeof(*b));
    LfmBatch *w = calloc(1, sizeof(*w));
    unsigned char *ia = malloc(LFM_IO), *ib = malloc(LFM_IO);
    LfmKV *ka = malloc(bytes), *kb = malloc(bytes);
    FILE *fa = fopen(argv[1], "rb"), *fb = fopen(argv[1], "rb");
    assert(a && b && w && ia && ib && ka && kb && fa && fb);
    assert(!lfm_init(a, fa, ia, ka) && !lfm_init(b, fb, ib, kb));
    const int sequence[] = {1, 6, 6423, 708, 22998, 768, 3290, 4161, 1337, 768,
        15481, 523, 7, 708, 6, 64015, 708, 1286, 779, 14587, 4887, 803, 51957,
        522, 52323, 521, 1695, 25583, 2644, 1886, 1264, 576, 4643, 871};
    for (int base = 0; base < LFM_CONTEXT;) {
        int ids[8], n = LFM_CONTEXT - base;
        if (n > 8) n = 8;
        /* Include a batch that starts exactly at 64, then a single position. */
        if (base == 64) n = 1;
        for (int i = 0; i < n; i++) {
            ids[i] = sequence[(base + i) % (sizeof(sequence) / sizeof(sequence[0]))];
            assert(!lfm_step(a, ids[i], i == n - 1));
        }
        assert(!lfm_prefill(b, w, ids, n, 1));
        assert(a->position == b->position && !memcmp(ka, kb, bytes));
        assert(!memcmp(a->conv, b->conv, sizeof(a->conv)));
        assert(!memcmp(a->x, b->x, sizeof(a->x)));
        assert(!memcmp(a->logits, b->logits, sizeof(a->logits)));
        base += n;
        if (base == 64 || base == 65 || base == LFM_CONTEXT) {
            char path[512];
            assert(snprintf(path, sizeof(path), "%s-%d.f32", argv[2], base) < (int)sizeof(path));
            FILE *out = fopen(path, "wb");
            assert(out && fwrite(a->logits, sizeof(float), LFM_VOCAB, out) == LFM_VOCAB);
            fclose(out);
            printf("PASS context=%d position=%d state_and_logits_bitwise\n", LFM_CONTEXT, base);
            fflush(stdout);
        }
    }
    uint64_t reads = a->reads, other_reads = b->reads;
    int token = 1;
    assert(lfm_step(a, token, 1) == -1 && a->failed && a->reads == reads);
    assert(lfm_prefill(b, w, &token, 1, 1) == -1 && b->failed && b->reads == other_reads);
    assert(!lfm_init(a, fa, ia, ka) && a->position == 0 && !a->failed);
    for (size_t i = 0; i < bytes; i++) assert(((unsigned char *)ka)[i] == 0);
    puts("PASS overflow refused before IO; reset clears full KV allocation");
    fclose(fa); fclose(fb); free(a); free(b); free(w); free(ia); free(ib); free(ka); free(kb);
}
