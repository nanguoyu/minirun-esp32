#include "lfm.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
    assert(argc == 2);
    size_t kvsize = LFM_KV_BYTES;
    Lfm *a = calloc(1, sizeof(*a)), *b = calloc(1, sizeof(*b));
    LfmBatch *w = calloc(1, sizeof(*w));
    unsigned char *ia = malloc(LFM_IO), *ib = malloc(LFM_IO);
    LfmKV *ka = malloc(kvsize), *kb = malloc(kvsize);
    FILE *fa = fopen(argv[1], "rb"), *fb = fopen(argv[1], "rb");
    assert(a && b && w && ia && ib && ka && kb && fa && fb);
    int ids[] = {1, 6, 6423, 708, 3493, 856, 730, 527, 2011, 730, 527, 540, 7, 708, 6, 64015, 708};
    for (int chunk = 1; chunk <= 8; chunk *= 2) {
        assert(lfm_init(a, fa, ia, ka) == 0 && lfm_init(b, fb, ib, kb) == 0);
        for (int offset = 0; offset < 17;) {
            int n = 17 - offset;
            if (n > chunk)
                n = chunk;
            for (int i = 0; i < n; i++)
                assert(lfm_step(a, ids[offset + i], i == n - 1) == 0);
            assert(lfm_prefill(b, w, ids + offset, n, 1) == 0);
            assert(a->position == b->position);
            assert(!memcmp(ka, kb, kvsize));
            assert(!memcmp(a->conv, b->conv, sizeof(a->conv)));
            assert(!memcmp(a->x, b->x, sizeof(a->x)));
            assert(!memcmp(a->logits, b->logits, sizeof(a->logits)));
            offset += n;
        }
    }
    int bad = -1;
    assert(lfm_prefill(b, w, &bad, 1, 1) == -1);
    fclose(fa);
    fclose(fb);
    free(a);
    free(b);
    free(w);
    free(ia);
    free(ib);
    free(ka);
    free(kb);
    puts("PASS: batch sizes 1/2/4/8 preserve all KV, convolution state, hidden state and logits");
}
