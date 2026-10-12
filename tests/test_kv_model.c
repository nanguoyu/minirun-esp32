#include "kv_file.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
int main(int argc, char **argv) {
    assert(argc == 3);
    char directory[] = "/tmp/lfm-kv-model-XXXXXX"; assert(mkdtemp(directory));
    Lfm *a = calloc(1, sizeof(*a)), *b = calloc(1, sizeof(*b));
    LfmBatch *w = calloc(1, sizeof(*w)); LfmKVFile *cache = calloc(1, sizeof(*cache));
    unsigned char *ia = malloc(LFM_IO), *ib = malloc(LFM_IO);
    LfmKV *memory = malloc(LFM_KV_BYTES);
    FILE *fa = fopen(argv[1], "rb"), *fb = fopen(argv[1], "rb");
    assert(a && b && w && cache && ia && ib && memory && fa && fb);
    assert(!lfm_init(a, fa, ia, memory) && !lfm_kv_file_open(cache, directory, ib));
    assert(!lfm_init_external(b, fb, ib, lfm_kv_file_attention, cache));
#if LFM_KV_SHARED
    b->attention_many_hook = lfm_kv_file_attention_many;
#endif
    const int seq[] = {1,6,6423,708,22998,768,3290,4161,1337,768,15481,523,7,708,6,64015,708,
        1286,779,14587,4887,803,51957,522,52323,521,1695,25583,2644,1886,1264,576,4643,871};
    const int boundaries[] = {8,9,16,17,64,65,128,129,256,257,511,512,513,1023,1024};
    int base = 0, checkpoint = 0;
    while (base < LFM_CONTEXT) {
        int end = boundaries[checkpoint];
        if (end > LFM_CONTEXT) end = LFM_CONTEXT;
        int n = end - base; if (n > 8) n = 8;
        int ids[8]; for (int i = 0; i < n; i++) ids[i] = seq[(base+i) % (sizeof(seq)/sizeof(seq[0]))];
        int head = base + n == end;
        assert(!lfm_prefill(a, w, ids, n, head));
        assert(!lfm_prefill(b, w, ids, n, head));
        assert(!b->kv && !memcmp(a->conv, b->conv, sizeof(a->conv)) && !memcmp(a->x, b->x, sizeof(a->x)));
        assert(a->read_bytes == b->read_bytes);
        base += n;
        if (head) {
            assert(!memcmp(a->logits, b->logits, sizeof(a->logits)));
            char path[512]; assert(snprintf(path, sizeof(path), "%s-%d.f32", argv[2], base) < (int)sizeof(path));
            FILE *f = fopen(path, "wb"); assert(f && fwrite(a->logits, 4, LFM_VOCAB, f) == LFM_VOCAB); fclose(f);
            printf("PASS bits=%d position=%d RAM_FILE_bitwise kv_read=%llu kv_write=%llu workspace=%zu\n", LFM_KV_BITS, base, (unsigned long long)cache->read_bytes, (unsigned long long)cache->write_bytes, sizeof(*cache)); fflush(stdout);
            checkpoint++;
        }
    }
    int token = 1; uint64_t reads = cache->read_bytes, writes = cache->write_bytes;
    assert(lfm_prefill(b, w, &token, 1, 1) == -1 && b->failed);
    assert(cache->read_bytes == reads && cache->write_bytes == writes);
    assert(!lfm_kv_file_close(cache)); assert(!lfm_kv_file_open(cache, directory, ib));
    assert(!lfm_init_external(b, fb, ib, lfm_kv_file_attention, cache));
#if LFM_KV_SHARED
    b->attention_many_hook = lfm_kv_file_attention_many;
#endif
    assert(!b->position && !cache->next[0] && !cache->read_bytes && !cache->write_bytes);
    assert(!lfm_kv_file_close(cache)); assert(!rmdir(directory));
    fclose(fa); fclose(fb); free(a); free(b); free(w); free(cache); free(ia); free(ib); free(memory);
    puts("PASS external cache capacity refusal and fresh-session reset");
}
