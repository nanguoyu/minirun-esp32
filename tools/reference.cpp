#include "ggml-backend.h"
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
int main(int argc, char **argv) {
    if (argc < 4)
        return 2;
    ggml_backend_load_all();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.vocab_only = (!strcmp(argv[3], "--tokenize") || !strcmp(argv[3], "--decode"));
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU),
                                    nullptr};
    mp.devices = devices;
    auto model = llama_model_load_from_file(argv[1], mp);
    if (!model)
        return 3;
    auto vocab = llama_model_get_vocab(model);
    if (!strcmp(argv[3], "--decode")) {
        for (int i = 4; i < argc; i++) {
            char piece[1024];
            int n = llama_token_to_piece(vocab, atoi(argv[i]), piece, sizeof(piece), 0, true);
            if (n < 0)
                return 6;
            fwrite(piece, 1, n, stdout);
        }
        llama_model_free(model);
        return 0;
    }
    std::vector<llama_token> ids;
    if (!strcmp(argv[3], "--text") || !strcmp(argv[3], "--tokenize")) {
        if (argc != 5)
            return 2;
        constexpr int max_text_tokens = 1024;
        ids.resize(max_text_tokens);
        int n = llama_tokenize(vocab, argv[4], strlen(argv[4]), ids.data(), ids.size(), true, true);
        if (n < 0) {
            fprintf(stderr, "Tokenized input exceeds the %d-token helper limit\n", max_text_tokens);
            return 4;
        }
        ids.resize(n);
    } else {
        for (int i = 3; i < argc; i++)
            ids.push_back(atoi(argv[i]));
    }
    printf("TOKENS");
    for (auto t : ids)
        printf(" %d", t);
    printf("\n");
    if (!strcmp(argv[3], "--tokenize")) {
        llama_model_free(model);
        return 0;
    }
    auto cp = llama_context_default_params();
    cp.offload_kqv = false;
    cp.op_offload = false;
    cp.n_ctx = getenv("LFM_REFERENCE_CONTEXT") ? atoi(getenv("LFM_REFERENCE_CONTEXT")) : 128;
    cp.n_batch = 1;
    cp.n_ubatch = 1;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    auto ctx = llama_init_from_model(model, cp);
    if (!ctx)
        return 4;
    for (auto t : ids) {
        auto b = llama_batch_get_one(&t, 1);
        if (llama_decode(ctx, b))
            return 5;
    }
    auto logits = llama_get_logits_ith(ctx, -1);
    int n = llama_vocab_n_tokens(vocab), best = 0;
    for (int i = 1; i < n; i++)
        if (logits[i] > logits[best])
            best = i;
    printf("argmax=%d logit=%.9g\n", best, logits[best]);
    FILE *f = fopen(argv[2], "wb");
    if (!f || fwrite(logits, 4, n, f) != (size_t)n)
        return 6;
    fclose(f);
    char s[1024];
    int len = llama_token_to_piece(vocab, best, s, sizeof(s), 0, true);
    if (len > 0)
        printf("piece=%.*s\n", len, s);
    int extra = getenv("LFM_GREEDY_TOKENS") ? atoi(getenv("LFM_GREEDY_TOKENS")) : 0;
    for (int step = 0; step < extra; step++) {
        printf("GENERATED %d\n", best);
        if (llama_vocab_is_eog(vocab, best))
            break;
        llama_token t = best;
        auto batch = llama_batch_get_one(&t, 1);
        if (llama_decode(ctx, batch))
            return 5;
        logits = llama_get_logits_ith(ctx, -1);
        best = 0;
        for (int i = 1; i < n; i++)
            if (logits[i] > logits[best])
                best = i;
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
