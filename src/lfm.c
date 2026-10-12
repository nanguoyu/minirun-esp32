#include "lfm.h"
#include "half.h"
#include "model_layout.h"
#include <math.h>
#include <string.h>

static float half(const unsigned char *p) {
    return lfm_half_expand((uint16_t)(p[0] | ((uint16_t)p[1] << 8)));
}
static inline float kv_load(LfmKV value) {
#if LFM_KV_BITS == 16
    return lfm_half_expand(value);
#else
    return value;
#endif
}
static int kv_store(LfmKV *dst, const float *src) {
#if LFM_KV_BITS == 16
    for (int i = 0; i < 512; i++) {
        uint16_t h = lfm_half_pack(src[i]);
        if ((h & 0x7c00) == 0x7c00) return -1;
        dst[i] = h;
    }
#else
    memcpy(dst, src, 512 * sizeof(float));
#endif
    return 0;
}

int lfm_decode_row(uint32_t type, const unsigned char *s, float *y, size_t n) {
    if (type == 0) {
        memcpy(y, s, n * 4);
        return 0;
    }
    if (type == 2 && n % 32 == 0) {
        for (size_t k = 0; k < n; k += 32, s += 18) {
            float d = half(s);
            for (int j = 0; j < 16; j++) {
                y[k + j] = d * ((s[2 + j] & 15) - 8);
                y[k + j + 16] = d * ((s[2 + j] >> 4) - 8);
            }
        }
        return 0;
    }
    if (type == 14 && n % 256 == 0) {
        for (size_t k = 0; k < n; k += 256, s += 210) {
            float d = half(s + 208);
            for (int h = 0; h < 2; h++)
                for (int j = 0; j < 32; j++) {
                    const unsigned char *lo = s + 64 * h, *hi = s + 128 + 32 * h;
                    const int8_t *sc = (const int8_t *)(s + 192 + 8 * h);
                    int q[4] = {(lo[j] & 15) | ((hi[j] & 3) << 4),
                                (lo[j + 32] & 15) | (((hi[j] >> 2) & 3) << 4),
                                (lo[j] >> 4) | (((hi[j] >> 4) & 3) << 4),
                                (lo[j + 32] >> 4) | (((hi[j] >> 6) & 3) << 4)};
                    for (int t = 0; t < 4; t++)
                        y[k + 128 * h + j + 32 * t] = d * sc[j / 16 + 2 * t] * (q[t] - 32);
                }
        }
        return 0;
    }
    return -1;
}
int lfm_read(Lfm *m, uint32_t off, void *dst, size_t n) {
    if (m->failed || off > MODEL_BYTES || n > MODEL_BYTES - off ||
        (m->cancelled && m->cancelled(m->opaque))) {
        m->failed = 1;
        return -1;
    }
    uint64_t started = m->clock_us ? m->clock_us() : 0;
    if ((m->cursor != off && fseek(m->file, (long)off, SEEK_SET) != 0) ||
        fread(dst, 1, n, m->file) != n) {
        m->failed = 1;
        return -1;
    }
    if (m->clock_us)
        m->io_us += m->clock_us() - started;
    m->cursor = off + n;
    m->read_bytes += n;
    m->reads++;
    return 0;
}
static const Tensor *get(const char *name) {
    for (size_t i = 0; i < sizeof(tensors) / sizeof(tensors[0]); i++)
        if (!strcmp(tensors[i].name, name))
            return &tensors[i];
    return NULL;
}
static const Tensor *layer(int l, const char *suffix) {
    char name[100];
    snprintf(name, sizeof(name), "blk.%d.%s.weight", l, suffix);
    return get(name);
}
size_t lfm_row_bytes(const Tensor *t) {
    return t->type == 0    ? 4 * t->cols
           : t->type == 2  ? t->cols / 32 * 18
           : t->type == 14 ? t->cols / 256 * 210
                           : 0;
}
int lfm_compute_rows_many(uint32_t type, const unsigned char *data, uint32_t cols, uint32_t rows,
                          int count, const float *input, size_t xs, float *out, size_t os) {
    if (cols > 3072 || count < 1 || count > LFM_BATCH)
        return -1;
    float decoded[3072];
    Tensor t = {.cols = cols, .type = type};
    size_t rb = lfm_row_bytes(&t);
    if (!rb)
        return -1;
    for (uint32_t r = 0; r < rows; r++) {
        if (lfm_decode_row(type, data + r * rb, decoded, cols))
            return -1;
        for (int b = 0; b < count; b++) {
            float sum = 0;
            for (uint32_t k = 0; k < cols; k++)
                sum += decoded[k] * input[b * xs + k];
            out[b * os + r] = sum;
        }
    }
    return 0;
}
int lfm_compute_rows(uint32_t type, const unsigned char *data, uint32_t cols, uint32_t rows,
                     const float *input, float *out) {
    return lfm_compute_rows_many(type, data, cols, rows, 1, input, cols, out, rows);
}
static int matvec(Lfm *m, const Tensor *t, const float *x, float *out) {
    if (!t || t->cols > 3072)
        return m->failed = 1, -1;
    if (m->tensor_progress)
        m->tensor_progress(t, 0, t->rows, m->opaque);
    if (m->matvec_hook)
        return m->matvec_hook(m, t, x, out);
    float input[3072];
    memcpy(input, x, t->cols * sizeof(float));
    size_t rb = lfm_row_bytes(t);
    if (!rb || rb > LFM_IO)
        return m->failed = 1, -1;
    size_t block = LFM_IO / rb;
    for (size_t r = 0; r < t->rows; r += block) {
        size_t nr = t->rows - r < block ? t->rows - r : block;
        if (lfm_read(m, t->offset + r * rb, m->io, nr * rb))
            return -1;
        uint64_t started = m->clock_us ? m->clock_us() : 0;
        if (lfm_compute_rows(t->type, m->io, t->cols, nr, input, out + r))
            return m->failed = 1, -1;
        if (m->clock_us)
            m->matmul_us += m->clock_us() - started;
        if (m->tensor_progress)
            m->tensor_progress(t, r + nr, t->rows, m->opaque);
    }
    return 0;
}
static int rms(Lfm *m, const Tensor *t, const float *x, float *out, int n) {
    if (!t || t->type != 0 || t->cols != (unsigned)n || t->rows != 1)
        return m->failed = 1, -1;
    if (m->norm_offset != t->offset || m->norm_count != n) {
        if (lfm_read(m, t->offset, m->norm, n * 4))
            return -1;
        m->norm_offset = t->offset;
        m->norm_count = n;
    }
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += x[i] * x[i];
    float scale = 1.0f / sqrtf(sum / n + 1e-5f);
    for (int i = 0; i < n; i++)
        out[i] = x[i] * scale * m->norm[i];
    return 0;
}
static void rope(float *x, int heads, int pos) {
    for (int h = 0; h < heads; h++)
        for (int j = 0; j < 32; j++) {
            float theta = pos * powf(1000000.0f, -(2.0f * j) / 64), c = cosf(theta),
                  s = sinf(theta);
            float a = x[h * 64 + j], b = x[h * 64 + j + 32];
            x[h * 64 + j] = a * c - b * s;
            x[h * 64 + j + 32] = a * s + b * c;
        }
}
static int init_common(Lfm *m, FILE *f, unsigned char *io) {
    if (!m || !f || !io)
        return -1;
    memset(m, 0, sizeof(*m));
    m->file = f;
    m->cursor = UINT32_MAX;
    m->norm_offset = UINT32_MAX;
    m->io = io;
    if (fseek(f, 0, SEEK_END) || ftell(f) != MODEL_BYTES)
        return m->failed = 1, -1;
    unsigned char magic[4];
    if (lfm_read(m, 0, magic, 4) || memcmp(magic, "GGUF", 4))
        return m->failed = 1, -1;
    return 0;
}
int lfm_init(Lfm *m, FILE *f, unsigned char *io, LfmKV *kv) {
    if (!kv || init_common(m, f, io)) return -1;
    memset(kv, 0, LFM_KV_BYTES);
    m->kv = kv;
    return 0;
}
int lfm_init_external(Lfm *m, FILE *f, unsigned char *io, LfmAttention hook, void *opaque) {
    if (!hook || !opaque || init_common(m, f, io)) return -1;
    m->attention_hook = hook;
    m->attention_opaque = opaque;
    return 0;
}
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if ((x) != 0) {                                                                            \
            m->failed = 1;                                                                         \
            return -1;                                                                             \
        }                                                                                          \
    } while (0)
int lfm_step(Lfm *m, int token, int need_logits) {
    if (m->failed || token < 0 || token >= LFM_VOCAB || m->position >= LFM_CONTEXT)
        return m->failed = 1, -1;
    const Tensor *embed = get("token_embd.weight");
    size_t rb = lfm_row_bytes(embed);
    CHECK(lfm_read(m, embed->offset + token * rb, m->io, rb));
    CHECK(lfm_decode_row(embed->type, m->io, m->x, 1024));
    int ci = 0, ai = 0, pos = m->position;
    for (int l = 0; l < 14; l++) {
        CHECK(rms(m, layer(l, "attn_norm"), m->x, m->a, 1024));
        if (layer(l, "shortconv.conv")) {
            CHECK(matvec(m, layer(l, "shortconv.in_proj"), m->a, m->b));
            CHECK(lfm_read(m, layer(l, "shortconv.conv")->offset, m->c, 3 * 1024 * 4));
            for (int i = 0; i < 1024; i++) {
                float bx = m->b[i] * m->b[2048 + i];
                m->a[i] =
                    m->b[1024 + i] * (m->conv[ci][0][i] * m->c[3 * i] +
                                      m->conv[ci][1][i] * m->c[3 * i + 1] + bx * m->c[3 * i + 2]);
                m->conv[ci][0][i] = m->conv[ci][1][i];
                m->conv[ci][1][i] = bx;
            }
            CHECK(matvec(m, layer(l, "shortconv.out_proj"), m->a, m->b));
            ci++;
        } else {
            CHECK(matvec(m, layer(l, "attn_q"), m->a, m->b));
            CHECK(matvec(m, layer(l, "attn_k"), m->a, m->c));
            CHECK(matvec(m, layer(l, "attn_v"), m->a, m->c + 512));
            for (int h = 0; h < 16; h++)
                CHECK(rms(m, layer(l, "attn_q_norm"), m->b + h * 64, m->b + h * 64, 64));
            for (int h = 0; h < 8; h++)
                CHECK(rms(m, layer(l, "attn_k_norm"), m->c + h * 64, m->c + h * 64, 64));
            rope(m->b, 16, pos);
            rope(m->c, 8, pos);
            if (m->attention_hook) {
                CHECK(m->attention_hook(m, ai, pos, m->b, m->c, m->c + 512, m->a));
            } else {
                LfmKV *keys = m->kv + ai * 2 * LFM_CONTEXT * 512, *vals = keys + LFM_CONTEXT * 512;
                CHECK(kv_store(keys + pos * 512, m->c));
                CHECK(kv_store(vals + pos * 512, m->c + 512));
                for (int h = 0; h < 16; h++) {
                    float max = -INFINITY, total = 0;
                    for (int t = 0; t <= pos; t++) {
                        float s = 0;
                        for (int d = 0; d < 64; d++)
                            s += m->b[h * 64 + d] * kv_load(keys[t * 512 + (h / 2) * 64 + d]);
                        s *= 0.125f;
                        m->scores[t] = s;
                        if (s > max)
                            max = s;
                    }
                    for (int t = 0; t <= pos; t++) {
                        m->scores[t] = expf(m->scores[t] - max);
                        total += m->scores[t];
                    }
                    for (int d = 0; d < 64; d++) {
                        float s = 0;
                        for (int t = 0; t <= pos; t++)
                            s += m->scores[t] / total * kv_load(vals[t * 512 + (h / 2) * 64 + d]);
                        m->a[h * 64 + d] = s;
                    }
                }
            }
            CHECK(matvec(m, layer(l, "attn_output"), m->a, m->b));
            ai++;
        }
        for (int i = 0; i < 1024; i++)
            m->x[i] += m->b[i];
        CHECK(rms(m, layer(l, "ffn_norm"), m->x, m->a, 1024));
        CHECK(matvec(m, layer(l, "ffn_gate"), m->a, m->b));
        CHECK(matvec(m, layer(l, "ffn_up"), m->a, m->c));
        for (int i = 0; i < 2560; i++)
            m->b[i] = (m->b[i] / (1.0f + expf(-m->b[i]))) * m->c[i];
        CHECK(matvec(m, layer(l, "ffn_down"), m->b, m->a));
        for (int i = 0; i < 1024; i++)
            m->x[i] += m->a[i];
        if (m->progress)
            m->progress(l, m->opaque);
    }
    if (need_logits) {
        CHECK(rms(m, get("token_embd_norm.weight"), m->x, m->a, 1024));
        CHECK(matvec(m, embed, m->a, m->logits));
        for (int i = 0; i < LFM_VOCAB; i++)
            if (!isfinite(m->logits[i]))
                return m->failed = 1, -1;
    }
    m->position++;
    return 0;
}
int lfm_argmax(const Lfm *m) {
    int b = 0;
    for (int i = 1; i < LFM_VOCAB; i++)
        if (m->logits[i] > m->logits[b])
            b = i;
    return b;
}
/* Layer-wise causal prefill: weights are read once per bounded microbatch. */
static int matvec_many(Lfm *m, const Tensor *t, int count, const float *x, size_t xs, float *out,
                       size_t os) {
    if (!t || t->cols > 3072 || count < 1 || count > LFM_BATCH)
        return -1;
    if (m->tensor_progress)
        m->tensor_progress(t, 0, t->rows, m->opaque);
    if (m->many_hook)
        return m->many_hook(m, t, count, x, xs, out, os);
    size_t rb = lfm_row_bytes(t);
    if (!rb || rb > LFM_IO)
        return -1;
    for (size_t row = 0; row < t->rows;) {
        size_t n = LFM_IO / rb;
        if (n > t->rows - row)
            n = t->rows - row;
        CHECK(lfm_read(m, t->offset + row * rb, m->io, n * rb));
        uint64_t start = m->clock_us ? m->clock_us() : 0;
        CHECK(lfm_compute_rows_many(t->type, m->io, t->cols, n, count, x, xs, out + row, os));
        if (m->clock_us)
            m->matmul_us += m->clock_us() - start;
        row += n;
        if (m->tensor_progress)
            m->tensor_progress(t, row, t->rows, m->opaque);
    }
    return 0;
}
int lfm_prefill(Lfm *m, LfmBatch *w, const int *tokens, int count, int need_logits) {
    if (!w || m->failed || count < 1 || count > LFM_BATCH || m->position + count > LFM_CONTEXT)
        return m->failed = 1, -1;
    for (int t = 0; t < count; t++)
        if (tokens[t] < 0 || tokens[t] >= LFM_VOCAB)
            return m->failed = 1, -1;
    const Tensor *embed = get("token_embd.weight");
    size_t rb = lfm_row_bytes(embed);
    for (int t = 0; t < count; t++) {
        CHECK(lfm_read(m, embed->offset + tokens[t] * rb, m->io, rb));
        CHECK(lfm_decode_row(embed->type, m->io, w->x[t], 1024));
    }
    int ci = 0, ai = 0, base = m->position;
    for (int l = 0; l < 14; l++) {
        for (int t = 0; t < count; t++)
            CHECK(rms(m, layer(l, "attn_norm"), w->x[t], w->a[t], 1024));
        if (layer(l, "shortconv.conv")) {
            CHECK(
                matvec_many(m, layer(l, "shortconv.in_proj"), count, w->a[0], 3072, w->b[0], 3072));
            CHECK(lfm_read(m, layer(l, "shortconv.conv")->offset, m->c, 3 * 1024 * 4));
            for (int t = 0; t < count; t++)
                for (int i = 0; i < 1024; i++) {
                    float bx = w->b[t][i] * w->b[t][2048 + i];
                    w->a[t][i] = w->b[t][1024 + i] *
                                 (m->conv[ci][0][i] * m->c[3 * i] +
                                  m->conv[ci][1][i] * m->c[3 * i + 1] + bx * m->c[3 * i + 2]);
                    m->conv[ci][0][i] = m->conv[ci][1][i];
                    m->conv[ci][1][i] = bx;
                }
            CHECK(matvec_many(m, layer(l, "shortconv.out_proj"), count, w->a[0], 3072, w->b[0],
                              3072));
            ci++;
        } else {
            CHECK(matvec_many(m, layer(l, "attn_q"), count, w->a[0], 3072, w->b[0], 3072));
            CHECK(matvec_many(m, layer(l, "attn_k"), count, w->a[0], 3072, w->c[0], 3072));
            CHECK(matvec_many(m, layer(l, "attn_v"), count, w->a[0], 3072, w->c[0] + 512, 3072));
            for (int t = 0; t < count; t++) {
                int pos = base + t;
                for (int h = 0; h < 16; h++)
                    CHECK(rms(m, layer(l, "attn_q_norm"), w->b[t] + h * 64, w->b[t] + h * 64, 64));
                for (int h = 0; h < 8; h++)
                    CHECK(rms(m, layer(l, "attn_k_norm"), w->c[t] + h * 64, w->c[t] + h * 64, 64));
                rope(w->b[t], 16, pos);
                rope(w->c[t], 8, pos);
                if (m->attention_many_hook) continue;
                if (m->attention_hook) {
                    CHECK(m->attention_hook(m, ai, pos, w->b[t], w->c[t], w->c[t] + 512, w->a[t]));
                } else {
                    LfmKV *keys = m->kv + ai * 2 * LFM_CONTEXT * 512;
                    LfmKV *vals = keys + LFM_CONTEXT * 512;
                    CHECK(kv_store(keys + pos * 512, w->c[t]));
                    CHECK(kv_store(vals + pos * 512, w->c[t] + 512));
                    for (int h = 0; h < 16; h++) {
                        float max = -INFINITY, total = 0;
                        for (int j = 0; j <= pos; j++) {
                            float s = 0;
                            for (int d = 0; d < 64; d++)
                                s += w->b[t][h * 64 + d] * kv_load(keys[j * 512 + (h / 2) * 64 + d]);
                            s *= 0.125f;
                            m->scores[j] = s;
                            if (s > max)
                                max = s;
                        }
                        for (int j = 0; j <= pos; j++) {
                            m->scores[j] = expf(m->scores[j] - max);
                            total += m->scores[j];
                        }
                        for (int d = 0; d < 64; d++) {
                            float s = 0;
                            for (int j = 0; j <= pos; j++)
                                s += m->scores[j] / total * kv_load(vals[j * 512 + (h / 2) * 64 + d]);
                            w->a[t][h * 64 + d] = s;
                        }
                    }
                }
            }
            if (m->attention_many_hook)
                CHECK(m->attention_many_hook(m, ai, base, count, w->b[0], 3072, w->c[0], w->c[0]+512, 3072, w->a[0], 3072));
            CHECK(matvec_many(m, layer(l, "attn_output"), count, w->a[0], 3072, w->b[0], 3072));
            ai++;
        }
        for (int t = 0; t < count; t++) {
            for (int i = 0; i < 1024; i++)
                w->x[t][i] += w->b[t][i];
            CHECK(rms(m, layer(l, "ffn_norm"), w->x[t], w->a[t], 1024));
        }
        CHECK(matvec_many(m, layer(l, "ffn_gate"), count, w->a[0], 3072, w->b[0], 3072));
        CHECK(matvec_many(m, layer(l, "ffn_up"), count, w->a[0], 3072, w->c[0], 3072));
        for (int t = 0; t < count; t++)
            for (int i = 0; i < 2560; i++)
                w->b[t][i] = (w->b[t][i] / (1.0f + expf(-w->b[t][i]))) * w->c[t][i];
        CHECK(matvec_many(m, layer(l, "ffn_down"), count, w->b[0], 3072, w->a[0], 3072));
        for (int t = 0; t < count; t++)
            for (int i = 0; i < 1024; i++)
                w->x[t][i] += w->a[t][i];
        if (m->progress)
            m->progress(l, m->opaque);
    }
    memcpy(m->x, w->x[count - 1], sizeof(m->x));
    if (need_logits) {
        CHECK(rms(m, get("token_embd_norm.weight"), m->x, m->a, 1024));
        CHECK(matvec(m, embed, m->a, m->logits));
        for (int i = 0; i < LFM_VOCAB; i++)
            if (!isfinite(m->logits[i]))
                return m->failed = 1, -1;
    }
    m->position += count;
    return 0;
}
