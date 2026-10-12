#include "ggml.h"
#include "lfm.h"
#include "half.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
    unsigned char b[840];
    float a[1024], c[1024];
    for (int typ = 2; typ <= 14; typ += 12)
        for (int trial = 0; trial < 100; trial++) {
            for (size_t i = 0; i < sizeof(b); i++)
                b[i] = rand() & 255;
            int step = typ == 2 ? 18 : 210, nb = typ == 2 ? 32 : 4;
            for (int j = 0; j < nb; j++) {
                int k = j * step + (typ == 2 ? 0 : 208);
                b[k] = 0;
                b[k + 1] = 0x3c;
            }
            assert(lfm_decode_row(typ, b, a, 1024) == 0);
            ggml_get_type_traits(typ)->to_float(b, c, 1024);
            for (int i = 0; i < 1024; i++)
                assert(a[i] == c[i]);
        }
    for (unsigned scale = 0; scale < 65536; scale++) {
        memset(b, 0x91, 18);
        b[0] = scale & 255;
        b[1] = scale >> 8;
        assert(lfm_decode_row(2, b, a, 32) == 0);
        ggml_get_type_traits(GGML_TYPE_Q4_0)->to_float(b, c, 32);
        for (int i = 0; i < 32; i++)
            assert((isnan(a[i]) && isnan(c[i])) ||
                   (a[i] == c[i] && signbit(a[i]) == signbit(c[i])));
    }
    /* Compare rounding against the independent ggml conversion implementation. */
    uint32_t state = 42;
    for (unsigned i = 0; i < 1000000; i++) {
        state = state * 1664525u + 1013904223u;
        float value; memcpy(&value, &state, 4);
        if (!isnan(value)) assert(lfm_half_pack(value) == ggml_fp32_to_fp16(value));
    }
    for (unsigned h = 0; h < 65536; h++) {
        float value = lfm_half_expand((uint16_t)h);
        if (!isnan(value)) assert(lfm_half_pack(value) == h);
    }
    assert(lfm_decode_row(5, b, a, 1024) == -1);
    assert(lfm_decode_row(2, b, a, 31) == -1);
    Lfm *m = calloc(1, sizeof(*m));
    assert(m);
    m->file = tmpfile();
    assert(m->file);
    assert(lfm_read(m, 149081055, b, 2) == -1);
    m->failed = 0;
    assert(lfm_read(m, 0, b, 4) == -1);
    m->failed = 0;
    m->position = LFM_CONTEXT;
    assert(lfm_step(m, 1, 1) == -1);
    fclose(m->file);
    free(m);
    puts("core tests passed: quantization parity, bounds, short read, context refusal");
}
