#pragma once
#include "lfm.h"
int pipeline_init(void);
int pipeline_matvec(Lfm *m, const Tensor *t, const float *input, float *output);
unsigned pipeline_stack_free(void);

int pipeline_many(Lfm *m, const Tensor *t, int count, const float *x, size_t xs, float *out,
                  size_t os);

uint8_t *pipeline_buffer(void);
