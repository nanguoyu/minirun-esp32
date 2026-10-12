#pragma once
#include <stdint.h>
int display_init(void);
void display_status(const char *text, int percent);
void display_text(const char *text);
void display_metrics(int position, uint64_t elapsed_us);

void display_frame(const uint8_t *gray4);

void display_span(int first, int last, uint64_t us);

void display_set_phase(const char *text);
