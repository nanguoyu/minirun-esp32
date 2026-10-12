#pragma once
#if defined(LFM_ALIGNED_MODEL) && LFM_ALIGNED_MODEL
#include "model_index_aligned.h"
#define MODEL_PATH "/sdcard/LFM4K.GGF"
#define MODEL_PART_PATH "/sdcard/LFM4K.PRT"
#else
#include "model_index.h"
#define MODEL_PATH "/sdcard/LFM230.GGF"
#define MODEL_PART_PATH "/sdcard/LFM230.PRT"
#endif
