#pragma once

#include "src/app/types.h"

// 1D (Megatron-style) TP profiling path: --tp-mode 1d-col | 1d-row.
// Isolated alternative to run_profile (SUMMA); same kernels underneath.
ProfileStats run_profile_1d(const Args& args, const DistContext& dist);
