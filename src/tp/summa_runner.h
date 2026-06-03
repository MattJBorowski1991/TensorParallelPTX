#pragma once

#include "src/app/types.h"

bool validate_runtime(const Args& args, int num_gpus);
ProfileStats run_profile(const Args& args, const DistContext& dist);
