#pragma once

#include "src/app/types.h"

void print_run_header(const Args& args, int rank);
void print_verify_header(int rank);
void print_profile_header(const Args& args, int rank);
void print_profile_summary(const Args& args, const ProfileStats& stats, int rank);
void append_walltime_log(const Args& args, const ProfileStats& stats);
