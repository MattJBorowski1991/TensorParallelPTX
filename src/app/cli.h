#pragma once

#include "src/app/types.h"

Args parse_args(int argc, char** argv);
bool validate_cli_args(const Args& args);
