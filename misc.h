#pragma once

#include <limits>
#include <fenv.h>

typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned long long ulonglong;

static inline void enable_floating_point_exceptions() {
#ifndef NDEBUG
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif
}

const double epsilon = std::numeric_limits<double>::epsilon();
const double single_epsilon = std::numeric_limits<float>::epsilon();
