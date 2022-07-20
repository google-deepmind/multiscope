#ifndef MULTISCOPE_CLIENTS_CPP_EXAMPLES_SCALAR_H_
#define MULTISCOPE_CLIENTS_CPP_EXAMPLES_SCALAR_H_

#include <cstdint>

#include "base/integral_types.h"

namespace multiscope {
// Run the scalar example for max_steps steps.
void RunScalarExample(int32_t max_steps, bool multiscope_strict_mode = false);
}  // namespace multiscope

#endif  // MULTISCOPE_CLIENTS_CPP_EXAMPLES_SCALAR_H_
