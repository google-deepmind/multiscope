#ifndef MULTISCOPE_CLIENTS_CPP_EXAMPLES_TICKER_H_
#define MULTISCOPE_CLIENTS_CPP_EXAMPLES_TICKER_H_

#include <cstdint>

#include "base/integral_types.h"

namespace multiscope {

// Run the ticker example for max_steps steps.
void RunTickerExample(int32_t max_steps, bool multiscope_strict_mode = false);

}  // namespace multiscope

#endif  // MULTISCOPE_CLIENTS_CPP_EXAMPLES_TICKER_H_
