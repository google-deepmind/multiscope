/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MULTISCOPE_CLIENTS_CPP_EXAMPLES_TICKER_H_
#define MULTISCOPE_CLIENTS_CPP_EXAMPLES_TICKER_H_

#include <cstdint>

#include "base/integral_types.h"

namespace multiscope {

// Run the ticker example for max_steps steps.
void RunTickerExample(int32_t max_steps, bool multiscope_strict_mode = false);

}  // namespace multiscope

#endif  // MULTISCOPE_CLIENTS_CPP_EXAMPLES_TICKER_H_
