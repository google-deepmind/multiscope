#include <cstdint>

#include "base/init_google.h"
#include "multiscope/clients/cpp/examples/ticker.h"

int main(int argc, char *argv[]) {
  InitGoogle(argv[0], &argc, &argv, true);
  multiscope::RunTickerExample(INT32_MAX);
}
