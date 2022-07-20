#include <cstdint>

#include "base/init_google.h"
#include "multiscope/clients/cpp/examples/scalar.h"

int main(int argc, char *argv[]) {
  InitGoogle(argv[0], &argc, &argv, true);
  multiscope::RunScalarExample(INT32_MAX);
}
