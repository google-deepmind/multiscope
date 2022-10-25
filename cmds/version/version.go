// Executable version prints the current version of the gRPC API.
// This is used to import Multiscope into google3 where the WASM file
// is suffixed with the version.
package main

import (
	"fmt"
	"multiscope/internal/version"
)

func main() {
	fmt.Println(version.Version)
}
