// Package version checks the version of the gRPC API.
package version

import (
	"context"
	"fmt"
	"multiscope/internal/httpgrpc"
	"multiscope/internal/version"
	rootpb "multiscope/protos/root_go_proto"
	uipb "multiscope/protos/ui_go_proto"
	"runtime/debug"

	"github.com/pkg/errors"
)

// Check the version of the gRPC API.
// Returns an error if it does not match.
func Check(addr *uipb.Connect) error {
	debug.PrintStack()
	conn := httpgrpc.Connect(addr.Scheme, addr.Host)
	rootClient := rootpb.NewRootClient(conn)
	resp, err := rootClient.GetVersion(context.Background(), &rootpb.GetVersionRequest{})
	if err != nil {
		return fmt.Errorf("cannot get the server gRPC API: %w", err)
	}
	if resp.Version != version.Version {
		return errors.Errorf("frontend client version %q does not match the server gRPC API version %q.\nYou may need to recompile the WASM frontend client, restart the server, then reload the page", resp.Version, version.Version)
	}
	return nil
}
