// Package version checks the version of the gRPC API.
package version

import (
	"context"
	"fmt"
	"multiscope/internal/httpgrpc"
	"multiscope/protos"
	rootpb "multiscope/protos/root_go_proto"
	uipb "multiscope/protos/ui_go_proto"
)

// Check the version of the gRPC API.
// Returns an error if it does not match.
func Check(addr *uipb.Connect) error {
	conn := httpgrpc.Connect(addr.Scheme, addr.Host)
	rootClient := rootpb.NewRootClient(conn)
	resp, err := rootClient.GetVersion(context.Background(), &rootpb.GetVersionRequest{})
	if err != nil {
		return fmt.Errorf("cannot get the server gRPC API: %v", err)
	}
	if resp.Version != protos.Version {
		return fmt.Errorf("frontend client version %q does not match the server gRPC API version %q.\nYou may need to recompile the WASM frontend client, restart the server, then reload the page", resp.Version, protos.Version)
	}
	return nil
}
