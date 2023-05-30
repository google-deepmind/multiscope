// Copyright 2023 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
