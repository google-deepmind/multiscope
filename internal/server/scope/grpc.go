// Copyright 2023 Google LLC
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

package scope

import (
	"fmt"
	"log"
	"multiscope/internal/server/treeservice"
	"net"
	"sync"

	"google.golang.org/grpc"
)

// RunGRPC starts a server given a root node and an event registry.
func RunGRPC(srv *treeservice.TreeServer, wg *sync.WaitGroup, addr string) (*net.TCPAddr, error) {
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("cannot start testing server: %w", err)
	}
	serv := grpc.NewServer()
	srv.RegisterServices(serv)
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := serv.Serve(listener); err != nil {
			log.Fatalf("cannot run GRPC server: %v", err)
		}
	}()
	return listener.Addr().(*net.TCPAddr), nil
}
