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

// Command server runs the Multiscope gRPC/http server.
package main

import (
	"flag"
	"fmt"
	"log"
	"sync"

	"multiscope/internal/server/scope"
)

var (
	httpPort = flag.Int("http_port", scope.DefaultPort, "http port")
	grpcPort = flag.Int("grpc_port", 0, "gRPC port")
	local    = flag.Bool("local", true, "local connections only")
)

func toAddr(port int) string {
	addr := fmt.Sprintf(":%d", port)
	if *local {
		addr = "localhost" + addr
	}
	return addr
}

func main() {
	flag.Parse()
	srv := scope.NewSingleton()
	wg := sync.WaitGroup{}

	httpAddr := toAddr(*httpPort)
	fmt.Printf("Running http server on %q\n", httpAddr)
	if err := scope.RunHTTP(srv, &wg, httpAddr); err != nil {
		log.Fatalf("cannot start http server: %v", err)
	}

	grpcAddr, err := scope.RunGRPC(srv, &wg, toAddr(*grpcPort))
	if err != nil {
		log.Fatalf("cannot start gRPC server: %v", err)
	}
	fmt.Printf("Running gRPC server on %q\n", grpcAddr.String())

	wg.Wait()
}
