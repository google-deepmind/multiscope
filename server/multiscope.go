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
	srv := scope.NewServer()
	wg := sync.WaitGroup{}

	httpAddr := toAddr(*httpPort)
	fmt.Printf("Running http server on %q\n", httpAddr)
	if err := scope.RunHTTP(srv, &wg, httpAddr); err != nil {
		log.Fatalf("cannot start http server: %v", err)
	}

	grpcAddr := toAddr(*grpcPort)
	if _, err := scope.RunGRPC(srv, &wg, grpcAddr); err != nil {
		log.Fatalf("cannot start gRPC server: %v", err)
	}

	wg.Wait()
}
