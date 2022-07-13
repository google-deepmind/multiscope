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
func RunGRPC(srv *treeservice.TreeServer, wg *sync.WaitGroup, addr string) (int, error) {
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return -1, fmt.Errorf("cannot start testing server: %w", err)
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
	port := listener.Addr().(*net.TCPAddr).Port
	return port, nil
}
