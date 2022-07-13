// Package writers provides the gRPC registry registering all the server writers.
package writers

import (
	"multiscope/internal/server/root"
	"multiscope/internal/server/treeservice"
	"multiscope/internal/server/writers/base"
	"multiscope/internal/server/writers/scalar"
	"multiscope/internal/server/writers/text"
	"multiscope/internal/server/writers/ticker"
)

// All returns all the writers available by default.
func All() []treeservice.RegisterServiceCallback {
	return []treeservice.RegisterServiceCallback{
		base.RegisterService,
		root.RegisterService,
		scalar.RegisterService,
		text.RegisterService,
		ticker.RegisterService,
	}
}

// NewRegistry returns a new registry with all the writers already registered.
func NewRegistry(state treeservice.State) *treeservice.Registry {
	registry := treeservice.NewRegistry(state)
	for _, srv := range All() {
		registry.RegisterService(srv)
	}
	return registry
}
