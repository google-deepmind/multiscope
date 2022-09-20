package treeservice

import (
	"google.golang.org/grpc"
)

type (
	// StateProvider provides a state.
	StateProvider func() State

	// RegisterServiceCallback to register a grpc service provided by a node.
	RegisterServiceCallback func(srv grpc.ServiceRegistrar, state StateProvider)

	// Registry of create handlers.
	Registry struct {
		// a handle to the current server state
		state State
		// Services provided by the different node types
		services []RegisterServiceCallback
	}
)

// ReplaceState returns a new Registry that uses the provided state. This is
// useful for implementing the ResetState grpc method.
func (r *Registry) ReplaceState(state State) *Registry {
	return &Registry{
		state:    state,
		services: r.services,
	}
}

// NewRegistry creates a new empty Registry.
func NewRegistry(state State) *Registry {
	return &Registry{state: state}
}

// RegisterService registers a service to register when the server is created.
func (r *Registry) RegisterService(service RegisterServiceCallback) {
	r.services = append(r.services, service)
}

// Services return the list of service to register to the gRPC server.
func (r *Registry) Services() []RegisterServiceCallback {
	return r.services
}
