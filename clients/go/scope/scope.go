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

// Package scope provides the main public API to the Multiscope Go client.
package scope

import (
	"context"
	"errors"
	"fmt"
	"multiscope/clients/go/reflect"
	"multiscope/clients/go/remote"
	"multiscope/internal/server/events"
	"multiscope/lib/tensor"
	treepb "multiscope/protos/tree_go_proto"
	"time"
)

// DefaultPort is the default port for Multiscope.
const DefaultPort = 5972

type (
	// Event is an alias to a generic stream.Event.
	Event = treepb.Event

	// Ticker synchronizes data in the tree.
	Ticker = remote.Ticker

	// ScalarWriter writes maps of key to scalars.
	ScalarWriter = remote.ScalarWriter

	// ImageWriter writes images.
	ImageWriter = remote.ImageWriter

	// HTMLWriter writes html and css.
	HTMLWriter = remote.HTMLWriter

	// TextWriter writes plain text.
	TextWriter = remote.TextWriter

	// TensorWriter writes tensor data.
	TensorWriter = remote.TensorWriter[float32]

	// Group is a directory node in the tree.
	Group = remote.Group

	// Path is a path in the Multiscope tree.
	Path = remote.Path
)

var scopeClient *remote.Client

// Start starts the Multiscope server. Doesn't take a context for
// convenience as this is a research-facing debugging library.
func Start(httpPort int, local bool) error {
	// Start the HTTP server on the specified port.
	httpAddr := fmt.Sprintf(":%d", httpPort)
	if local {
		httpAddr = "localhost" + httpAddr
	}
	grpcAddr, err := remote.StartServer(httpAddr)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	clt, err := remote.Connect(ctx, grpcAddr)
	if err != nil {
		return fmt.Errorf("connecting to multiscope: %v", err)
	}
	SetGlobalClient(clt)
	fmt.Println("Listening to", httpAddr)
	return nil
}

// SetGlobalClient sets the global Multiscope client.
func SetGlobalClient(clt *remote.Client) {
	scopeClient = clt
}

// NewGroup creates a new group to which children can be added.
func NewGroup(name string, parent remote.Path) (*remote.Group, error) {
	clt, err := Client()
	if err != nil {
		return nil, err
	}
	return remote.NewGroup(clt, name, parent)
}

// NewScalarWriter creates a new writer to write scalars to Multiscope.
func NewScalarWriter(name string, parent remote.Path) (*remote.ScalarWriter, error) {
	clt, err := Client()
	if err != nil {
		return nil, err
	}
	return remote.NewScalarWriter(clt, name, parent)
}

// NewPlayer creates a new player in Multiscope.
func NewPlayer(name string, ignorePause bool, parent remote.Path) (*remote.Player, error) {
	clt, err := Client()
	if err != nil {
		return nil, err
	}
	return remote.NewPlayer(clt, name, ignorePause, parent)
}

// NewTicker creates a new Ticker in Multiscope.
func NewTicker(name string, parent remote.Path) (*Ticker, error) {
	clt, err := Client()
	if err != nil {
		return nil, err
	}
	return remote.NewTicker(clt, name, parent)
}

// NewTextWriter creates a new writer to display raw text in Multiscope.
func NewTextWriter(name string, parent remote.Path) (*remote.TextWriter, error) {
	clt, err := Client()
	if err != nil {
		return nil, err
	}
	return remote.NewTextWriter(clt, name, parent)
}

// NewHTMLWriter creates a new writer to display HTML in Multiscope.
func NewHTMLWriter(name string, parent remote.Path) (*remote.HTMLWriter, error) {
	clt, err := Client()
	if err != nil {
		return nil, err
	}
	return remote.NewHTMLWriter(clt, name, parent)
}

// NewImageWriter creates a new writer to display images in Multiscope.
func NewImageWriter(name string, parent remote.Path) (*remote.ImageWriter, error) {
	clt, err := Client()
	if err != nil {
		return nil, err
	}
	return remote.NewImageWriter(clt, name, parent)
}

// NewTensorWriter creates a new writer to display images in Multiscope.
func NewTensorWriter[T tensor.Supported](name string, parent remote.Path) (*remote.TensorWriter[T], error) {
	clt, err := Client()
	if err != nil {
		return nil, err
	}
	return remote.NewTensorWriter[T](clt, name, parent)
}

// EventsManager returns the registry mapping path to callback of the main Multiscope remote.
func EventsManager() *events.Registry {
	return scopeClient.EventsManager()
}

// Client returns the Multiscope singleton remote.
// It returns an error if Multiscope has not been initialized.
func Client() (*remote.Client, error) {
	if scopeClient == nil {
		return nil, errors.New("multiscope client is nil (did you call scope.Start()?)")
	}
	return scopeClient, nil
}

// Reflect builds a Multiscope tree by parsing a Go instance.
func Reflect(root remote.Node, name string, obj any) (remote.Node, error) {
	if root == nil {
		clt, err := Client()
		if err != nil {
			return nil, err
		}
		root = remote.Root(clt)
	}
	return reflect.On(root, name, obj)
}
