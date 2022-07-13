// Package scope provides the main public API to the Multiscope Go client.
package scope

import (
	"context"
	"errors"
	"fmt"
	"multiscope/clients/go/remote"
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

	// HTMLWriter writes html and css.
	HTMLWriter = remote.HTMLWriter

	// TextWriter writes plain text.
	TextWriter = remote.TextWriter

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

// NewScalarWriter creates a new writer to write scalars to Multiscope.
func NewScalarWriter(name string, parent remote.Path) (*remote.ScalarWriter, error) {
	clt, err := Client()
	if err != nil {
		return nil, err
	}
	return remote.NewScalarWriter(clt, name, parent)
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

// NewHTMLWriter creates a new writer to display images in Multiscope.
func NewHTMLWriter(name string, parent remote.Path) (*remote.HTMLWriter, error) {
	clt, err := Client()
	if err != nil {
		return nil, err
	}
	return remote.NewHTMLWriter(clt, name, parent)
}

// EventsManager returns the registry mapping path to callback of the main Multiscope remote.
func EventsManager() *remote.Events {
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
