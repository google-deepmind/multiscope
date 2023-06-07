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

// Package remote implements a Multiscope client to the Multiscope server.
package remote

import (
	"context"
	"fmt"
	"multiscope/internal/grpc/client"
	"multiscope/internal/server/events"
	"multiscope/internal/server/scope"
	rootpb "multiscope/protos/root_go_proto"
	rootpbgrpc "multiscope/protos/root_go_proto"
	pb "multiscope/protos/tree_go_proto"
	pbgrpc "multiscope/protos/tree_go_proto"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"sync"

	"github.com/pkg/errors"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// StartServer starts a Multiscope server and returns its URL to connect to it.
func StartServer(httpAddr string) (string, error) {
	srv := scope.NewServer()
	wg := sync.WaitGroup{}
	if err := scope.RunHTTP(srv, &wg, httpAddr); err != nil {
		return "", err
	}
	// Start the GRPC server on a random port.
	addr, err := scope.RunGRPC(srv, &wg, "localhost:0")
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("localhost:%d", addr.Port), nil
}

// Client is a Multiscope client writing data to a Multiscope server.
type Client struct {
	prefix  Path
	conn    grpc.ClientConnInterface
	client  pbgrpc.TreeClient
	treeID  *pb.TreeID
	display *Display
	events  *Events
	active  *Active
}

// Connect to a Multiscope server and returns a Multiscope remote.
func Connect(ctx context.Context, targetURL string, treeID *pb.TreeID) (*Client, error) {
	parsedURL, err := url.Parse(targetURL)
	if err != nil {
		return nil, err
	}
	var conn *grpc.ClientConn
	if parsedURL.Scheme == "unix" {
		conn, err = connectUDS(ctx, parsedURL)
	} else {
		conn, err = client.Connect(targetURL)
	}
	if err != nil {
		return nil, err
	}
	clt, err := NewClient(conn, treeID)
	if err != nil {
		return nil, err
	}
	if err := clt.setDefaultKeyDict(); err != nil {
		return nil, err
	}
	return clt, nil
}

func connectUDS(ctx context.Context, parsedURL *url.URL) (*grpc.ClientConn, error) {
	credentialOpt := grpc.WithTransportCredentials(insecure.NewCredentials())
	// Use  the third_party name resolution because grpcprod overwrites the naming
	// resolution but does not support UDS.
	// This allows multiscope to be used alongside code that imports grpcprod.
	nameResolutionOpt := grpc.WithContextDialer(func(ctx context.Context, addr string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, "unix", addr)
	})
	return grpc.DialContext(ctx, "passthrough://unix/"+parsedURL.Path,
		credentialOpt, nameResolutionOpt, grpc.WithBlock())
}

func newTreeID(clt *Client) (*pb.TreeID, error) {
	ctx := context.Background()
	resp, err := clt.client.GetTreeID(ctx, &pb.GetTreeIDRequest{})
	if err != nil {
		return nil, errors.Errorf("cannot fetch a tree ID: %v", err)
	}
	return resp.TreeId, nil
}

// NewClient returns a new Multiscope client given a stream client connected to a Multiscope server.
func NewClient(conn grpc.ClientConnInterface, treeID *pb.TreeID) (*Client, error) {
	clt := &Client{
		conn:   conn,
		client: pbgrpc.NewTreeClient(conn),
	}
	var err error
	if treeID == nil {
		clt.treeID, err = newTreeID(clt)
	}
	if err != nil {
		return nil, err
	}
	clt.active, err = newActive(clt)
	if err != nil {
		return nil, err
	}
	clt.events, err = newEvents(clt)
	if err != nil {
		return nil, err
	}
	clt.display = newDisplay(clt)
	return clt, nil
}

// Connection returns the connection from the client to the server.
func (clt *Client) Connection() grpc.ClientConnInterface {
	return clt.conn
}

// Prefix returns the prefix used by this client to create new nodes.
func (clt *Client) Prefix() Path {
	return clt.prefix
}

// ToChildPath returns a path to a parent for this client, that is
// the path will have a prefix if the client has one.
func (clt *Client) toChildPath(name string, parent Path) Path {
	if !parent.HasPrefix(clt.prefix) {
		parent = clt.prefix.Append(parent...)
	}
	return parent.Append(name)
}

// ResetState resets the state of the server, removing all writers.
func (clt *Client) ResetState() error {
	ctx := context.Background()
	_, err := clt.TreeClient().ResetState(ctx, &pb.ResetStateRequest{})
	return err
}

// TreeClient returns the client used to communicate with the Multiscope server.
func (clt *Client) TreeClient() pbgrpc.TreeClient {
	return clt.client
}

// TreeID returns a the ID of the tree to which this client is connected to.
func (clt *Client) TreeID() *pb.TreeID {
	return clt.treeID
}

// Active returns the registry to update the status of the node (active vs non-active).
func (clt *Client) Active() *Active {
	return clt.active
}

// EventsManager returns the event registry mapping paths to callbacks.
func (clt *Client) EventsManager() *events.Registry {
	return clt.events.reg
}

// NewChildClient creates a group node called name, then returns a new client that will create all nodes inside that group node.
func (clt *Client) NewChildClient(name string) (*Client, error) {
	grp, err := NewGroup(clt, name, nil)
	if err != nil {
		return nil, err
	}
	return &Client{
		conn:    clt.conn,
		treeID:  clt.treeID,
		client:  clt.client,
		prefix:  grp.Path(),
		events:  clt.events,
		display: clt.display,
		active:  clt.active,
	}, nil
}

// Display returns the client related to display on the dashboard.
func (clt *Client) Display() *Display {
	return clt.display
}

// SetGlobalName sets a global name on the server used to retrieve the settings on the UI.
func (clt *Client) SetGlobalName(name string) error {
	rootClt := rootpbgrpc.NewRootClient(clt.Connection())
	if _, err := rootClt.SetKeySettings(context.Background(), &rootpb.SetKeySettingsRequest{
		TreeId:      clt.TreeID(),
		KeySettings: name,
	}); err != nil {
		return errors.Errorf("cannot set global name: %v", err)
	}
	return nil
}

func (clt *Client) setDefaultKeyDict() error {
	name := fmt.Sprintf("go/client/default/%s", filepath.Base(os.Args[0]))
	return clt.SetGlobalName(name)
}
