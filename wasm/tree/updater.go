package tree

import (
	"context"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/ui"

	"github.com/pkg/errors"
	"google.golang.org/protobuf/encoding/prototext"
	"honnef.co/go/js/dom/v2"
)

// Updater updates the tree displayed with the latest data from the server.
type Updater struct {
	el  *Element
	clt treepb.TreeClient
}

// NewUpdater maintains a Multiscope tree.
func NewUpdater(mui ui.UI) (*Updater, error) {
	up := &Updater{
		el:  newElement(mui),
		clt: mui.TreeClient(),
	}
	nodes, err := up.fetchNode([]*treepb.NodePath{
		&treepb.NodePath{},
	})
	if err != nil {
		return nil, err
	}
	up.el.setRoot(up.el.newNode(nodes[0]))
	return up, nil
}

func (up *Updater) fetchNode(paths []*treepb.NodePath) ([]*treepb.Node, error) {
	req := &treepb.NodeStructRequest{Paths: paths}
	rep, err := up.clt.GetNodeStruct(context.Background(), req)
	if err != nil {
		return nil, err
	}
	if len(rep.Nodes) != len(paths) {
		return nil, errors.Errorf("wrong number of nodes returned by the server for the request:\n%s\nServer returned %d nodes, but want %d nodes", prototext.Format(req), len(rep.Nodes), len(paths))
	}
	return rep.Nodes, nil
}

// Root HTML element.
func (up *Updater) Root() dom.Node {
	return up.el.el
}
