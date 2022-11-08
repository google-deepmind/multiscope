package tree

import (
	"context"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/ui"

	"github.com/pkg/errors"
	"google.golang.org/protobuf/encoding/prototext"
	"honnef.co/go/js/dom/v2"
)

// Element maintains a HTML tree.
type Element struct {
	ui       ui.UI
	settings *settings

	p    *dom.HTMLParagraphElement
	root *Node
}

// NewElement returns a new tree element displaying the Multiscope tree.
func NewElement(mui ui.UI) (*Element, error) {
	el := &Element{
		ui: mui,
		p:  mui.Owner().Doc().CreateElement("p").(*dom.HTMLParagraphElement),
	}
	el.settings = newSettings(el)
	el.settings.registerListener()
	if el.root != nil {
		return el, nil
	}
	// No settings has been loaded. Force a refresh.
	if err := el.refresh(nil); err != nil {
		return nil, err
	}
	return el, nil
}

func (el *Element) refresh(src any) error {
	if src == el.settings {
		return nil
	}
	var err error
	el.root, err = el.newNode(nil, "")
	if err != nil {
		return err
	}
	if err := el.root.expand(); err != nil {
		return err
	}
	el.p.AppendChild(el.root.childrenList)
	return nil
}

func (el *Element) fetchNode(path []string) (*treepb.Node, error) {
	req := &treepb.NodeStructRequest{Paths: []*treepb.NodePath{
		{Path: path},
	}}
	rep, err := el.ui.TreeClient().GetNodeStruct(context.Background(), req)
	if err != nil {
		return nil, err
	}
	if len(rep.Nodes) != 1 {
		return nil, errors.Errorf("wrong number of nodes returned by the server for the request:\n%s\nServer returned %d nodes, but want %d nodes", prototext.Format(req), len(rep.Nodes), 1)
	}
	return rep.Nodes[0], nil
}

// Root HTML element.
func (el *Element) Root() dom.Node {
	return el.p
}
