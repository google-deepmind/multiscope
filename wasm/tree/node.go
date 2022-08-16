package tree

import (
	treepb "multiscope/protos/tree_go_proto"

	"honnef.co/go/js/dom/v2"
)

// Node in the tree.
type Node struct {
	node *treepb.Node

	list *dom.HTMLUListElement
}

func (el *Element) newNode(node *treepb.Node) *Node {
	n := &Node{
		node: node,
		list: el.ui.Owner().CreateElement("ul").(*dom.HTMLUListElement),
	}
	for _, child := range n.node.Children {
		if child == nil {
			continue
		}
		item := el.ui.Owner().CreateElement("li")
		item.SetInnerHTML(child.Name)
		n.list.AppendChild(item)
	}
	return n
}
