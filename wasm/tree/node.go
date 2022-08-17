package tree

import (
	treepb "multiscope/protos/tree_go_proto"

	"honnef.co/go/js/dom/v2"
)

// Node in the tree.
type Node struct {
	el   *Element
	node *treepb.Node

	item         dom.HTMLElement
	caret        *dom.HTMLSpanElement
	childrenList *dom.HTMLUListElement
	children     map[string]*Node
}

func (el *Element) newNode(parent *Node, name string) (*Node, error) {
	path := []string{}
	if parent != nil {
		path = append([]string{}, parent.node.Path.Path...)
		path = append(path, name)
	}

	node, err := el.fetchNode(path)
	if err != nil {
		return nil, err
	}
	n := &Node{
		el:   el,
		node: node,
	}
	if parent == nil {
		n.item = el.p
		return n, nil
	}
	n.item = el.ui.Owner().CreateElement("li").(dom.HTMLElement)
	n.item.Class().Add("tree-list")
	if node.HasChildren {
		n.appendExpandButton()
	}
	n.item.AppendChild(el.ui.Owner().CreateTextNode(name))

	if node.HasChildren && el.settings.isVisible(node.Path.Path) {
		n.el.ui.Run(n.expand)
	}
	return n, nil
}

func (n *Node) expand() error {
	if !n.node.HasChildren {
		return nil
	}
	if n.children != nil {
		if n.el.settings.isVisible(n.node.Path.Path) {
			n.hide()
		} else {
			n.show()
		}
		return nil
	}
	n.children = make(map[string]*Node)
	n.childrenList = n.el.ui.Owner().CreateElement("ul").(*dom.HTMLUListElement)
	n.childrenList.Class().Add("tree-list")
	for _, child := range n.node.Children {
		childNode, err := n.el.newNode(n, child.Name)
		if err != nil {
			return err
		}
		n.children[child.Name] = childNode
		n.childrenList.AppendChild(childNode.item)
	}
	n.item.AppendChild(n.childrenList)
	if n.caret != nil {
		n.show()
	}
	return nil
}

func (n *Node) appendExpandButton() {
	n.caret = n.el.ui.Owner().CreateElement("span").(*dom.HTMLSpanElement)
	n.caret.Class().Add("tree-caret")
	n.caret.SetInnerHTML("▶ ")
	n.caret.AddEventListener("click", false, func(ev dom.Event) {
		n.el.ui.Run(n.expand)
	})
	n.item.AppendChild(n.caret)
}

func (n *Node) hide() {
	n.childrenList.Style().SetProperty("display", "none", "")
	n.el.settings.hideNode(n.node.Path.Path)
	n.caret.SetInnerHTML("▶ ")
}

func (n *Node) show() {
	n.childrenList.Style().SetProperty("display", "", "")
	n.el.settings.showNode(n.node.Path.Path)
	n.caret.SetInnerHTML("▼ ")
}
