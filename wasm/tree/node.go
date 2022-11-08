// Package tree builds the tree on the left of the UI.
package tree

import (
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/ui"

	"honnef.co/go/js/dom/v2"
)

// Node in the tree.
type Node struct {
	el   *Element
	node *treepb.Node

	item         dom.HTMLElement
	caret        *dom.HTMLAnchorElement
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
	owner := el.ui.Owner()
	n.item = owner.Doc().CreateElement("li").(dom.HTMLElement)
	n.item.Class().Add("tree-list")
	if node.HasChildren {
		n.caret = owner.NewTextButton(n.item, "▶ ", func(ev dom.Event) {
			if err := n.expand(); err != nil {
				el.ui.DisplayErr(err)
			}
		})
	}
	if ui.Builder(n.node.Mime) == nil {
		owner.CreateTextNode(n.item, name)
	} else {
		owner.NewTextButton(n.item, name, n.openPanel)
	}

	if node.HasChildren && el.settings.isVisible(node.Path.Path) {
		n.el.ui.Run(n.expand)
	}
	return n, nil
}

func (n *Node) openPanel(dom.Event) {
	if err := n.el.ui.Dashboard().OpenPanel(n.node); err != nil {
		n.el.ui.DisplayErr(err)
	}
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
	n.childrenList = n.el.ui.Owner().CreateChild(n.item, "ul").(*dom.HTMLUListElement)
	n.childrenList.Class().Add("tree-list")
	for _, child := range n.node.Children {
		childNode, err := n.el.newNode(n, child.Name)
		if err != nil {
			return err
		}
		n.children[child.Name] = childNode
		n.childrenList.AppendChild(childNode.item)
	}
	if n.caret != nil {
		n.show()
	}
	return nil
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
