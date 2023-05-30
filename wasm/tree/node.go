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
	n.item.Class().Add(toStyleClass(path))
	if node.HasChildren {
		n.caret = owner.NewTextButton(n.item, "▶ ", func(gui ui.UI, ev dom.Event) {
			if err := n.expand(); err != nil {
				gui.DisplayErr(err)
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

func (n *Node) openPanel(gui ui.UI, _ dom.Event) {
	if err := gui.Dashboard().OpenPanel(n.node); err != nil {
		gui.DisplayErr(err)
	}
}

func toStyleClass(path []string) string {
	if len(path) <= 1 {
		return "tree-root"
	}
	return "tree-list"
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
	n.childrenList.Class().Add(toStyleClass(n.node.Path.Path))
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
