package tree

import (
	"multiscope/wasm/ui"

	"honnef.co/go/js/dom/v2"
)

// Element maintains a HTML tree.
type Element struct {
	ui   ui.UI
	el   *dom.HTMLParagraphElement
	root *Node
}

func newElement(mui ui.UI) *Element {
	el := &Element{ui: mui}
	el.el = el.ui.Owner().CreateElement("p").(*dom.HTMLParagraphElement)
	return el
}

func (el *Element) setRoot(node *Node) {
	el.root = node
	el.el.AppendChild(el.root.list)
}
