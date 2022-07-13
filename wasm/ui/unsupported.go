package ui

import (
	"fmt"
	treepb "multiscope/protos/tree_go_proto"
	"path/filepath"

	"honnef.co/go/js/dom/v2"
)

type unsupported struct {
	node    *treepb.Node
	element dom.HTMLElement
}

const unsupportedHTML = `
no panel to display %q data
`

func newUnsupported(dbd *Dashboard, node *treepb.Node) (*Panel, error) {
	dsp := &unsupported{
		node:    node,
		element: dbd.NewErrorElement(),
	}
	dsp.element.SetInnerHTML(fmt.Sprintf(unsupportedHTML, node.Mime))
	desc := NewDescriptor(dsp, dsp.element, nil, node.Path)
	return dbd.NewPanel(filepath.Join(node.Path.Path...), desc)
}

// Display the latest data.
func (dsp *unsupported) Display(data *treepb.NodeData) error {
	return nil
}
