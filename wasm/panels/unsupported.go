package panels

import (
	"fmt"
	"multiscope/internal/mime"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/ui"
	"path/filepath"

	"honnef.co/go/js/dom/v2"
)

func init() {
	ui.RegisterDisplay(mime.Unsupported, newUnsupported)
}

type unsupported struct {
	node *treepb.Node
	root dom.HTMLElement
}

const unsupportedHTML = `
no panel to display %q data
`

func newUnsupported(dbd *ui.Dashboard, node *treepb.Node) (ui.Panel, error) {
	dsp := &unsupported{
		node: node,
		root: dbd.NewErrorElement(),
	}
	dsp.root.SetInnerHTML(fmt.Sprintf(unsupportedHTML, node.Mime))
	desc := dbd.NewDescriptor(dsp, nil, node.Path)
	return NewPanel(filepath.Join(node.Path.Path...), desc)
}

// Display the latest data.
func (dsp *unsupported) Display(data *treepb.NodeData) error {
	return nil
}

// Root returns the root element of the unsupported display.
func (dsp *unsupported) Root() dom.HTMLElement {
	return dsp.root
}
