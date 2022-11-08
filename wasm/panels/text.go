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
	ui.RegisterBuilder(mime.PlainText, newText)
}

type text struct {
	root *dom.HTMLParagraphElement
}

func newText(dbd ui.Dashboard, node *treepb.Node) (ui.Panel, error) {
	dsp := &text{
		root: dbd.UI().Owner().Doc().CreateElement("p").(*dom.HTMLParagraphElement),
	}
	dsp.root.Class().Add("text-content")
	desc := dbd.NewDescriptor(node, nil, node.Path)
	return NewPanel(filepath.Join(node.Path.Path...), desc, dsp)
}

// Display the latest data.
func (dsp *text) Display(data *treepb.NodeData) error {
	dsp.root.SetInnerHTML(fmt.Sprintf("<pre>%s</pre>", string(data.GetRaw())))
	return nil
}

func (dsp *text) Root() dom.HTMLElement {
	return dsp.root
}
