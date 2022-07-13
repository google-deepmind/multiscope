package ui

import (
	"fmt"
	"multiscope/internal/mime"
	treepb "multiscope/protos/tree_go_proto"
	"path/filepath"

	"honnef.co/go/js/dom/v2"
)

func init() {
	RegisterDisplay(mime.PlainText, newText)
}

type text struct {
	el *dom.HTMLParagraphElement
}

func newText(dbd *Dashboard, node *treepb.Node) (*Panel, error) {
	dsp := &text{}
	dsp.el = dbd.Owner().CreateElement("p").(*dom.HTMLParagraphElement)
	dsp.el.Class().Add("panel-text")
	desc := NewDescriptor(dsp, dsp.el, nil, node.Path)
	return dbd.NewPanel(filepath.Join(node.Path.Path...), desc)
}

// Display the latest data.
func (dsp *text) Display(data *treepb.NodeData) error {
	dsp.el.SetInnerHTML(fmt.Sprintf("<pre>%s</pre>", string(data.GetRaw())))
	return nil
}
