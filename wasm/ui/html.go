package ui

import (
	"multiscope/internal/mime"
	treepb "multiscope/protos/tree_go_proto"
	"path/filepath"

	"honnef.co/go/js/dom/v2"
)

func init() {
	RegisterDisplay(mime.HTMLParent, newHTML)
}

type html struct {
	el *dom.HTMLParagraphElement
}

func newHTML(dbd *Dashboard, node *treepb.Node) (*Panel, error) {
	dsp := &html{}
	dsp.el = dbd.Owner().CreateElement("p").(*dom.HTMLParagraphElement)
	dsp.el.Class().Add("panel-html")
	htmlPath := &treepb.NodePath{
		Path: append(append([]string{}, node.Path.Path...), mime.NodeNameHTML),
	}
	cssPath := &treepb.NodePath{
		Path: append(append([]string{}, node.Path.Path...), mime.NodeNameCSS),
	}
	desc := NewDescriptor(dsp, dsp.el, nil, node.Path, htmlPath, cssPath)
	return dbd.NewPanel(filepath.Join(node.Path.Path...), desc)
}

// Display the latest data.
func (dsp *html) Display(data *treepb.NodeData) error {
	raw := string(data.GetRaw())
	switch data.Mime {
	case mime.HTMLText:
		dsp.el.SetInnerHTML(raw)
	case mime.CSSText:
		dsp.el.Style().Set("cssText", raw)
	}
	return nil
}
