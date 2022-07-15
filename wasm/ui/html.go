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
	el    *dom.HTMLParagraphElement
	style *dom.HTMLStyleElement
}

func newHTML(dbd *Dashboard, node *treepb.Node) (*Panel, error) {
	dsp := &html{}
	div := dbd.Owner().CreateElement("div").(*dom.HTMLDivElement)
	div.Class().Add("panel-html")
	dsp.style = dbd.Owner().CreateElement("style").(*dom.HTMLStyleElement)
	dsp.style.SetAttribute("scoped", "")
	div.AppendChild(dsp.style)
	dsp.el = dbd.Owner().CreateElement("p").(*dom.HTMLParagraphElement)
	div.AppendChild(dsp.el)
	htmlPath := &treepb.NodePath{
		Path: append(append([]string{}, node.Path.Path...), mime.NodeNameHTML),
	}
	cssPath := &treepb.NodePath{
		Path: append(append([]string{}, node.Path.Path...), mime.NodeNameCSS),
	}
	desc := NewDescriptor(dsp, div, nil, node.Path, htmlPath, cssPath)
	return dbd.NewPanel(filepath.Join(node.Path.Path...), desc)
}

// Display the latest data.
func (dsp *html) Display(data *treepb.NodeData) error {
	raw := string(data.GetRaw())
	switch data.Mime {
	case mime.HTMLText:
		dsp.el.SetInnerHTML(raw)
	case mime.CSSText:
		dsp.style.SetTextContent(raw)
	}
	return nil
}
