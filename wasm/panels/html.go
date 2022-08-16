package panels

import (
	"multiscope/internal/mime"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/ui"
	"path/filepath"

	"honnef.co/go/js/dom/v2"
)

func init() {
	ui.RegisterDisplay(mime.HTMLParent, newHTML)
}

type html struct {
	root *dom.HTMLDivElement

	el    *dom.HTMLParagraphElement
	style *dom.HTMLStyleElement
}

func newHTML(dbd *ui.Dashboard, node *treepb.Node) (ui.Panel, error) {
	dsp := &html{
		root: dbd.Owner().CreateElement("div").(*dom.HTMLDivElement),
	}
	dsp.root.Class().Add("panel-html")
	dsp.style = dbd.Owner().CreateElement("style").(*dom.HTMLStyleElement)
	dsp.style.SetAttribute("scoped", "")
	dsp.root.AppendChild(dsp.style)
	dsp.el = dbd.Owner().CreateElement("p").(*dom.HTMLParagraphElement)
	dsp.root.AppendChild(dsp.el)
	htmlPath := &treepb.NodePath{
		Path: append(append([]string{}, node.Path.Path...), mime.NodeNameHTML),
	}
	cssPath := &treepb.NodePath{
		Path: append(append([]string{}, node.Path.Path...), mime.NodeNameCSS),
	}
	desc := dbd.NewDescriptor(nil, node.Path, htmlPath, cssPath)
	return NewPanel(filepath.Join(node.Path.Path...), desc, dsp)
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

// Root returns the root element of the html display.
func (dsp *html) Root() dom.HTMLElement {
	return dsp.root
}
