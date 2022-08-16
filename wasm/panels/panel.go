package panels

import (
	"fmt"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/ui"

	"honnef.co/go/js/dom/v2"
)

// Panel is the window containing the display and some additional info.
// It also manages the interaction with the user.
type Panel struct {
	root dom.Element
	desc *ui.Descriptor

	// err is a paragraph to display an error.
	err dom.HTMLElement

	lastErr string
}

// NewPanel returns a new container panel to show a displayer on the UI.
func NewPanel(title string, desc *ui.Descriptor) (ui.Panel, error) {
	dbd := desc.Dashboard()
	pnl := &Panel{
		desc: desc,
		root: dbd.Owner().CreateElement("div"),
	}
	pnl.appendTitle(title)
	pnl.appendErr()
	pnl.root.AppendChild(desc.Displayer().Root())
	pnl.refreshErrorPanel("")
	return pnl, dbd.RegisterPanel(pnl)
}

func (pnl *Panel) appendTitle(title string) {
	p := pnl.desc.Dashboard().Owner().CreateElement("p")
	pnl.root.AppendChild(p)
	p.SetInnerHTML(fmt.Sprintf("<h2>%s</h2>", title))
}

func (pnl *Panel) appendErr() {
	pnl.err = pnl.desc.Dashboard().NewErrorElement()
	pnl.root.AppendChild(pnl.err)
}

const errorMessage = `
<strong>error:</strong>
<pre>%s</pre>
`

func (pnl *Panel) refreshErrorPanel(err string) {
	child := pnl.desc.Displayer().Root()
	pnl.lastErr = err
	if pnl.lastErr == "" {
		pnl.err.Style().SetProperty("display", "none", "")
		child.Style().SetProperty("display", "block", "")
		return
	}
	pnl.err.Style().SetProperty("display", "block", "")
	child.Style().SetProperty("display", "none", "")
	pnl.err.SetInnerHTML(fmt.Sprintf(errorMessage, pnl.lastErr))
}

func (pnl *Panel) updateError(err string) bool {
	if err == pnl.lastErr {
		return false
	}
	pnl.refreshErrorPanel(err)
	return true
}

// Root returns the root node of a panel.
// This node is added to the dashboard node when a panel is registered.
func (pnl *Panel) Root() dom.Node {
	return pnl.root
}

// Desc returns the panel descriptor.
func (pnl *Panel) Desc() *ui.Descriptor {
	return pnl.desc
}

// Display the latest data.
func (pnl *Panel) Display(data *treepb.NodeData) {
	if pnl.updateError(data.Error) {
		return
	}
	if err := pnl.desc.Displayer().Display(data); err != nil {
		pnl.updateError(err.Error())
	}
}
