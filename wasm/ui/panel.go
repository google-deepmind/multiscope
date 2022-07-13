package ui

import (
	"fmt"
	treepb "multiscope/protos/tree_go_proto"

	"honnef.co/go/js/dom/v2"
)

// Panel is the window containing the display and some additional info.
// It also manages the interaction with the user.
type Panel struct {
	dbd  *Dashboard
	desc *Descriptor

	// root is the div containing all the children elements.
	root dom.Element
	// err is a paragraph to display an error.
	err dom.HTMLElement

	lastErr string
}

// NewPanel returns a new container panel to show a displayer on the UI.
func (dbd *Dashboard) NewPanel(title string, desc *Descriptor) (*Panel, error) {
	pnl := &Panel{
		dbd:  dbd,
		desc: desc,
		root: dbd.ui.Owner().CreateElement("div"),
	}
	dbd.root.AppendChild(pnl.root)
	pnl.appendTitle(title)
	pnl.appendErr()
	pnl.root.AppendChild(desc.root)
	pnl.refreshErrorPanel("")
	dbd.panels[desc.ID()] = pnl
	return pnl, dbd.ui.puller.registerPanel(desc)
}

func (pnl *Panel) appendTitle(title string) {
	p := pnl.dbd.ui.Owner().CreateElement("p")
	pnl.root.AppendChild(p)
	p.SetInnerHTML(fmt.Sprintf("<h2>%s</h2>", title))
}

func (pnl *Panel) appendErr() {
	pnl.err = pnl.dbd.NewErrorElement()
	pnl.root.AppendChild(pnl.err)
}

const errorMessage = `
<strong>error:</strong>
<pre>%s</pre>
`

func (pnl *Panel) refreshErrorPanel(err string) {
	pnl.lastErr = err
	if pnl.lastErr == "" {
		pnl.err.Style().SetProperty("display", "none", "")
		pnl.desc.root.Style().SetProperty("display", "block", "")
		return
	}
	pnl.err.Style().SetProperty("display", "block", "")
	pnl.desc.root.Style().SetProperty("display", "none", "")
	pnl.err.SetInnerHTML(fmt.Sprintf(errorMessage, pnl.lastErr))
}

func (pnl *Panel) updateError(err string) bool {
	if err == pnl.lastErr {
		return false
	}
	pnl.refreshErrorPanel(err)
	return true
}

// Display the latest data.
func (pnl *Panel) Display(data *treepb.NodeData) {
	if pnl.updateError(data.Error) {
		return
	}
	if err := pnl.desc.disp.Display(data); err != nil {
		pnl.updateError(err.Error())
	}
}
