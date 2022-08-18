package panels

import (
	"fmt"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/ui"

	"honnef.co/go/js/dom/v2"
)

type (

	// Displayer display the content inside a panel.
	// A displayer may need to send data to a renderer in a work unit
	// to process the data to display.
	Displayer interface {
		// Display the latest data.
		Display(*treepb.NodeData) error

		// Root returns the root element of the display.
		Root() dom.HTMLElement
	}

	// Panel is the window containing the display and some additional info.
	// It also manages the interaction with the user.
	Panel struct {
		desc ui.Descriptor
		dsp  Displayer

		root *dom.HTMLDivElement
		// err is a paragraph to display an error.
		err     dom.HTMLElement
		lastErr string
	}
)

// NewPanel returns a new container panel to show a displayer on the UI.
func NewPanel(title string, desc ui.Descriptor, dsp Displayer) (ui.Panel, error) {
	dbd := desc.Dashboard()
	pnl := &Panel{
		desc: desc,
		dsp:  dsp,
	}
	pnl.root = dbd.UI().Owner().CreateElement("div").(*dom.HTMLDivElement)
	pnl.appendTitle(title)
	pnl.appendErr()
	pnl.root.AppendChild(dsp.Root())
	pnl.refreshErrorPanel("")
	return pnl, nil
}

func (pnl *Panel) appendTitle(title string) {
	owner := pnl.desc.Dashboard().UI().Owner()
	bar := owner.CreateElement("div")
	bar.Class().Add("panel-title")
	pnl.root.AppendChild(bar)

	bar.AppendChild(owner.CreateTextNode(title))

	rightSpan := owner.CreateElement("span").(*dom.HTMLSpanElement)
	rightSpan.Style().SetProperty("float", "right", "")
	closeButton := ui.NewButton(owner, "Ⓧ", pnl.processCloseEvent)
	rightSpan.AppendChild(closeButton)
	bar.AppendChild(rightSpan)
}

func (pnl *Panel) appendErr() {
	pnl.err = NewErrorElement(pnl.desc.Dashboard().UI().Owner())
	pnl.root.AppendChild(pnl.err)
}

const errorMessage = `
<strong>error:</strong>
<pre>%s</pre>
`

func (pnl *Panel) refreshErrorPanel(err string) {
	child := pnl.dsp.Root()
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

func (pnl *Panel) processCloseEvent(ev dom.Event) {
	if err := pnl.Desc().Dashboard().ClosePanel(pnl); err != nil {
		pnl.desc.Dashboard().UI().DisplayErr(err)
	}
}

// Root returns the root node of a panel.
// This node is added to the dashboard node when a panel is registered.
func (pnl *Panel) Root() *dom.HTMLDivElement {
	return pnl.root
}

// Desc returns the panel descriptor.
func (pnl *Panel) Desc() ui.Descriptor {
	return pnl.desc
}

// Display the latest data.
func (pnl *Panel) Display(data *treepb.NodeData) {
	if pnl.updateError(data.Error) {
		return
	}
	if err := pnl.dsp.Display(data); err != nil {
		pnl.updateError(err.Error())
	}
}

// NewErrorElement returns an element to display an error.
func NewErrorElement(owner dom.HTMLDocument) dom.HTMLElement {
	el := owner.CreateElement("p").(dom.HTMLElement)
	el.Class().Add("error-content")
	return el
}
