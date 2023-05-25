// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package panels implements panels to display different kind of data.
package panels

import (
	"fmt"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/ui"
	"syscall/js"

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
		ui   ui.UI
		desc ui.Descriptor
		dsp  Displayer

		root    *dom.HTMLDivElement
		content *dom.HTMLDivElement
		// err is a paragraph to display an error.
		err     dom.HTMLElement
		lastErr string
	}
)

// NewPanel returns a new container panel to show a displayer on the UI.
func NewPanel(title string, desc ui.Descriptor, dsp Displayer, opts ...Option) (*Panel, error) {
	dbd := desc.Dashboard()
	pnl := &Panel{
		ui:   dbd.UI(),
		desc: desc,
		dsp:  dsp,
	}
	pnl.root = dbd.UI().Owner().Doc().CreateElement("div").(*dom.HTMLDivElement)
	pnl.root.Class().Add("panel")

	pnl.appendTitle(title)
	pnl.appendErr()
	defer pnl.refreshErrorPanel("")

	pnl.content = dbd.UI().Owner().CreateChild(pnl.root, "div").(*dom.HTMLDivElement)
	pnl.content.Class().Add("panel-content")
	pnl.content.AppendChild(dsp.Root())

	return pnl, runAll(opts, pnl)
}

func (pnl *Panel) appendTitle(title string) {
	mui := pnl.desc.Dashboard().UI()
	bar := mui.Owner().CreateChild(pnl.root, "div")
	bar.Class().Add("panel-title")

	mui.Owner().NewIconButton(bar, "close", pnl.processCloseEvent)
	textTitleSpan := mui.Owner().CreateChild(bar, "span").(*dom.HTMLSpanElement)
	textTitleSpan.Class().Add("panel-text-title")
	mui.Owner().CreateTextNode(textTitleSpan, title)
}

func (pnl *Panel) appendErr() {
	pnl.err = NewErrorElement(pnl.desc.Dashboard().UI())
	pnl.root.AppendChild(pnl.err)
}

const errorMessage = `
<strong>error:</strong>
<pre>%s</pre>
`

func (pnl *Panel) refreshErrorPanel(err string) {
	child := pnl.content
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

func (pnl *Panel) displayError(err string) {
	if err == pnl.lastErr {
		return
	}
	pnl.refreshErrorPanel(err)
}

func (pnl *Panel) processCloseEvent(gui ui.UI, ev dom.Event) {
	if err := pnl.Desc().Dashboard().ClosePanel(pnl); err != nil {
		gui.DisplayErr(err)
	}
}

// OnResize calls a function everytime an element is resized.
func (pnl *Panel) OnResize(f func(*Panel)) {
	listener := func(this js.Value, args []js.Value) any {
		f(pnl)
		return nil
	}
	observer := js.Global().Get("ResizeObserver").New(js.FuncOf(listener))
	observer.Call("observe", pnl.root.Underlying())
}

func (pnl *Panel) size() (int, int) {
	width := pnl.content.Get("offsetWidth").Int()
	height := pnl.content.Get("offsetHeight").Int()
	return width, height
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
	if data.Error != "" {
		pnl.displayError(fmt.Sprintf("failed to fetch data from the puller: %s", data.Error))
		return
	}
	if err := pnl.dsp.Display(data); err != nil {
		pnl.displayError(fmt.Sprintf("displayer %T returned an error: %s", pnl.dsp, err.Error()))
		return
	}
}

// NewErrorElement returns an element to display an error.
func NewErrorElement(mui ui.UI) dom.HTMLElement {
	el := mui.Owner().Doc().CreateElement("p").(dom.HTMLElement)
	el.Class().Add("error-content")
	return el
}
