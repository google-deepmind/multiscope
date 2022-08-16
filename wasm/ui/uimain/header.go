package uimain

import (
	"syscall/js"

	"github.com/pkg/errors"
	"honnef.co/go/js/dom/v2"
)

const headerHTML = `
<h1 class="toppage">
  <a class="icon" onclick="toggleTreeSideBar()">☰</a>Multiscope
</h1>
`

// Header is the page header.
type Header struct {
	ui  *UI
	hdr dom.HTMLElement
}

func newHeader(ui *UI) (*Header, error) {
	const headerTag = "header"
	elements := ui.Owner().GetElementsByTagName(headerTag)
	if len(elements) != 1 {
		return nil, errors.Errorf("wrong number of elements of tag %q: got %d but want 1", headerTag, len(elements))
	}
	h := &Header{
		ui:  ui,
		hdr: elements[0].(dom.HTMLElement),
	}
	h.hdr.SetInnerHTML(headerHTML)
	js.Global().Set("toggleTreeSideBar", js.FuncOf(h.toggleTreeSideBar))
	return h, nil
}

func (h *Header) toggleTreeSideBar(this js.Value, args []js.Value) interface{} {
	left := h.ui.Layout().LeftBar()
	if left.isVisible() {
		left.hide()
	} else {
		left.show()
	}
	return nil
}
