package uimain

import (
	"multiscope/wasm/ui"

	"github.com/pkg/errors"
	"honnef.co/go/js/dom/v2"
)

// Header is the page header.
type Header struct {
	ui  *UI
	hdr dom.HTMLElement
}

func newHeader(mui *UI) (*Header, error) {
	const headerTag = "header"
	elements := mui.Owner().GetElementsByTagName(headerTag)
	if len(elements) != 1 {
		return nil, errors.Errorf("wrong number of elements of tag %q: got %d but want 1", headerTag, len(elements))
	}
	h := &Header{
		ui:  mui,
		hdr: elements[0].(dom.HTMLElement),
	}
	h1 := mui.Owner().CreateElement("h1")
	h1.Class().Add("toppage")
	h1.AppendChild(ui.NewButton(mui.Owner(), "☰", h.toggleTreeSideBar))
	h1.AppendChild(mui.Owner().CreateTextNode("Multiscope"))
	h.hdr.AppendChild(h1)
	return h, nil
}

func (h *Header) toggleTreeSideBar(dom.Event) {
	left := h.ui.Layout().LeftBar()
	if left.isVisible() {
		left.hide()
	} else {
		left.show()
	}
}
