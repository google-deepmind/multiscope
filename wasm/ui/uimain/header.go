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
	owner := mui.Owner()
	elements := owner.Doc().GetElementsByTagName(headerTag)
	if len(elements) != 1 {
		return nil, errors.Errorf("wrong number of elements of tag %q: got %d but want 1", headerTag, len(elements))
	}
	h := &Header{
		ui:  mui,
		hdr: elements[0].(dom.HTMLElement),
	}
	h1 := owner.CreateChild(h.hdr, "h1")
	h1.Class().Add("toppage")
	// Left side.
	left := owner.CreateChild(h1, "p").(*dom.HTMLParagraphElement)
	owner.NewIconButton(left, "menu", h.toggleTreeSideBar)
	mainTitle := owner.CreateChild(left, "span").(*dom.HTMLSpanElement)
	mainTitle.Class().Add("toppage-main-title")
	mui.Owner().CreateTextNode(mainTitle, "Multiscope")

	// Right side.
	right := owner.CreateChild(left, "span").(*dom.HTMLSpanElement)
	right.Style().SetProperty("float", "right", "")
	owner.NewIconButton(right, "refresh", h.reloadLayout)
	owner.NewIconButton(right, "close", h.emptyLayout)
	return h, nil
}

func (h *Header) reloadLayout(ui.UI, dom.Event) {
	if err := h.ui.layout.Dashboard().refresh(); err != nil {
		h.ui.DisplayErr(err)
	}
}

func (h *Header) emptyLayout(ui.UI, dom.Event) {
	h.ui.layout.Dashboard().closeAll()
}

func (h *Header) toggleTreeSideBar(ui.UI, dom.Event) {
	left := h.ui.Layout().LeftBar()
	if left.isVisible() {
		left.hide()
	} else {
		left.show()
	}
}
