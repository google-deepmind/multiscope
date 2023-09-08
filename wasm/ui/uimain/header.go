// Copyright 2023 DeepMind Technologies Limited
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

package uimain

import (
	rootpb "multiscope/protos/root_go_proto"
	"multiscope/wasm/ui"

	"github.com/pkg/errors"
	"honnef.co/go/js/dom/v2"
)

const (
	enableCapture  = "Capture"
	disableCapture = "Stop capture"
)

// Header is the page header.
type Header struct {
	ui  *UI
	hdr dom.HTMLElement

	captureElement *dom.HTMLAnchorElement
}

func newHeader(mui *UI, rootInfo *rootpb.RootInfo) (*Header, error) {
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

	// Capture which appears dynamically depending on the client.
	h.captureElement = owner.NewTextButton(right, enableCapture, h.capture)
	h.captureElement.Class().Add("bordered")
	h.captureElement.Style().SetProperty("font-size", "60%", "")
	h.displayCapture(rootInfo.EnableCapture)

	// Refresh and close all panels buttons.
	owner.NewIconButton(right, "refresh", h.reloadLayout)
	owner.NewIconButton(right, "close", h.emptyLayout)
	return h, nil
}

func (h *Header) displayCapture(enable bool) {
	if enable {
		h.captureElement.Style().SetProperty("display", "", "")
	} else {
		h.captureElement.Style().SetProperty("display", "none", "")
	}
}

func (h *Header) capture(ui.UI, dom.Event) {
	enabled := h.ui.capturer.Toggle()
	if enabled {
		// Capture is enabled. So, we set the label to disable capture.
		h.captureElement.SetInnerHTML(disableCapture)
	} else {
		// Capture is disabled. So, we set the label to enable capture.
		h.captureElement.SetInnerHTML(enableCapture)
	}
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
