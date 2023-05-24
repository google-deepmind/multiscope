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

package panels

import (
	"multiscope/internal/mime"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/ui"
	"path/filepath"

	"honnef.co/go/js/dom/v2"
)

func init() {
	ui.RegisterBuilder(mime.HTMLParent, newHTML)
}

type html struct {
	root *dom.HTMLDivElement

	el    *dom.HTMLParagraphElement
	style *dom.HTMLStyleElement
}

func newHTML(dbd ui.Dashboard, node *treepb.Node) (ui.Panel, error) {
	owner := dbd.UI().Owner()
	dsp := &html{
		root: owner.Doc().CreateElement("div").(*dom.HTMLDivElement),
	}
	dsp.root.Class().Add("html-content")
	dsp.style = owner.CreateChild(dsp.root, "style").(*dom.HTMLStyleElement)
	dsp.style.SetAttribute("scoped", "")
	dsp.el = owner.CreateChild(dsp.root, "p").(*dom.HTMLParagraphElement)
	htmlPath := &treepb.NodePath{
		Path: append(append([]string{}, node.Path.Path...), mime.NodeNameHTML),
	}
	cssPath := &treepb.NodePath{
		Path: append(append([]string{}, node.Path.Path...), mime.NodeNameCSS),
	}
	desc := dbd.NewDescriptor(node, nil, node.Path, htmlPath, cssPath)
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
