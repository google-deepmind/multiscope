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

package panels

import (
	"fmt"
	"multiscope/internal/mime"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/ui"
	"path/filepath"

	"honnef.co/go/js/dom/v2"
)

func init() {
	ui.RegisterBuilder(mime.PlainText, newText)
}

type text struct {
	root *dom.HTMLParagraphElement
}

func newText(dbd ui.Dashboard, node *treepb.Node) (ui.Panel, error) {
	dsp := &text{
		root: dbd.UI().Owner().Doc().CreateElement("p").(*dom.HTMLParagraphElement),
	}
	dsp.root.Class().Add("text-content")
	desc := dbd.NewDescriptor(node, nil, node.Path)
	return NewPanel(filepath.Join(node.Path.Path...), desc, dsp)
}

// Display the latest data.
func (dsp *text) Display(data *treepb.NodeData) error {
	dsp.root.SetInnerHTML(fmt.Sprintf("<pre>%s</pre>", string(data.GetRaw())))
	return nil
}

func (dsp *text) Root() dom.HTMLElement {
	return dsp.root
}
