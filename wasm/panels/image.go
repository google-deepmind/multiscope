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
	"multiscope/internal/mime"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/renderers"
	"multiscope/wasm/ui"
	"path/filepath"

	"honnef.co/go/js/dom/v2"
)

func init() {
	ui.RegisterBuilder(mime.PNG, newImagePanel)
}

type imagePanel struct {
	canvas *dom.HTMLCanvasElement
}

func newImagePanel(dbd ui.Dashboard, node *treepb.Node) (ui.Panel, error) {
	dsp := &imagePanel{}
	dsp.canvas = dbd.UI().Owner().Doc().CreateElement("canvas").(*dom.HTMLCanvasElement)
	desc := dbd.NewDescriptor(node, renderers.NewImageRenderer, node.Path)
	desc.AddTransferable("offscreen", dsp.canvas.Call("transferControlToOffscreen"))
	return NewPanel(filepath.Join(node.Path.Path...), desc, dsp)
}

// Display the latest data.
func (dsp *imagePanel) Display(data *treepb.NodeData) error {
	return nil
}

// Root element of the display.
func (dsp *imagePanel) Root() dom.HTMLElement {
	return dsp.canvas
}
