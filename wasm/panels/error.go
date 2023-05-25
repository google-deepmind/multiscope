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
	"fmt"
	"multiscope/internal/mime"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/ui"
	"path/filepath"

	"honnef.co/go/js/dom/v2"
)

func init() {
	newError := func(dbd ui.Dashboard, node *treepb.Node) (ui.Panel, error) {
		return newErrorDisplayer(dbd, node, func(node *treepb.Node) string {
			return fmt.Sprintf("server error: %s", node.Error)
		})
	}
	ui.RegisterBuilder(mime.Error, newError)

	newUnsupported := func(dbd ui.Dashboard, node *treepb.Node) (ui.Panel, error) {
		return newErrorDisplayer(dbd, node, func(node *treepb.Node) string {
			return fmt.Sprintf("no panel to display %q data", node.Mime)
		})
	}
	ui.RegisterBuilder(mime.Unsupported, newUnsupported)
}

type (
	errToText func(*treepb.Node) string

	errorDisplayer struct {
		node *treepb.Node
		root dom.HTMLElement
	}
)

func newErrorDisplayer(dbd ui.Dashboard, node *treepb.Node, f errToText) (ui.Panel, error) {
	dsp := &errorDisplayer{
		node: node,
		root: NewErrorElement(dbd.UI()),
	}
	dsp.root.AppendChild(dbd.UI().Owner().Doc().CreateTextNode(f(node)))
	desc := dbd.NewDescriptor(node, nil, node.Path)
	return NewPanel(filepath.Join(node.Path.Path...), desc, dsp)
}

// Display the latest data.
func (dsp *errorDisplayer) Display(data *treepb.NodeData) error {
	return nil
}

// Root returns the root element of the unsupported display.
func (dsp *errorDisplayer) Root() dom.HTMLElement {
	return dsp.root
}
