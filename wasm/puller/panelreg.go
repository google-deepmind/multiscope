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

package puller

import (
	uipb "multiscope/protos/ui_go_proto"
	"multiscope/wasm/renderers"
	"multiscope/wasm/ui"
)

type (
	panelS struct {
		pb  *uipb.Panel
		rdr renderers.Renderer
	}

	panelRegistry struct {
		reg map[ui.PanelID]*panelS
	}
)

func newPanelRegistry() *panelRegistry {
	return &panelRegistry{
		reg: make(map[ui.PanelID]*panelS),
	}
}

func (reg *panelRegistry) register(pbPanel *uipb.Panel, rdr renderers.Renderer) *panelS {
	panel := &panelS{pbPanel, rdr}
	reg.reg[ui.PanelID(panel.pb.Id)] = panel
	return panel
}

func (reg *panelRegistry) renderer(id ui.PanelID) renderers.Renderer {
	panel := reg.reg[id]
	if panel == nil {
		return nil
	}
	return panel.rdr
}

func (reg *panelRegistry) unregister(id ui.PanelID) {
	delete(reg.reg, id)
}
