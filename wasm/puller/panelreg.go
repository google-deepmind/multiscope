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
