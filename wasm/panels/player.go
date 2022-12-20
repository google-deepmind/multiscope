package panels

import (
	"fmt"
	tickerpb "multiscope/protos/ticker_go_proto"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/ui"
	"path/filepath"

	"honnef.co/go/js/dom/v2"
)

func init() {
	ui.RegisterBuilderPB(&tickerpb.PlayerData{}, newPlayer)
}

type player struct {
	pb      tickerpb.PlayerData
	node    *treepb.Node
	root    *dom.HTMLParagraphElement
	display *dom.HTMLParagraphElement
}

func newPlayer(dbd ui.Dashboard, node *treepb.Node) (ui.Panel, error) {
	dsp := &player{node: node}
	owner := dbd.UI().Owner()
	dsp.root = owner.Doc().CreateElement("p").(*dom.HTMLParagraphElement)
	dsp.root.Class().Add("ticker-content")
	dsp.display = owner.CreateChild(dsp.root, "p").(*dom.HTMLParagraphElement)
	desc := dbd.NewDescriptor(node, nil, node.Path)
	return NewPanel(filepath.Join(node.Path.Path...), desc, dsp)
}

// Display the latest data.
func (p *player) Display(data *treepb.NodeData) error {
	if err := data.GetPb().UnmarshalTo(&p.pb); err != nil {
		return err
	}
	tick := p.pb.Tick
	p.display.SetInnerHTML(fmt.Sprintf("Tick: %d", tick))
	return nil
}

// Root returns the root element of the ticker.
func (p *player) Root() dom.HTMLElement {
	return p.root
}
