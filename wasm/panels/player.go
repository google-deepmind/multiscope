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
	pb          tickerpb.PlayerData
	node        *treepb.Node
	root        *dom.HTMLParagraphElement
	display     *dom.HTMLParagraphElement
	timeControl *dom.HTMLDivElement
}

func newPlayer(dbd ui.Dashboard, node *treepb.Node) (ui.Panel, error) {
	p := &player{node: node}
	owner := dbd.UI().Owner()
	p.root = owner.Doc().CreateElement("p").(*dom.HTMLParagraphElement)
	p.root.Class().Add("ticker-content")
	p.display = owner.CreateChild(p.root, "p").(*dom.HTMLParagraphElement)
	desc := dbd.NewDescriptor(node, nil, node.Path)
	p.timeControl = p.newTimeControl(owner, p.root)
	return NewPanel(filepath.Join(node.Path.Path...), desc, p)
}

func (p *player) newTimeControl(owner *ui.Owner, parent *dom.HTMLParagraphElement) *dom.HTMLDivElement {
	timeControl := owner.CreateChild(parent, "div").(*dom.HTMLDivElement)
	owner.NewSlider(timeControl, p.onSliderChange)
	return timeControl
}

func (p *player) onSliderChange(slider *dom.HTMLInputElement) {
	fmt.Println("slider", slider.Get("value").String())
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
