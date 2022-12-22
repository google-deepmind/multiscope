package panels

import (
	"fmt"
	"math"
	tickerpb "multiscope/protos/ticker_go_proto"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/ui"
	"path/filepath"

	"honnef.co/go/js/dom/v2"
)

func init() {
	ui.RegisterBuilderPB(&tickerpb.PlayerInfo{}, newPlayer)
}

type player struct {
	pb          tickerpb.PlayerInfo
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

func (p *player) onSliderChange(gui ui.UI, slider *dom.HTMLInputElement) {
	tline := p.pb.Timeline
	if tline == nil {
		return
	}
	historyLength := float32(tline.HistoryLength)
	if historyLength == 0 {
		return
	}
	relative, err := ui.SliderValue(slider)
	if err != nil {
		gui.DisplayErr(err)
		return
	}
	tick := uint64(relative*historyLength) + tline.OldestTick
	if relative == 1 {
		tick = math.MaxUint64
	}
	ui.SendEvent(gui, p.node.Path, &tickerpb.PlayerAction{
		Action: &tickerpb.PlayerAction_TickView{TickView: &tickerpb.SetTickView{
			TickCommand: &tickerpb.SetTickView_ToDisplay{ToDisplay: tick}},
		},
	})
}

const playerHTML = `
Frames:
<table>
  <tr>
    <td>Current:</td>
    <td>%d</td>
  </tr>
  <tr>
    <td>Most recent:</td>
    <td>%d</td>
  </tr>
  <tr>
    <td>Oldest:</td>
    <td>%d</td>
  </tr>
  <tr>
    <td>History length:</td>
    <td>%d</td>
  </tr>
</table>
Storage capacity: %s
`

// Display the latest data.
func (p *player) Display(data *treepb.NodeData) error {
	if err := data.GetPb().UnmarshalTo(&p.pb); err != nil {
		return err
	}
	tline := p.pb.Timeline
	if tline == nil {
		p.display.SetInnerHTML("no info")
		return nil
	}
	p.display.SetInnerHTML(fmt.Sprintf(playerHTML,
		tline.DisplayTick,
		tline.OldestTick+tline.HistoryLength,
		tline.OldestTick,
		tline.HistoryLength,
		tline.StorageCapacity,
	))
	return nil
}

// Root returns the root element of the ticker.
func (p *player) Root() dom.HTMLElement {
	return p.root
}
