package panels

import (
	"fmt"
	"math"
	tickerpb "multiscope/protos/ticker_go_proto"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/ui"
	"multiscope/wasm/widgets"
	"path/filepath"

	"honnef.co/go/js/dom/v2"
)

func init() {
	ui.RegisterBuilderPB(&tickerpb.PlayerInfo{}, newPlayer)
}

type player struct {
	pb      tickerpb.PlayerInfo
	node    *treepb.Node
	root    *dom.HTMLParagraphElement
	display *dom.HTMLParagraphElement
	slider  *widgets.Slider
}

func newPlayer(dbd ui.Dashboard, node *treepb.Node) (ui.Panel, error) {
	p := &player{node: node}
	owner := dbd.UI().Owner()
	p.root = owner.Doc().CreateElement("p").(*dom.HTMLParagraphElement)
	p.root.Class().Add("ticker-content")
	p.display = owner.CreateChild(p.root, "p").(*dom.HTMLParagraphElement)
	desc := dbd.NewDescriptor(node, nil, node.Path)
	p.newTimeControl(dbd.UI(), p.root)
	return NewPanel(filepath.Join(node.Path.Path...), desc, p)
}

func (p *player) newTimeControl(gui ui.UI, parent *dom.HTMLParagraphElement) {
	owner := gui.Owner()
	timeControl := owner.CreateChild(parent, "div").(*dom.HTMLDivElement)
	p.slider = widgets.NewSlider(gui, timeControl, p.onSliderChange)
	p.createControls(owner, timeControl)
}

func (p *player) createControls(owner *ui.Owner, control dom.Element) {
	owner.NewIconButton(control, "play_arrow", func(gui ui.UI, ev dom.Event) {
		p.sendControlAction(gui, tickerpb.Command_RUN)
	})
	owner.NewIconButton(control, "pause", func(gui ui.UI, ev dom.Event) {
		p.sendControlAction(gui, tickerpb.Command_PAUSE)
	})
	owner.NewIconButton(control, "skip_next", func(gui ui.UI, ev dom.Event) {
		p.sendControlAction(gui, tickerpb.Command_STEP)
	})
}

func (p *player) sendControlAction(gui ui.UI, cmd tickerpb.Command) {
	ui.SendEvent(gui, p.node.Path, &tickerpb.PlayerAction{
		Action: &tickerpb.PlayerAction_Command{Command: cmd},
	})
}

func (p *player) sendTickViewAction(gui ui.UI, setTick *tickerpb.SetTickView) {
	ui.SendEvent(gui, p.node.Path, &tickerpb.PlayerAction{
		Action: &tickerpb.PlayerAction_TickView{TickView: setTick},
	})
}

func (p *player) onSliderChange(gui ui.UI, val float32) {
	tline := p.pb.Timeline
	if tline == nil {
		return
	}
	historyLength := float32(tline.HistoryLength)
	if historyLength == 0 {
		return
	}
	tick := uint64(val*historyLength) + tline.OldestTick
	if val == 1 {
		tick = math.MaxUint64
	}
	p.sendTickViewAction(gui, &tickerpb.SetTickView{
		TickCommand: &tickerpb.SetTickView_ToDisplay{ToDisplay: tick},
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
		tline.OldestTick+tline.HistoryLength-1,
		tline.OldestTick,
		tline.HistoryLength,
		tline.StorageCapacity,
	))
	p.slider.Set(float32(tline.DisplayTick-tline.OldestTick) / float32(tline.HistoryLength-1))
	return nil
}

// Root returns the root element of the ticker.
func (p *player) Root() dom.HTMLElement {
	return p.root
}
