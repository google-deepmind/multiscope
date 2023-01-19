package panels

import (
	"fmt"
	"math"
	"multiscope/internal/utils/temporizer"
	tickerpb "multiscope/protos/ticker_go_proto"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/ui"
	"multiscope/wasm/widgets"
	"path/filepath"
	"time"

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

	// userInputTemporizer avoids changing the slider position when the server sends
	// a new position while the user interacts with the slider.
	//
	// There are a few cases where the slider locations can change because of the server:
	// 1. two users look at the same process using Multiscope, one user moves the slider.
	//    The position of the slider will be updated on the frontend of the other user.
	// 2. the process continues to run, storage is running out of memory, so the position
	//    of the slider needs to be updated
	// 3. the user moves the slider, the command is sent to the server, the server takes a
	//    while to process the command, and by the time the response comes back to the UI,
	//    the user has already moved the slider to a different position. This is even
	//    worse when a lot of commands are sent to the server (see next temporizer).
	userInputTemporizer *temporizer.Temporizer[float32]
	// sendTickValueTemporizer avoids sending too many commands to the server
	// when the user moves the slider.
	sendTickValueTemporizer *temporizer.Temporizer[uint64]
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
	p.sendTickValueTemporizer = temporizer.NewTemporizer(func(tick uint64) {
		p.sendTickViewAction(gui, &tickerpb.SetTickView{
			TickCommand: &tickerpb.SetTickView_ToDisplay{ToDisplay: tick},
		})
	})
	p.slider = widgets.NewSlider(gui, timeControl, p.onSliderChange)
	p.userInputTemporizer = temporizer.NewTemporizer(func(val float32) {
		p.slider.Set(val)
	})
	p.createPeriod(owner, timeControl)
	p.createControls(owner, timeControl)
}

func (p *player) createControls(owner *ui.Owner, parent dom.Element) {
	control := owner.CreateChild(parent, "div").(dom.HTMLElement)
	control.Class().Add("fit-content")
	owner.NewIconButton(control, "skip_previous", func(gui ui.UI, ev dom.Event) {
		p.sendControlAction(gui, tickerpb.Command_CMD_STEPBACK)
	})
	owner.NewIconButton(control, "play_arrow", func(gui ui.UI, ev dom.Event) {
		p.sendControlAction(gui, tickerpb.Command_CMD_RUN)
	})
	owner.NewIconButton(control, "pause", func(gui ui.UI, ev dom.Event) {
		p.sendControlAction(gui, tickerpb.Command_CMD_PAUSE)
	})
	owner.NewIconButton(control, "skip_next", func(gui ui.UI, ev dom.Event) {
		p.sendControlAction(gui, tickerpb.Command_CMD_STEP)
	})
}

func (p *player) createPeriod(owner *ui.Owner, parent dom.HTMLElement) {
	period := owner.CreateChild(parent, "div").(dom.HTMLElement)
	period.Class().Add("fit-content")
	newPeriodButton(owner, period, "0ms", 0, p.sendPeriod)
	newPeriodButton(owner, period, "10ms", 10*time.Millisecond, p.sendPeriod)
	newPeriodButton(owner, period, "100ms", 100*time.Millisecond, p.sendPeriod)
	newPeriodButton(owner, period, "1s", time.Second, p.sendPeriod)
}

func (p *player) sendPeriod(gui ui.UI, d time.Duration) {
	p.userInputTemporizer.SetDuration(0)
	gui.SendToServer(p.node.Path, &tickerpb.PlayerAction{
		Action: &tickerpb.PlayerAction_SetPeriod{SetPeriod: &tickerpb.SetPeriod{
			PeriodMs: int64(d / time.Millisecond),
		}},
	})
	p.sendControlAction(gui, tickerpb.Command_CMD_RUN)
}

func (p *player) sendControlAction(gui ui.UI, cmd tickerpb.Command) {
	p.userInputTemporizer.SetDuration(0)
	gui.SendToServer(p.node.Path, &tickerpb.PlayerAction{
		Action: &tickerpb.PlayerAction_Command{Command: cmd},
	})
}

func (p *player) sendTickViewAction(gui ui.UI, setTick *tickerpb.SetTickView) {
	gui.SendToServer(p.node.Path, &tickerpb.PlayerAction{
		Action: &tickerpb.PlayerAction_TickView{TickView: setTick},
	})
}

func (p *player) onSliderChange(gui ui.UI, val float32) {
	p.userInputTemporizer.SetDuration(5 * time.Second)
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
	p.sendTickValueTemporizer.Set(tick, 150*time.Millisecond)
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
	p.userInputTemporizer.Set(
		float32(tline.DisplayTick-tline.OldestTick)/float32(tline.HistoryLength-1),
		50*time.Millisecond,
	)
	return nil
}

// Root returns the root element of the ticker.
func (p *player) Root() dom.HTMLElement {
	return p.root
}
