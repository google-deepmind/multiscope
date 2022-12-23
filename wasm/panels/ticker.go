package panels

import (
	"fmt"
	tickerpb "multiscope/protos/ticker_go_proto"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/ui"
	"path/filepath"
	"time"

	"honnef.co/go/js/dom/v2"
)

func init() {
	ui.RegisterBuilderPB(&tickerpb.TickerData{}, newTicker)
}

type ticker struct {
	pb      tickerpb.TickerData
	node    *treepb.Node
	root    *dom.HTMLParagraphElement
	display *dom.HTMLParagraphElement
}

func newTicker(dbd ui.Dashboard, node *treepb.Node) (ui.Panel, error) {
	t := &ticker{node: node}
	owner := dbd.UI().Owner()
	t.root = owner.Doc().CreateElement("p").(*dom.HTMLParagraphElement)
	t.root.Class().Add("ticker-content")
	t.display = owner.CreateChild(t.root, "p").(*dom.HTMLParagraphElement)
	t.createControls(owner, t.root)
	t.createPeriod(owner, t.root)
	desc := dbd.NewDescriptor(node, nil, node.Path)
	return NewPanel(filepath.Join(node.Path.Path...), desc, t)
}

func (t *ticker) createControls(owner *ui.Owner, parent *dom.HTMLParagraphElement) {
	control := owner.CreateChild(parent, "p").(*dom.HTMLParagraphElement)
	owner.NewIconButton(control, "play_arrow", func(gui ui.UI, ev dom.Event) {
		t.sendAction(gui, tickerpb.Command_RUN)
	})
	owner.NewIconButton(control, "pause", func(gui ui.UI, ev dom.Event) {
		t.sendAction(gui, tickerpb.Command_PAUSE)
	})
	owner.NewIconButton(control, "skip_next", func(gui ui.UI, ev dom.Event) {
		t.sendAction(gui, tickerpb.Command_STEP)
	})
}

func (t *ticker) sendAction(gui ui.UI, cmd tickerpb.Command) {
	ui.SendEvent(gui, t.node.Path, &tickerpb.TickerAction{
		Action: &tickerpb.TickerAction_Command{Command: cmd},
	})
}

func (t *ticker) createPeriod(owner *ui.Owner, parent *dom.HTMLParagraphElement) {
	period := owner.CreateChild(parent, "p").(*dom.HTMLParagraphElement)
	owner.NewTextButton(period, "0ms", func(gui ui.UI, ev dom.Event) {
		t.sendPeriod(gui, 0)
	})
	owner.NewTextButton(period, "10ms", func(gui ui.UI, ev dom.Event) {
		t.sendPeriod(gui, 10*time.Millisecond)
	})
	owner.NewTextButton(period, "100ms", func(gui ui.UI, ev dom.Event) {
		t.sendPeriod(gui, 100*time.Millisecond)
	})
	owner.NewTextButton(period, "1s", func(gui ui.UI, ev dom.Event) {
		t.sendPeriod(gui, time.Second)
	})
}

func (t *ticker) sendPeriod(gui ui.UI, d time.Duration) {
	ui.SendEvent(gui, t.node.Path, &tickerpb.TickerAction{
		Action: &tickerpb.TickerAction_SetPeriod{SetPeriod: &tickerpb.SetPeriod{
			PeriodMs: int64(d / time.Millisecond),
		}},
	})
}

const tickerHTML = `
Tick: %d</br>
Periods:
<table>
  <tr>
    <td>total</td>
    <td>%s</td>
  </tr>
  <tr>
    <td>experiment</td>
    <td>%s</td>
  </tr>
  <tr>
    <td>callbacks</td>
    <td>%s</td>
  </tr>
  <tr>
    <td>idle</td>
    <td>%s</td>
  </tr>
</table>
`

// Display the latest data.
func (t *ticker) Display(data *treepb.NodeData) error {
	if err := data.GetPb().UnmarshalTo(&t.pb); err != nil {
		return err
	}
	periods := t.pb.Periods
	if periods == nil {
		return nil
	}
	t.display.SetInnerHTML(fmt.Sprintf(tickerHTML,
		t.pb.Tick,
		periods.Total.AsDuration(),
		periods.Experiment.AsDuration(),
		periods.Callbacks.AsDuration(),
		periods.Idle.AsDuration(),
	))
	return nil
}

// Root returns the root element of the ticker.
func (t *ticker) Root() dom.HTMLElement {
	return t.root
}
