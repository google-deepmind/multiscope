package panels

import (
	"context"
	"fmt"
	tickerpb "multiscope/protos/ticker_go_proto"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/ui"
	"path/filepath"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
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
	dsp := &ticker{node: node}
	owner := dbd.UI().Owner()
	dsp.root = owner.Doc().CreateElement("p").(*dom.HTMLParagraphElement)
	dsp.root.Class().Add("ticker-content")
	dsp.display = owner.CreateChild(dsp.root, "p").(*dom.HTMLParagraphElement)
	control := owner.CreateChild(dsp.root, "p").(*dom.HTMLParagraphElement)
	dsp.createControls(dbd, control)
	desc := dbd.NewDescriptor(node, nil, node.Path)
	return NewPanel(filepath.Join(node.Path.Path...), desc, dsp)
}

func (t *ticker) createControls(dbd ui.Dashboard, control *dom.HTMLParagraphElement) {
	owner := dbd.UI().Owner()
	owner.NewIconButton(control, "play_arrow", func(ev dom.Event) {
		t.sendAction(dbd, tickerpb.Command_RUN)
	})
	owner.NewIconButton(control, "pause", func(ev dom.Event) {
		t.sendAction(dbd, tickerpb.Command_PAUSE)
	})
	owner.NewIconButton(control, "skip_next", func(ev dom.Event) {
		t.sendAction(dbd, tickerpb.Command_STEP)
	})
}

func (t *ticker) sendAction(dbd ui.Dashboard, cmd tickerpb.Command) {
	clt := dbd.UI().TreeClient()
	ctx := context.Background()
	var event anypb.Any
	if err := anypb.MarshalFrom(&event, &tickerpb.TickerAction{
		Action: &tickerpb.TickerAction_Command{Command: cmd},
	}, proto.MarshalOptions{}); err != nil {
		dbd.UI().DisplayErr(err)
		return
	}
	_, err := clt.SendEvents(ctx, &treepb.SendEventsRequest{
		Events: []*treepb.Event{{
			Path:    t.node.Path,
			Payload: &event,
		}},
	})
	if err != nil {
		dbd.UI().DisplayErr(err)
	}
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
