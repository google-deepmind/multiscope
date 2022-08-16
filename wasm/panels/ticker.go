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
	ui.RegisterDisplayPB(&tickerpb.TickerData{}, newTicker)
}

type ticker struct {
	pb   tickerpb.TickerData
	root *dom.HTMLParagraphElement
}

func newTicker(dbd *ui.Dashboard, node *treepb.Node) (ui.Panel, error) {
	dsp := &ticker{}
	dsp.root = dbd.Owner().CreateElement("p").(*dom.HTMLParagraphElement)
	dsp.root.Class().Add("panel-ticker")
	desc := dbd.NewDescriptor(dsp, nil, node.Path)
	return NewPanel(filepath.Join(node.Path.Path...), desc)
}

const tickerHTML = `
Tick: %d</br>
Periods:
<table>
  <tr>
    <th>total</th>
    <th>%s</th>
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
func (dsp *ticker) Display(data *treepb.NodeData) error {
	if err := data.GetPb().UnmarshalTo(&dsp.pb); err != nil {
		return err
	}
	periods := dsp.pb.Periods
	if periods == nil {
		return nil
	}
	dsp.root.SetInnerHTML(fmt.Sprintf(tickerHTML,
		dsp.pb.Tick,
		periods.Total.AsDuration(),
		periods.Experiment.AsDuration(),
		periods.Callbacks.AsDuration(),
		periods.Idle.AsDuration(),
	))
	return nil
}

// Root returns the root element of the ticker.
func (dsp *ticker) Root() dom.HTMLElement {
	return dsp.root
}
