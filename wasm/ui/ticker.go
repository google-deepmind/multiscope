package ui

import (
	"fmt"
	tickerpb "multiscope/protos/ticker_go_proto"
	treepb "multiscope/protos/tree_go_proto"
	"path/filepath"

	"honnef.co/go/js/dom/v2"
)

func init() {
	RegisterDisplayPB(&tickerpb.TickerData{}, newTicker)
}

type ticker struct {
	pb tickerpb.TickerData
	el *dom.HTMLParagraphElement
}

func newTicker(dbd *Dashboard, node *treepb.Node) (*Panel, error) {
	dsp := &ticker{}
	dsp.el = dbd.Owner().CreateElement("p").(*dom.HTMLParagraphElement)
	dsp.el.Class().Add("panel-ticker")
	desc := NewDescriptor(dsp, dsp.el, nil, node.Path)
	return dbd.NewPanel(filepath.Join(node.Path.Path...), desc)
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
	dsp.el.SetInnerHTML(fmt.Sprintf(tickerHTML,
		dsp.pb.Tick,
		periods.Total.AsDuration(),
		periods.Experiment.AsDuration(),
		periods.Callbacks.AsDuration(),
		periods.Idle.AsDuration(),
	))
	return nil
}
