// Package puller implements a puller querying the server to get the latest data.
package puller

import (
	"context"
	"fmt"
	"multiscope/internal/fmtx"
	"multiscope/internal/httpgrpc"
	"multiscope/internal/settings"
	"multiscope/internal/style"
	treepb "multiscope/protos/tree_go_proto"
	treepbgrpc "multiscope/protos/tree_go_proto"
	uipb "multiscope/protos/ui_go_proto"
	"multiscope/wasm/renderers"
	"multiscope/wasm/ui"
	"multiscope/wasm/worker"
	"syscall/js"

	"github.com/pkg/errors"
	"go.uber.org/multierr"
	"gonum.org/v1/plot/font"
	"google.golang.org/protobuf/encoding/prototext"
)

type (
	queryS struct {
		pb  *uipb.ToPuller
		aux js.Value
	}

	// Puller pulls data from the Multiscope server.
	Puller struct {
		queries chan queryS

		messager   *worker.Worker
		settings   settings.Settings
		style      *style.Style
		treeClient treepb.TreeClient
		req        *request
	}
)

// New returns a new puller to fetch the data from the server.
func New(messager *worker.Worker) *Puller {
	p := &Puller{
		queries:  make(chan queryS, 1),
		messager: messager,
		req:      newRequest(),
	}
	if err := p.init(); err != nil {
		p.sendError(err)
		return nil
	}
	return p
}

func (p *Puller) init() error {
	p.settings = newSettings(p.sendError)
	p.style = style.NewStyle(p.settings)
	connect := uipb.Connect{}
	if _, err := p.messager.Recv(&connect); err != nil {
		return errors.Errorf("cannot unmarshal connection info from the main worker: %v", err)
	}
	if connect.Scheme == "" {
		return errors.Errorf("invalid scheme: %q", connect.Scheme)
	}
	if connect.Host == "" {
		return errors.Errorf("invalid host: %q", connect.Host)
	}
	p.treeClient = treepbgrpc.NewTreeClient(httpgrpc.Connect(connect.Scheme, connect.Host))
	go p.receiveQuery()
	return nil
}

func (p *Puller) sendError(err error) {
	err = fmtx.FormatError(err)
	if sendErr := p.messager.Send(&uipb.DisplayData{Err: err.Error()}, nil); sendErr != nil {
		fmt.Printf("WORKER ERROR: cannot send error to main: %v\nError being sent: %v\n", sendErr, err)
	}
}

func (p *Puller) receiveQuery() {
	for {
		toPuller := uipb.ToPuller{}
		aux, err := p.messager.Recv(&toPuller)
		if err != nil {
			p.sendError(err)
			continue
		}
		p.queries <- queryS{&toPuller, aux}
	}
}

func (p *Puller) processRegisterPanel(pbPanel *uipb.Panel, aux js.Value) error {
	rdr, err := renderers.New(p.style, pbPanel, aux)
	if err != nil {
		return err
	}
	panel := &panelS{pbPanel, rdr}
	for _, path := range panel.pb.Paths {
		p.req.registerPath(panel, path)
	}
	return nil
}

func (p *Puller) processUnregisterPanel(pbPanel *uipb.Panel, aux js.Value) error {
	id := ui.PanelID(pbPanel.Id)
	var gErr error
	for _, path := range pbPanel.Paths {
		if err := p.req.unregisterPath(id, path); err != nil {
			gErr = multierr.Append(gErr, err)
		}
	}
	return gErr
}

func renderData(rdr renderers.Renderer, data *treepb.NodeData) (rendered *treepb.NodeData, err error) {
	defer func() {
		if r := recover(); r != nil {
			rendered = nil
			err = errors.Errorf("renderer %T generated the following rendering error:\n%q\n for the following treepb.NodeData:\n%s", rdr, r, prototext.Format(data))
		}
	}()
	return rdr.Render(data)
}

func (p *Puller) processPullQuery(pull *uipb.Pull) error {
	ctx := context.Background()
	resp, err := p.treeClient.GetNodeData(ctx, p.req.pb())
	if err != nil {
		return err
	}
	displayData := &uipb.DisplayData{
		Data: make(map[uint32]*uipb.PanelData),
	}
	for _, nodeData := range resp.NodeData {
		if nodeData == nil || nodeData.Data == nil {
			continue
		}
		p.req.setLastTick(nodeData.Path, nodeData.Tick)
		for _, panel := range p.req.panels(nodeData.Path) {
			renderedData, err := renderData(panel.rdr, nodeData)
			if renderedData == nil {
				renderedData = &treepb.NodeData{}
			}
			if err != nil {
				renderedData.Error = err.Error()
			}
			panelData := displayData.Data[panel.pb.Id]
			if panelData == nil {
				panelData = &uipb.PanelData{}
				displayData.Data[panel.pb.Id] = panelData
			}
			panelData.Nodes = append(panelData.Nodes, nodeData)
		}
	}
	return p.messager.Send(displayData, nil)
}

func (p *Puller) processTheme(stl *uipb.StyleChange) error {
	p.style.Set(stl.Theme, stl.FontFamily, font.Length(stl.FontSize))
	return nil
}

// Pull data from the server forever.
func (p *Puller) Pull() {
	for q := range p.queries {
		var err error
		switch query := q.pb.Query.(type) {
		case *uipb.ToPuller_RegisterPanel:
			err = p.processRegisterPanel(query.RegisterPanel, q.aux)
		case *uipb.ToPuller_UnregisterPanel:
			err = p.processUnregisterPanel(query.UnregisterPanel, q.aux)
		case *uipb.ToPuller_Pull:
			err = p.processPullQuery(query.Pull)
		case *uipb.ToPuller_Style:
			err = p.processTheme(query.Style)
		default:
			err = errors.Errorf("query of type %T not supported in puller worker", q.pb.Query)
		}
		if err != nil {
			p.sendError(err)
		}
	}
}
