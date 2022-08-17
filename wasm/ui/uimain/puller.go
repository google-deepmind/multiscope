package uimain

import (
	"fmt"
	"multiscope/internal/style"
	uipb "multiscope/protos/ui_go_proto"
	"multiscope/wasm/worker"
)

type puller struct {
	ui   *UI
	wkr  *worker.Worker
	toUI chan *uipb.DisplayData
}

func newPuller(ui *UI, wkr *worker.Worker) *puller {
	p := &puller{
		ui:   ui,
		wkr:  wkr,
		toUI: make(chan *uipb.DisplayData),
	}
	if err := p.wkr.Send(ui.addr, nil); err != nil {
		ui.DisplayErr(err)
	}
	ui.style.OnChange(p.onPaletteChange)
	go p.run()
	return p
}

func errorF(s string, args ...any) *uipb.DisplayData {
	return &uipb.DisplayData{Err: fmt.Sprintf(s, args...)}
}

func (p *puller) onPaletteChange(s *style.Style) {
	if err := p.wkr.Send(&uipb.ToPuller{Query: &uipb.ToPuller_Style{
		Style: &uipb.StyleChange{
			Theme:      s.Theme().Name,
			FontSize:   float64(s.FontSize()),
			FontFamily: s.FontFamily(),
		},
	},
	}, nil); err != nil {
		p.ui.DisplayErr(err)
	}
}

func (p *puller) unregisterPanel(desc *Descriptor) error {
	info, _ := desc.PanelPB()
	toPuller := &uipb.ToPuller{
		Query: &uipb.ToPuller_UnregisterPanel{
			UnregisterPanel: info,
		},
	}
	if err := p.wkr.Send(toPuller, nil); err != nil {
		return fmt.Errorf("cannot unregister panel to pull worker: %w", err)
	}
	return nil
}

func (p *puller) registerPanel(desc *Descriptor) error {
	info, aux := desc.PanelPB()
	toPuller := &uipb.ToPuller{
		Query: &uipb.ToPuller_RegisterPanel{
			RegisterPanel: info,
		},
	}
	transferables := []any{}
	for _, trf := range aux {
		transferables = append(transferables, trf)
	}
	if err := p.wkr.Send(toPuller, aux, transferables...); err != nil {
		return fmt.Errorf("cannot register panel to pull worker: %w", err)
	}
	return nil
}

func (p *puller) run() {
	toPuller := &uipb.ToPuller{
		Query: &uipb.ToPuller_Pull{},
	}
	for {
		if err := p.wkr.Send(toPuller, nil); err != nil {
			p.toUI <- errorF("cannot send pull request to puller worker: %v", err)
			continue
		}
		displayData := uipb.DisplayData{}
		if _, err := p.wkr.Recv(&displayData); err != nil {
			p.toUI <- errorF("cannot receive display data from puller worker: %v", err)
			continue
		}
		p.toUI <- &displayData
	}
}

func (p *puller) lastDisplayData() *uipb.DisplayData {
	select {
	case dd := <-p.toUI:
		return dd
	default:
		return nil
	}
}
