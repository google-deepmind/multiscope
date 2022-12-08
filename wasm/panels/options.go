package panels

import (
	uipb "multiscope/protos/ui_go_proto"
)

// Option to apply to a panel.
// Not used at the moment. Kept here for future reference.
type Option func(*Panel) error

func runAll(opts []Option, pnl *Panel) error {
	for _, opt := range opts {
		if err := opt(pnl); err != nil {
			return err
		}
	}
	return nil
}

// ForwardResizeToRenderer forwards an internal UI ParentResize event to the renderer in the web worker when the panel is resized.
func ForwardResizeToRenderer(pnl *Panel) error {
	onResize := func(pnl *Panel) {
		width, height := pnl.size()
		pnl.ui.SendToRenderers(&uipb.UIEvent{
			Event: &uipb.UIEvent_Resize{
				Resize: &uipb.ParentResize{
					PanelID:     uint32(pnl.Desc().ID()),
					ChildWidth:  int32(width),
					ChildHeight: int32(height),
				},
			},
		})
	}
	pnl.OnResize(onResize)
	return nil
}
