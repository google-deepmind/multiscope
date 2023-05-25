// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
					PanelID: uint32(pnl.Desc().ID()),
					ChildSize: &uipb.ElementSize{
						Width:  int32(width),
						Height: int32(height),
					},
				},
			},
		})
	}
	pnl.OnResize(onResize)
	return nil
}
