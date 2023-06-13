// Copyright 2023 DeepMind Technologies Limited
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

// Package jsapi defines the public API available from Javascript.
package jsapi

import (
	pathlib "multiscope/lib/path"
	widgetpb "multiscope/protos/widget_go_proto"
	"multiscope/wasm/ui/uimain"
	"syscall/js"

	"github.com/pkg/errors"
)

type api struct {
	ui *uimain.UI
}

func (api *api) sendEvent(this js.Value, args []js.Value) any {
	if len(args) < 2 {
		api.ui.DisplayErr(errors.Errorf("sendEvents called with less than 2 arguments"))
		return nil
	}
	if args[0].Type() != js.TypeString {
		api.ui.DisplayErr(errors.Errorf("args[0] has type %q but want type 'string'", args[0].Type()))
		return nil
	}
	if args[1].Type() != js.TypeNumber {
		api.ui.DisplayErr(errors.Errorf("args[1] has type %q but want type 'number'", args[1].Type()))
		return nil
	}
	nodePath, err := pathlib.FromBase64(args[0].String())
	if err != nil {
		api.ui.DisplayErr(errors.Errorf("cannot decode string %q to nodepath: %v", args[0].Type(), err))
		return nil
	}
	widgetID := args[1].Int()
	api.ui.SendToServer(nodePath, &widgetpb.Event{
		WidgetId: int64(widgetID),
	})
	return nil
}

// BuildAPI builds the public API on the client.
func BuildAPI(ui *uimain.UI) {
	glob := js.Global()
	jsAPI := glob.Get("Object").New()
	glob.Set("multiscope", jsAPI)
	api := api{ui: ui}
	jsAPI.Set("sendEvent", js.FuncOf(api.sendEvent))
}
