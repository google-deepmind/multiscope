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

// Package dblayout stores a layout.
package dblayout

import (
	"multiscope/wasm/ui"

	rootpb "multiscope/protos/root_go_proto"
	uipb "multiscope/protos/ui_go_proto"

	"github.com/pkg/errors"
	"honnef.co/go/js/dom/v2"
)

// SettingKey is the key used in the settings to store the layout.
const SettingKey = "layout"

type (
	// Layout within the dashboard organizing the display of the panels.
	Layout interface {
		PreferredSize() *uipb.ElementSize

		Root() dom.Node

		Load(lyt *rootpb.Layout) error

		Append(ui.Panel)

		Remove(ui.Panel)
	}

	// Size represents the size of a panel.
	Size struct {
		Width, Height int
	}
)

// New returns a new layout given a protocol buffer description.
func New(dbd ui.Dashboard, lyt *rootpb.Layout) (Layout, error) {
	if lyt == nil {
		return newList(dbd), nil
	}
	if list := lyt.GetList(); list != nil {
		return newList(dbd), nil
	}
	return newList(dbd), errors.Errorf("unknown layout")
}
