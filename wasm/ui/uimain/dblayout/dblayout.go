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
