// Package dblayout stores a layout.
package dblayout

import (
	"multiscope/wasm/ui"

	rootpb "multiscope/protos/root_go_proto"

	"github.com/pkg/errors"
	"honnef.co/go/js/dom/v2"
)

const (
	defaultPanelWidth  = 640
	defaultPanelHeight = 480
)

// SettingKey is the key used in the settings to store the layout.
const SettingKey = "layout"

// Layout within the dashboard organizing the display of the panels.
type Layout interface {
	Root() dom.Node

	Load(lyt *rootpb.Layout) error

	Append(ui.Panel)

	Remove(ui.Panel)
}

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
