package dblayout

import (
	"multiscope/wasm/ui"

	rootpb "multiscope/protos/root_go_proto"

	"github.com/pkg/errors"
	"honnef.co/go/js/dom/v2"
)

const settingKey = "layout"

// Layout within the dashboard organizing the display of the panels.
type Layout interface {
	Root() dom.Node

	Load(lyt *rootpb.Layout) error

	Append(ui.Panel)

	Remove(ui.Panel)
}

// NewLayout returns a new layout given a protocol buffer description.
func New(dbd ui.Dashboard, lyt *rootpb.Layout) (Layout, error) {
	if lyt == nil {
		return newList(dbd), nil
	}
	if list := lyt.GetList(); list != nil {
		return newList(dbd), nil
	}
	return newList(dbd), errors.Errorf("unknown layout")
}

// LoadPB loads a layout configuration from the settings.
func LoadPB(dbd ui.Dashboard) *rootpb.Layout {
	l := &rootpb.Layout{}
	if !dbd.UI().Settings().Get(settingKey, l) {
		return nil
	}
	return l
}
