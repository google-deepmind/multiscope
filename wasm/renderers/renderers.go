// Package renderers provides data processor to compute the rendering in a webworker.
package renderers

import (
	"multiscope/internal/style"
	treepb "multiscope/protos/tree_go_proto"
	uipb "multiscope/protos/ui_go_proto"
	"reflect"
	"runtime"
	"syscall/js"

	"github.com/pkg/errors"
)

type (
	// Newer creates a new renderer.
	Newer func(*style.Style, *uipb.Panel, js.Value) Renderer

	// Renderer renders data and returns a (possibly nil) NodeData to transmit to the main thread.
	Renderer interface {
		Render(data *treepb.NodeData) (*treepb.NodeData, error)
	}

	noop struct{}
)

var nameToNewer = map[string]Newer{
	"": newNoop,
}

func newNoop(*style.Style, *uipb.Panel, js.Value) Renderer {
	return &noop{}
}

func (noop) Render(data *treepb.NodeData) (*treepb.NodeData, error) {
	return data, nil
}

// Register a function to create a new renderer given the name of the function.
func Register(f Newer) {
	nameToNewer[Name(f)] = f
}

// New creates a new renderer given a panel.
func New(stl *style.Style, panel *uipb.Panel, aux js.Value) (Renderer, error) {
	newer := nameToNewer[panel.Renderer]
	if newer == nil {
		return nil, errors.Errorf("renderer %q cannot be found. Available renderers are: %v", panel.Renderer, nameToNewer)
	}
	return newer(stl, panel, aux), nil
}

// Name returns the name of a renderer constructor.
func Name(newer Newer) string {
	if newer == nil {
		return ""
	}
	return runtime.FuncForPC(reflect.ValueOf(newer).Pointer()).Name()
}
