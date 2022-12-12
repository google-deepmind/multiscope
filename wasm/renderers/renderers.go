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
	Newer func(*style.Style, *uipb.RegisterPanel, js.Value) Renderer

	// Renderer renders data and returns a (possibly nil) NodeData to transmit to the main thread.
	Renderer interface {
		Render(data *treepb.NodeData) (*treepb.NodeData, error)
	}

	// Resizer resizes the content on which the rendering is done.
	Resizer interface {
		Resize(resize *uipb.ParentResize)
	}

	noop struct{}
)

var nameToNewer = map[string]Newer{
	"": newNoop,
}

func newNoop(*style.Style, *uipb.RegisterPanel, js.Value) Renderer {
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
func New(stl *style.Style, regPanel *uipb.RegisterPanel, aux js.Value) (Renderer, error) {
	panel := regPanel.Panel
	newer := nameToNewer[panel.Renderer]
	if newer == nil {
		return nil, errors.Errorf("renderer %q cannot be found. Available renderers are: %v", panel.Renderer, nameToNewer)
	}
	return newer(stl, regPanel, aux), nil
}

// Name returns the name of a renderer constructor.
func Name(newer Newer) string {
	if newer == nil {
		return ""
	}
	return runtime.FuncForPC(reflect.ValueOf(newer).Pointer()).Name()
}

// Resize processes a UI main resize event by a renderer.
// Returns an error if the renderer cannot process resize events.
func Resize(rdr Renderer, resize *uipb.ParentResize) error {
	if rdr == nil {
		return nil
	}
	resizer, ok := rdr.(Resizer)
	if !ok {
		return errors.Errorf("renderer %T has received a resize event but does not implement Resizer", rdr)
	}
	resizer.Resize(resize)
	return nil
}

const (
	defaultWidth  = 640
	defaultHeight = 480
)

func computePreferredSize(size *uipb.ElementSize, ratio float32) (width, height int) {
	if size.Height == 0 && size.Width == 0 {
		size.Width = defaultWidth
		size.Height = defaultHeight
	}
	width = int(size.Width)
	if width == 0 {
		width = int(float32(size.Height) / ratio)
	}
	height = int(size.Height)
	if height == 0 {
		height = int(float32(size.Width) * ratio)
	}
	return width, height
}
