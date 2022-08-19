package renderers

import (
	"multiscope/internal/style"
	treepb "multiscope/protos/tree_go_proto"
	uipb "multiscope/protos/ui_go_proto"
	"multiscope/wasm/canvas"
	"syscall/js"
)

type imageRenderer struct {
	offscreen *canvas.OffscreenCanvas
}

func init() {
	Register(NewImageRenderer)
}

// NewImageRenderer creates a new renderer to display images on a canvas.
func NewImageRenderer(style *style.Style, panel *uipb.Panel, aux js.Value) Renderer {
	return &imageRenderer{
		offscreen: &canvas.OffscreenCanvas{Value: aux.Get("offscreen")},
	}
}

func (rdr *imageRenderer) Render(data *treepb.NodeData) (*treepb.NodeData, error) {
	imageBitmap := canvas.ToImageBitMap(data.GetRaw())
	rdr.offscreen.SetWidth(imageBitmap.Width())
	rdr.offscreen.SetHeight(imageBitmap.Height())
	ctx := rdr.offscreen.GetContext2d()
	ctx.Value.Set("imageSmoothingEnabled", false)
	ctx.Value.Call("drawImage", imageBitmap.Value, 0, 0)
	return nil, nil
}
