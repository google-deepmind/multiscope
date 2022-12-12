package renderers

import (
	"math"
	"multiscope/internal/style"
	treepb "multiscope/protos/tree_go_proto"
	uipb "multiscope/protos/ui_go_proto"
	"multiscope/wasm/canvas"
	"syscall/js"
)

type imageRenderer struct {
	offscreen     *canvas.OffscreenCanvas
	preferredSize *uipb.ElementSize
}

func init() {
	Register(NewImageRenderer)
}

// NewImageRenderer creates a new renderer to display images on a canvas.
func NewImageRenderer(stl *style.Style, regPanel *uipb.RegisterPanel, aux js.Value) Renderer {
	return &imageRenderer{
		offscreen:     &canvas.OffscreenCanvas{Value: aux.Get("offscreen")},
		preferredSize: regPanel.PreferredSize,
	}
}

func computeRatio(imageSize, canvasSize int) (float64, float64) {
	return float64(imageSize), float64(canvasSize) / float64(imageSize)
}

func (rdr *imageRenderer) Render(data *treepb.NodeData) (*treepb.NodeData, error) {
	if len(data.GetRaw()) == 0 {
		return nil, nil
	}
	imageBitmap := canvas.ToImageBitMap(data.GetRaw())

	if rdr.preferredSize != nil {
		rdr.resizeToPreferredSize(imageBitmap)
		rdr.preferredSize = nil
	}
	width, ratioWidth := computeRatio(imageBitmap.Width(), rdr.offscreen.Width())
	height, ratioHeight := computeRatio(imageBitmap.Height(), rdr.offscreen.Height())
	imageRatio := math.Min(ratioWidth, ratioHeight)

	finalWidth := int(width * imageRatio)
	finalHeight := int(height * imageRatio)

	ctx := rdr.offscreen.GetContext2d()
	ctx.Value.Set("imageSmoothingEnabled", false)
	ctx.Value.Call("drawImage", imageBitmap.Value, 0, 0, finalWidth, finalHeight)
	return nil, nil
}

func (rdr *imageRenderer) resizeToPreferredSize(img *canvas.ImageBitmap) {
	ratio := float32(img.Height()) / float32(img.Width())
	width, height := computePreferredSize(rdr.preferredSize, ratio)
	rdr.offscreen.SetSize(width, height)
}

func (rdr *imageRenderer) Resize(resize *uipb.ParentResize) {
	width := int(resize.ChildSize.Width)
	// Remove 5 pixels to avoid triggering a resize on the parent.
	height := int(resize.ChildSize.Height - 5)
	rdr.offscreen.SetSize(width, height)
}
