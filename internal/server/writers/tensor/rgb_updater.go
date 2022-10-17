package tensor

import (
	"image"
	"image/color"
	"multiscope/internal/server/core"
	"multiscope/internal/server/writers/tensor/tnrdr"
)

type rgbUpdater struct {
	parent *Writer
	img    *image.RGBA
	writer *imageWriter
	key    core.Key
}

func newRGBUpdater(parent *Writer) *rgbUpdater {
	up := &rgbUpdater{
		parent: parent,
		img:    image.NewRGBA(image.Rect(0, 0, 0, 0)),
		writer: newImageWriter(),
	}
	parent.AddChild(NodeNameRGBImage, up.writer)
	return up
}

func (u *rgbUpdater) forwardActive(parent *core.Path) {
	path := parent.PathTo(NodeNameRGBImage)
	u.parent.state.PathLog().Forward(parent, path)
	u.key = path.ToKey()
}

func alloc(img *image.RGBA, shape []int) *image.RGBA {
	shapeX, shapeY := 0, 0
	if IsRGBShape(shape) {
		shapeX = shape[1]
		shapeY = shape[0]
	}
	size := img.Bounds().Size()
	if shapeX == size.X && shapeY == size.Y {
		return img
	}
	return image.NewRGBA(image.Rect(0, 0, shapeX, shapeY))
}

func toUInt8(m *tnrdr.Metrics, tns Tensor) []uint8 {
	var valsUint8 []uint8
	if tnsuint8, ok := tns.(WithUInt8); ok {
		valsUint8 = tnsuint8.ValuesUInt8()
	}
	if valsUint8 != nil {
		return valsUint8
	}
	vals := tns.Values()
	valsUint8 = make([]uint8, len(vals))
	scale := float32(1.0)
	// TODO(degris): this is bad because uint8 images with a maximum of 1 will be scaled.
	// This will be fixed once we change the format of tensors used in Multiscope.
	if m.Min >= 0 && m.Max <= 1 && m.Max > 0 {
		scale = 255
	}
	for i, f := range vals {
		valsUint8[i] = uint8(f * scale)
	}
	return valsUint8
}

func (u *rgbUpdater) reset() error {
	u.img = image.NewRGBA(image.Rect(0, 0, 0, 0))
	return u.writer.Write(u.img)
}

func (u *rgbUpdater) update(Tensor) error {
	if !u.parent.state.PathLog().IsActive(u.key) {
		return nil
	}
	u.img = alloc(u.img, u.parent.tensor.Shape())
	const (
		rOffset = iota
		gOffset
		bOffset
		stride
	)
	size := u.img.Bounds().Size()
	data := toUInt8(&u.parent.m, u.parent.tensor)
	for x := 0; x < size.X; x++ {
		for y := 0; y < size.Y; y++ {
			pos := stride * (x + y*size.X)
			u.img.SetRGBA(x, y, color.RGBA{
				R: data[pos+rOffset],
				G: data[pos+gOffset],
				B: data[pos+bOffset],
				A: 255,
			})
		}
	}
	return u.writer.Write(u.img)
}

// IsRGBShape returns true if the shape of the tensor matches the shape of a RGB image.
func IsRGBShape(shape []int) bool {
	return len(shape) == 3 && shape[2] == 3
}
