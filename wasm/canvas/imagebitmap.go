package canvas

import (
	"sync"
	"syscall/js"
)

// ImageBitmap maps an ImageBitmap JS object.
type ImageBitmap struct {
	Value js.Value
}

func ToImageBitMap(buf []byte) *ImageBitmap {
	// Pass the data to JS.
	typedArray := js.Global().Get("Uint8Array").New(len(buf))
	js.CopyBytesToJS(typedArray, buf)
	// Create the blob.
	array := js.Global().Get("Array").New(1)
	array.SetIndex(0, typedArray)
	blob := js.Global().Get("Blob").New(array)
	// Create the image.
	promise := js.Global().Call("createImageBitmap", blob)
	bmp := &ImageBitmap{}
	var wg sync.WaitGroup
	wg.Add(1)
	promise.Call("then", js.FuncOf(func(this js.Value, args []js.Value) any {
		bmp.Value = args[0]
		wg.Done()
		return nil
	}))
	wg.Wait()
	return bmp
}

// Width returns the width of the image.
func (bmp *ImageBitmap) Width() int {
	return bmp.Value.Get("width").Int()
}

// Height returns the height of the image.
func (bmp *ImageBitmap) Height() int {
	return bmp.Value.Get("height").Int()
}
