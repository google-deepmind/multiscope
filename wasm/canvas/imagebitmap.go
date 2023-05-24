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

package canvas

import (
	"sync"
	"syscall/js"
)

// ImageBitmap maps an ImageBitmap JS object.
type ImageBitmap struct {
	Value js.Value
}

// ToImageBitMap takes a data buffer and builds a JS ImageBitmap.
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
