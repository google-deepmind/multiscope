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

// Package css provides utilities to interact with CSS.
package css

import (
	"fmt"
	"image/color"
)

// Color returns a string representing a HTML color.
func Color(c color.Color) string {
	r, g, b, a := c.RGBA()
	r = (r * 0xff) / 0xffff
	g = (g * 0xff) / 0xffff
	b = (b * 0xff) / 0xffff
	a = (a * 0xff) / 0xffff
	return fmt.Sprintf("rgba(%d,%d,%d,%d)", r, g, b, a)
}
