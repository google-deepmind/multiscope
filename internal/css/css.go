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
