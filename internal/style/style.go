// Package style manages the style of the ui.
package style

import (
	"image/color"
	"multiscope/internal/settings"

	"github.com/tdegris/base16-go/themes"
	"gonum.org/v1/plot/font"
	"gonum.org/v1/plot/vg"
)

var googleLight = themes.Theme{
	Name:    "Google Light",
	Author:  "Seth Wright (http://sethawright.com)",
	Color00: color.RGBA{R: 255, G: 255, B: 255, A: 255},
	Color01: color.RGBA{R: 224, G: 224, B: 224, A: 255},
	Color02: color.RGBA{R: 197, G: 200, B: 198, A: 255},
	Color03: color.RGBA{R: 180, G: 183, B: 180, A: 255},
	Color04: color.RGBA{R: 150, G: 152, B: 150, A: 255},
	Color05: color.RGBA{R: 55, G: 59, B: 65, A: 255},
	Color06: color.RGBA{R: 40, G: 42, B: 46, A: 255},
	Color07: color.RGBA{R: 29, G: 31, B: 33, A: 255},
	Color08: color.RGBA{R: 204, G: 52, B: 43, A: 255},
	Color09: color.RGBA{R: 249, G: 106, B: 56, A: 255},
	Color0A: color.RGBA{R: 251, G: 169, B: 34, A: 255},
	Color0B: color.RGBA{R: 25, G: 136, B: 68, A: 255},
	Color0C: color.RGBA{R: 57, G: 113, B: 237, A: 255},
	Color0D: color.RGBA{R: 57, G: 113, B: 237, A: 255},
	Color0E: color.RGBA{R: 163, G: 106, B: 199, A: 255},
	Color0F: color.RGBA{R: 57, G: 113, B: 237, A: 255},
}

type (
	// Style stores all attributes related to style.
	Style struct {
		themeName string
		theme     themes.Theme

		fontFamily string
		fontSize   font.Length
		settings   settings.Settings

		toApply []func(*Style)
	}
)

// NewStyle returns a new style with some default.
func NewStyle(sets settings.Settings) *Style {
	s := &Style{
		fontFamily: "helvetica",
		fontSize:   vg.Points(12),
		theme:      googleLight,
		settings:   sets,
	}
	s.settings.Listen("style", &s.themeName, s.updateCB)
	return s
}

// SetTheme sets the current theme.
func (s *Style) SetTheme(theme string) {
	s.Set(theme, "", -1)
}

// Set the current style.
func (s *Style) Set(theme, fontFamily string, fontSize font.Length) {
	if theme != "" {
		s.settings.Set(s, "style", theme)
	}
	if fontFamily != "" {
		s.fontFamily = fontFamily
	}
	if fontSize >= 0 {
		s.fontSize = fontSize
	}
}

func (s *Style) updateCB(any) error {
	if t, ok := themes.Base16[s.themeName]; ok {
		s.theme = t
	}
	for _, f := range s.toApply {
		f(s)
	}
	return nil
}

// Theme returns the current theme.
func (s *Style) Theme() themes.Theme {
	return s.theme
}

// FontFamily returns the current font family.
func (s *Style) FontFamily() string {
	return s.fontFamily
}

// FontSize returns the current font size.
func (s *Style) FontSize() font.Length {
	return s.fontSize
}

// OnChange registers a function to call when the theme changes.
func (s *Style) OnChange(f func(*Style)) {
	f(s)
	s.toApply = append(s.toApply, f)
}

// Background returns the current background color.
func (s *Style) Background() color.RGBA {
	return s.theme.Color00
}

// BackgroundSub returns a lighter background.
func (s *Style) BackgroundSub() color.RGBA {
	return s.theme.Color01
}

// Foreground returns the current foreground color.
func (s *Style) Foreground() color.RGBA {
	return s.theme.Color05
}
