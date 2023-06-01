// Copyright 2023 DeepMind Technologies Limited
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

package widgets

import (
	"fmt"
	"multiscope/wasm/ui"
	"strconv"
	"syscall/js"
	"time"

	"honnef.co/go/js/dom/v2"
)

// Slider selects a value between 0 and 1.
type Slider struct {
	ui ui.UI
	el *dom.HTMLInputElement

	onChange   func(ui.UI, float32)
	lastChange time.Time
}

const sliderRangeMax = 10000

// NewSlider creates a new slider in a parent.
func NewSlider(gui ui.UI, parent dom.Element, onChange func(ui.UI, float32)) *Slider {
	s := &Slider{ui: gui, onChange: onChange}
	owner := gui.Owner()
	container := owner.CreateChild(parent, "div").(*dom.HTMLDivElement)
	container.Class().Add("slidecontainer")
	s.el = owner.CreateChild(container, "input").(*dom.HTMLInputElement)
	s.el.Class().Add("slider")
	s.el.SetAttribute("type", "range")
	s.el.SetAttribute("min", "0")
	max := strconv.FormatUint(sliderRangeMax, 10)
	s.el.SetAttribute("max", max)
	s.el.SetAttribute("value", max)
	s.el.Set("oninput", js.FuncOf(s.onEvent))
	return s
}

func (s *Slider) onEvent(js.Value, []js.Value) any {
	if s.onChange == nil {
		return nil
	}
	v, err := s.Value()
	if err != nil {
		s.ui.DisplayErr(err)
	}
	s.lastChange = time.Now()
	go s.onChange(s.ui, v)
	return nil
}

// Value returns the current value of a slider between 0 and 1.
func (s *Slider) Value() (float32, error) {
	valueS := s.el.Get("value").String()
	value, err := strconv.ParseInt(valueS, 10, 64)
	if err != nil {
		return -1, fmt.Errorf("cannot parse value from slider: %w", err)
	}
	return float32(value) / sliderRangeMax, nil
}

// Set the value of the slider.
// The function does nothing if the user sets the slider a short time ago.
func (s *Slider) Set(v float32) {
	s.el.Set("value", int(v*sliderRangeMax))
}
