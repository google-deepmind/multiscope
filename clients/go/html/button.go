// Copyright 2023 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package html

import (
	"fmt"
	"html/template"

	eventspb "multiscope/protos/events_go_proto"
)

const buttonTemplate = `<button onClick='multiscope.sendEventWidgetClick("%s", %d)'>%s</button>`

// Button represents an HTML button widget with click event handling.
type Button struct {
	cb   func() error
	cnt  *Content
	text Stringer
	id   CallbackID
}

// NewButton creates and returns a new Button instance.
// It automatically registers the button for event handling.
func NewButton(cnt *Content, text Stringer) *Button {
	if cnt == nil {
		panic("Content cannot be nil")
	}
	if text == nil {
		panic("text cannot be nil")
	}

	b := &Button{
		cnt:  cnt,
		text: text,
	}
	b.id = cnt.Listen(b.processEvent)
	return b
}

// processEvent handles widget events for the button.
// It executes the registered callback function if one exists.
func (b *Button) processEvent(ev *eventspb.Widget) error {
	if b.cb == nil {
		return nil
	}
	return b.cb()
}

// OnClick registers a callback function to be executed when the button is clicked.
// Returns the button instance for method chaining.
func (b *Button) OnClick(cb func() error) *Button {
	b.cb = cb
	return b
}

// String returns the HTML representation of the button.
// The HTML includes JavaScript for event handling.
func (b *Button) String() string {
	// Escape the text content to prevent XSS attacks
	escapedText := template.HTMLEscapeString(b.text.String())
	return fmt.Sprintf(buttonTemplate, b.cnt.Writer().NodeID(), b.id, escapedText)
}
