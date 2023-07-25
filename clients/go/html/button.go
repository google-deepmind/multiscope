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
	widgetpb "multiscope/protos/widget_go_proto"
)

const template = `<button onClick='multiscope.sendEvent("%s", %d)'>%s</button>`

// Button is a HTML button.
type Button struct {
	cb   func() error
	cnt  *Content
	text Stringer
	id   CallbackID
}

// NewButton returns a new button.
func NewButton(cnt *Content, text Stringer) *Button {
	b := &Button{
		cnt:  cnt,
		text: text,
	}
	b.id = cnt.Listen(b.processEvent)
	return b
}

func (b *Button) processEvent(ev *widgetpb.Event) error {
	if b.cb == nil {
		return nil
	}
	return b.cb()
}

// OnClick sets a listener for when the button is clicked.
func (b *Button) OnClick(cb func() error) *Button {
	b.cb = cb
	return b
}

// String returns the HTML representation of a button.
func (b *Button) String() string {
	return fmt.Sprintf(template, b.cnt.Writer().NodeID(), b.id, b.text.String())
}
