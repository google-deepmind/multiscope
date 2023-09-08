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
	"multiscope/clients/go/remote"
	eventspb "multiscope/protos/events_go_proto"
	treepb "multiscope/protos/tree_go_proto"
	"strings"

	"github.com/pkg/errors"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
)

type (
	// CallbackID is an ID for a registered callback that can be inserted in HTML code.
	CallbackID int64

	// Callback to process events.
	Callback func(*eventspb.Widget) error

	// Stringer is implemented by any value that has a String method,
	// which returns HTML code.
	Stringer interface {
		String() string
	}

	// Content represents HTML content.
	Content struct {
		w            *remote.HTMLWriter
		content      []Stringer
		lastID       CallbackID
		idToCallback map[CallbackID]Callback
	}

	// HTML is raw HTML content.
	HTML string
)

var _ remote.Callbacker = (*Content)(nil)

// BR is a new HTML line.
const BR = HTML("</br>")

// String returns the underlying string without modifications.
func (c HTML) String() string {
	return string(c)
}

// NewContent returns a content structure for a given client.
func NewContent(w *remote.HTMLWriter) *Content {
	return &Content{w: w, idToCallback: make(map[CallbackID]Callback)}
}

// Append HTML content.
func (c *Content) Append(s ...Stringer) {
	c.content = append(c.content, s...)
}

func (c *Content) processEvent(ev *treepb.Event) (err error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("cannot process widget event: %w", err)
		}
	}()
	var widgetEvent eventspb.Widget
	if err := anypb.UnmarshalTo(ev.Payload, &widgetEvent, proto.UnmarshalOptions{}); err != nil {
		return errors.Errorf("cannot unmarshal event: %v", err)
	}
	cb, ok := c.idToCallback[CallbackID(widgetEvent.WidgetId)]
	if !ok {
		return errors.Errorf("unknown event ID: %d", widgetEvent.WidgetId)
	}
	return cb(&widgetEvent)
}

// Callback returns the callback to process events for this content.
func (c *Content) Callback() remote.Callback {
	return c.processEvent
}

// Listen registers a callback and returns its ID to put in the HTML.
func (c *Content) Listen(cb Callback) CallbackID {
	c.lastID++
	c.idToCallback[c.lastID] = cb
	return c.lastID
}

func (c *Content) String() string {
	all := make([]string, len(c.content))
	for i, str := range c.content {
		all[i] = str.String()
	}
	return strings.Join(all, "\n")
}

// Writer returns the HTML writer owning the content.
func (c *Content) Writer() *remote.HTMLWriter {
	return c.w
}
