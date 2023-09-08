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

package ui

import (
	"honnef.co/go/js/dom/v2"
)

type (
	// ChildOption is an option applied to a child before it is added to its parent.
	ChildOption interface {
		Apply(el dom.Node) dom.Node
	}

	// Owner owns all the element of a HTML document.
	Owner struct {
		ui   UI
		html dom.HTMLDocument
	}
)

// NewOwner returns the owner of the document.
func NewOwner(ui UI, html dom.HTMLDocument) *Owner {
	return &Owner{ui: ui, html: html}
}

func applyOptions(el dom.Node, options []ChildOption) dom.Node {
	for _, opt := range options {
		el = opt.Apply(el)
	}
	return el
}

// CreateChild creates a new element as a child of another element.
func (o *Owner) CreateChild(parent dom.Node, name string, options ...ChildOption) dom.Element {
	el := o.html.CreateElement(name)
	parent.AppendChild(applyOptions(el, options))
	return el
}

// Doc returns the HTML document owner.
func (o *Owner) Doc() dom.HTMLDocument {
	return o.html
}

// NewTextButton creates a new button with an event listener associated with it.
func (o *Owner) NewTextButton(parent dom.Element, text string, f func(ui UI, ev dom.Event), options ...ChildOption) *dom.HTMLAnchorElement {
	el := o.CreateChild(parent, "a", options...).(*dom.HTMLAnchorElement)
	el.Class().Add("button")
	el.SetTextContent(text)
	el.AddEventListener("click", true, func(ev dom.Event) {
		go f(o.ui, ev)
	})
	return el
}

// NewIconButton creates a new button with an event listener associated with it.
// See https://fonts.google.com/icons for the list of available icons.
func (o *Owner) NewIconButton(parent dom.Element, text string, f func(ui UI, ev dom.Event), options ...ChildOption) *dom.HTMLAnchorElement {
	span := o.CreateChild(parent, "span", options...).(*dom.HTMLSpanElement)
	span.Class().Add("material-icons")
	el := o.CreateChild(span, "a").(*dom.HTMLAnchorElement)
	el.Class().Add("icon")
	el.Class().Add("button")
	el.SetTextContent(text)
	el.AddEventListener("click", true, func(ev dom.Event) {
		go f(o.ui, ev)
	})
	return el
}

// CreateTextNode returns a new text node.
func (o *Owner) CreateTextNode(parent dom.Element, s string, options ...ChildOption) *dom.Text {
	el := o.html.CreateTextNode(s)
	parent.AppendChild(applyOptions(el, options))
	return el
}
