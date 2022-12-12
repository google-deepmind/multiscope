package ui

import "honnef.co/go/js/dom/v2"

// Owner owns all the element of a HTML document.
type Owner struct {
	html dom.HTMLDocument
}

// NewOwner returns the owner of the document.
func NewOwner(html dom.HTMLDocument) *Owner {
	return &Owner{html: html}
}

// CreateChild creates a new element as a child of another element.
func (o *Owner) CreateChild(parent dom.Node, name string) dom.Element {
	el := o.html.CreateElement(name)
	parent.AppendChild(el)
	return el
}

// Doc returns the HTML document owner.
func (o *Owner) Doc() dom.HTMLDocument {
	return o.html
}

// NewTextButton creates a new button with an event listener associated with it.
func (o *Owner) NewTextButton(parent dom.Element, text string, f func(ev dom.Event)) *dom.HTMLAnchorElement {
	el := o.CreateChild(parent, "a").(*dom.HTMLAnchorElement)
	el.Class().Add("icon")
	el.Class().Add("button")
	el.SetTextContent(text)
	el.AddEventListener("click", true, func(ev dom.Event) {
		go f(ev)
	})
	return el
}

// NewIconButton creates a new button with an event listener associated with it.
// See https://fonts.google.com/icons for the list of available icons.
func (o *Owner) NewIconButton(parent dom.Element, text string, f func(ev dom.Event)) *dom.HTMLAnchorElement {
	span := o.CreateChild(parent, "span").(*dom.HTMLSpanElement)
	span.Class().Add("material-icons")
	el := o.CreateChild(span, "a").(*dom.HTMLAnchorElement)
	el.Class().Add("icon")
	el.Class().Add("button")
	el.SetTextContent(text)
	el.AddEventListener("click", true, func(ev dom.Event) {
		go f(ev)
	})
	return el
}

// CreateTextNode returns a new text node.
func (o *Owner) CreateTextNode(parent dom.Element, s string) *dom.Text {
	el := o.html.CreateTextNode(s)
	parent.AppendChild(el)
	return el
}
