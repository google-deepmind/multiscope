// Package ui implements the Multiscope UI.
package ui

import "honnef.co/go/js/dom/v2"

// NewButton creates a new button with an event listener associated with it.
func NewButton(owner dom.HTMLDocument, text string, f func(ev dom.Event)) *dom.HTMLAnchorElement {
	el := owner.CreateElement("a").(*dom.HTMLAnchorElement)
	el.Class().Add("icon")
	el.Class().Add("button")
	el.AppendChild(owner.CreateTextNode(text))
	el.AddEventListener("click", true, func(ev dom.Event) {
		go f(ev)
	})
	return el
}
