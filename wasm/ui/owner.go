package ui

import (
	"strconv"
	"syscall/js"

	"honnef.co/go/js/dom/v2"
)

// Owner owns all the element of a HTML document.
type Owner struct {
	ui   UI
	html dom.HTMLDocument
}

// NewOwner returns the owner of the document.
func NewOwner(ui UI, html dom.HTMLDocument) *Owner {
	return &Owner{ui: ui, html: html}
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
func (o *Owner) NewTextButton(parent dom.Element, text string, f func(ui UI, ev dom.Event)) *dom.HTMLAnchorElement {
	el := o.CreateChild(parent, "a").(*dom.HTMLAnchorElement)
	el.Class().Add("icon")
	el.Class().Add("button")
	el.SetTextContent(text)
	el.AddEventListener("click", true, func(ev dom.Event) {
		go f(o.ui, ev)
	})
	return el
}

// NewIconButton creates a new button with an event listener associated with it.
// See https://fonts.google.com/icons for the list of available icons.
func (o *Owner) NewIconButton(parent dom.Element, text string, f func(ui UI, ev dom.Event)) *dom.HTMLAnchorElement {
	span := o.CreateChild(parent, "span").(*dom.HTMLSpanElement)
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
func (o *Owner) CreateTextNode(parent dom.Element, s string) *dom.Text {
	el := o.html.CreateTextNode(s)
	parent.AppendChild(el)
	return el
}

const sliderRangeMax = 10000

// NewSlider returns a new slider.
func (o *Owner) NewSlider(parent dom.Element, f func(ui UI, slider *dom.HTMLInputElement)) *dom.HTMLInputElement {
	container := o.CreateChild(parent, "div").(*dom.HTMLDivElement)
	container.Class().Add("slidecontainer")
	slider := o.CreateChild(container, "input").(*dom.HTMLInputElement)
	slider.Class().Add("slider")
	slider.SetAttribute("type", "range")
	slider.SetAttribute("min", "0")
	max := strconv.FormatUint(sliderRangeMax, 10)
	slider.SetAttribute("max", max)
	slider.SetAttribute("value", max)
	listener := func(js.Value, []js.Value) any {
		f(o.ui, slider)
		return nil
	}
	slider.Set("oninput", js.FuncOf(listener))
	return slider
}

// SliderValue returns the value of a slider between 0 and 1.
func SliderValue(slider *dom.HTMLInputElement) (float32, error) {
	valueS := slider.Get("value").String()
	value, err := strconv.ParseInt(valueS, 10, 64)
	if err != nil {
		return -1, err
	}
	return float32(value) / sliderRangeMax, nil
}
