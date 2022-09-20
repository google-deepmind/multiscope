// Package fatal displays a fatal error on the web page.
// A fatal error requires the user to reload the page.
package fatal

import (
	"fmt"
	"strings"

	"honnef.co/go/js/dom/v2"
)

// Display a fatal error.
func Display(err error) {
	html := fmt.Sprintf("FATAL ERROR:\n%v.", err)
	html = strings.ReplaceAll(html, "\n", "</br>")
	doc := dom.GetWindow().Document().(dom.HTMLDocument)
	doc.Body().SetInnerHTML(html)
}
