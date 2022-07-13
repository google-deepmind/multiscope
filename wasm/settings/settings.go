// Package settings stores user settings.
package settings

import (
	"fmt"
	"net/http"

	"honnef.co/go/js/dom/v2"
)

// Settings stores settings in cookies.
type Settings struct {
}

func doc() dom.HTMLDocument {
	return dom.GetWindow().Document().(dom.HTMLDocument)
}

// Set a key,value pair in the settings.
func (s *Settings) Set(key, val string) {
	doc().SetCookie(fmt.Sprintf("%s=%s; Tue, 19 Jan 2038 04:14:07 GMT", key, val))
}

// Get a key,value pair from the setting.
func (s *Settings) Get(key string) (string, bool) {
	header := http.Header{}
	header.Add("Cookie", doc().Cookie())
	req := http.Request{Header: header}
	cookie, err := req.Cookie(key)
	if err != nil || cookie == nil {
		return "", false
	}
	return cookie.Value, true
}
