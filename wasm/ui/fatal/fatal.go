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

// Package fatal displays a fatal error on the web page.
// A fatal error requires the user to reload the page.
package fatal

import (
	"fmt"
	"strings"

	"github.com/pkg/errors"
	"honnef.co/go/js/dom/v2"
)

type stackTracer interface {
	StackTrace() errors.StackTrace
}

const coreError = "\n\n\n===========\nCORE ERROR:"

func formatCoreError(err error) string {
	last := err
	for unwrap := last; unwrap != nil; unwrap = errors.Unwrap(unwrap) {
		last = unwrap
	}
	return coreError + last.Error()
}

// FormatError returns a formatted error by reporting the stack trace of the cause of the error.
func FormatError(err error) error {
	if err == nil {
		return nil
	}
	s := err.Error()
	var withSt stackTracer
	if errors.As(err, &withSt) {
		s += fmt.Sprintf("%s%+v\n", coreError, withSt.StackTrace())
	} else {
		s += formatCoreError(err)
	}
	return errors.New(s)
}

// Display a fatal error.
func Display(err error) {
	html := fmt.Sprintf("FATAL ERROR:\n%v.", FormatError(err))
	html = strings.ReplaceAll(html, "\n", "</br>")
	doc := dom.GetWindow().Document().(dom.HTMLDocument)
	doc.Body().SetInnerHTML(html)
}
