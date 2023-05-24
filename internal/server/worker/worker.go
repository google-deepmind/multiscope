// Copyright 2023 Google LLC
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

// Package worker provides web-worker on the server side.
package worker

import (
	"fmt"
	"io/fs"
	"log"
	"multiscope/internal/server/wasmurl"
	"multiscope/internal/template"
	"net/http"
	"strings"
)

func wErrf(w http.ResponseWriter, format string, a ...any) {
	fmt.Println("error:", fmt.Sprintf(format, a...))
	if _, err := fmt.Fprintf(w, format, a...); err != nil {
		log.Printf("error sending the error back to the client: %v", err)
	}
}

func parseFuncName(s string) (string, error) {
	prefix, funcName, found := strings.Cut(s, "/")
	if prefix == "" {
		prefix, funcName, found = strings.Cut(funcName, "/")
	}
	if !found {
		return "", fmt.Errorf("malformed URL %q: cannot find separator '/'", s)
	}
	if prefix != "worker" {
		return "", fmt.Errorf("URL has the wrong prefix: got %q but want worker", prefix)
	}
	return funcName, nil
}

// Handle requests to start new web workers.
func Handle(w http.ResponseWriter, r *http.Request, root fs.FS) {
	w.Header().Set("content-type", "application/javascript")
	funcName, err := parseFuncName(r.URL.Path)
	if err != nil {
		wErrf(w, "cannot provide worker: %v", err)
		return
	}
	args := map[string]string{
		"funcName": funcName,
		"wasmURL":  wasmurl.URL(),
	}
	if err := template.Execute(w, root, "res/worker.js", args); err != nil {
		wErrf(w, "error parsing template: %v", err)
		return
	}
}
