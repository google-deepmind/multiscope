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

package scope

import (
	"fmt"
	"log"
	"multiscope/internal/httpgrpc"
	"multiscope/internal/server/httphooks"
	"multiscope/internal/server/treeservice"
	"multiscope/internal/server/wasmurl"
	"multiscope/internal/server/worker"
	"multiscope/internal/template"
	"multiscope/web"
	"net/http"
	"strings"
	"sync"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

const logQuery = false

func wErrf(w http.ResponseWriter, format string, a ...any) {
	fmt.Println("error:", fmt.Sprintf(format, a...))
	if _, err := fmt.Fprintf(w, format, a...); err != nil {
		log.Printf("error writing error back to the client: %v", err)
	}
}

// RunHTTP the http server serving both httpgrpc and the ui.
func RunHTTP(srv *treeservice.TreeServer, wg *sync.WaitGroup, addr string) error {
	r := chi.NewRouter()
	r.Use(middleware.NoCache)
	if logQuery {
		r.Use(middleware.Logger)
	}
	server := httpgrpc.NewServer()
	srv.RegisterServices(server)
	r.HandleFunc("/httpgrpc", server.Post)
	root := web.FS()
	r.Get("/worker/*", func(w http.ResponseWriter, r *http.Request) {
		worker.Handle(w, r, root)
	})
	r.Get("/multiscope", func(w http.ResponseWriter, r *http.Request) {
		args := map[string]string{"wasmURL": wasmurl.URL()}
		if err := template.Execute(w, root, "res/index.html", args); err != nil {
			wErrf(w, "error parsing template: %v", err)
			return
		}
	})
	r.Get("/res/*", func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, ".wasm") {
			r.URL.Path += ".gz"
			w.Header().Add("Content-Encoding", "gzip")
		}
		http.FileServer(http.FS(root)).ServeHTTP(w, r)
	})
	r.Get("/", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "multiscope", 301)
	})
	if err := httphooks.RunAll(r); err != nil {
		return err
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := http.ListenAndServe(addr, r); err != nil {
			log.Fatalf("cannot run HTTP server: %v", err)
		}
	}()
	return nil
}
