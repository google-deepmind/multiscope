package scope

import (
	"fmt"
	"io/ioutil"
	"log"
	"multiscope/internal/httpgrpc"
	"multiscope/internal/worker/workerserver"
	"multiscope/web"
	"net/http"
	"sync"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

const logQuery = false

func wErr(w http.ResponseWriter, format string, a ...interface{}) {
	fmt.Println("error:", fmt.Sprintf(format, a...))
	if _, err := w.Write([]byte(fmt.Sprintf(format, a...))); err != nil {
		log.Printf("error writing error back to the client: %v", err)
	}
}

// RunHTTP the http server serving both httpgrpc and the ui.
func RunHTTP(srv httpgrpc.Service, wg *sync.WaitGroup, addr string) error {
	r := chi.NewRouter()
	r.Use(middleware.NoCache)
	if logQuery {
		r.Use(middleware.Logger)
	}
	server := httpgrpc.NewServer()
	if err := server.Register(srv); err != nil {
		return fmt.Errorf("cannot register service: %w", err)
	}
	r.HandleFunc("/httpgrpc", server.Post)
	r.Get("/worker/*", workerserver.Handle)
	root := web.FS()
	const indexHTML = "res/index.html"
	r.Get("/multiscope", func(w http.ResponseWriter, r *http.Request) {
		file, err := root.Open(indexHTML)
		if err != nil {
			wErr(w, "error opening %q: %v", indexHTML, err)
			return
		}
		defer file.Close()
		buf, err := ioutil.ReadAll(file)
		if err != nil {
			wErr(w, "cannot read %q: %v", indexHTML, err)
			return
		}
		if _, err := w.Write(buf); err != nil {
			wErr(w, "error writing content: %v", err)
			return
		}
	})
	r.Get("/res/*", func(w http.ResponseWriter, r *http.Request) {
		http.FileServer(http.FS(root)).ServeHTTP(w, r)
	})
	r.Get("/", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "multiscope", 301)
	})
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := http.ListenAndServe(addr, r); err != nil {
			log.Fatalf("cannot run HTTP server: %v", err)
		}
	}()
	return nil
}
