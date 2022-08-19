package scope

import (
	"fmt"
	"io/fs"
	"io/ioutil"
	"log"
	"multiscope/internal/httpgrpc"
	"multiscope/internal/server/worker"
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

func readIndexHTML(root fs.FS) ([]byte, error) {
	const indexHTML = "res/index.html"
	file, err := root.Open(indexHTML)
	if err != nil {
		return nil, fmt.Errorf("error opening %q: %v", indexHTML, err)
	}
	defer file.Close()
	buf, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("cannot read %q: %v", indexHTML, err)
	}
	return buf, nil
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
	r.Get("/worker/*", worker.Handle)
	root := web.FS()
	r.Get("/multiscope", func(w http.ResponseWriter, r *http.Request) {
		buf, err := readIndexHTML(root)
		if err != nil {
			wErr(w, "cannot read index.html: %v", err)
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
