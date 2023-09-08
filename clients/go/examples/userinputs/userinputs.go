// Command userinputs shows Multiscope capturing keyboard events.
package main

import (
	"flag"
	"fmt"
	"log"
	"multiscope/clients/go/scope"
	"multiscope/internal/fmtx"

	"google.golang.org/protobuf/encoding/prototext"
)

var (
	httpPort = flag.Int("http_port", scope.DefaultPort, "http port")
	local    = flag.Bool("local", true, "open the port to local connection only")

	writer *scope.HTMLWriter
)

func htmlWrite(title, text string) error {
	t := fmt.Sprintf("*%s*:\n%s", title, text)
	return writer.Write(t)
}

func keyboardHandler(event *scope.EventKeyboard) error {
	if event.Shift {
		return nil
	}
	eventBytes, _ := prototext.MarshalOptions{Multiline: true}.Marshal(event)
	err := htmlWrite("keyboardHandler", string(eventBytes))
	return err
}

func main() {
	flag.Parse()
	if err := scope.Start(*httpPort, *local); err != nil {
		log.Fatal(fmtx.FormatError(err))
	}
	var err error
	if writer, err = scope.NewHTMLWriter("Input panel", nil); err != nil {
		log.Fatal(err)
	}
	if err := htmlWrite("No input", ""); err != nil {
		log.Fatal(err)
	}
	if err := scope.RegisterKeyboardCallback(keyboardHandler); err != nil {
		log.Fatal(err)
	}

	<-make(chan bool)
}
