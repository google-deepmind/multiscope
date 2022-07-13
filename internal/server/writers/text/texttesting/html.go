package texttesting

import (
	"context"
	"fmt"
	"multiscope/internal/grpc/client"
	"multiscope/internal/mime"
	pbgrpc "multiscope/protos/tree_go_proto"

	"github.com/google/go-cmp/cmp"
)

const (
	// HTML01Name is the name of the HTML writer in the tree.
	HTML01Name = "text01"
	// CSS01Data is the data being written
	CSS01Data = "This is some invalid CSS"
)

// HTML01Data is the data being written to the HTML writer.
var HTML01Data = []string{"Hello World", "Hello<br>World"}

// CheckHTML01 checks the data that can be read from the html01 node.
func CheckHTML01(clt pbgrpc.TreeClient, path []string, i int) error {
	ctx := context.Background()
	nodes, err := client.PathToNodes(ctx, clt,
		path,
		append(append([]string{}, path...), mime.NodeNameHTML),
		append(append([]string{}, path...), mime.NodeNameCSS),
	)
	if err != nil {
		return err
	}
	for i, mimeType := range []string{mime.HTMLParent, mime.HTMLText, mime.CSSText} {
		if diff := cmp.Diff(nodes[i].GetMime(), mimeType); len(diff) > 0 {
			return fmt.Errorf("mime type error: %s", diff)
		}
	}

	data, err := client.NodesData(ctx, clt, nodes)
	if err != nil {
		return err
	}
	html, err := client.ToRaw(data[1])
	if err != nil {
		return fmt.Errorf("cannot read html data: %v", err)
	}
	bodyGot := string(html)
	if bodyGot != HTML01Data[i] {
		return fmt.Errorf("html body error in test %d: got %q, want %q", i, bodyGot, HTML01Data[i])
	}

	css, err := client.ToRaw(data[2])
	if err != nil {
		return fmt.Errorf("cannot read css data: %v", err)
	}
	cssGot := string(css)
	if cssGot != CSS01Data {
		return fmt.Errorf("html css error in test %d: got %q, want %q", i, cssGot, CSS01Data)
	}
	return nil
}
