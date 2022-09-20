// Package texttesting provides helpers to test the text writer node.
package texttesting

import (
	"context"
	"fmt"
	"multiscope/internal/grpc/client"
	"multiscope/internal/mime"
	pbgrpc "multiscope/protos/tree_go_proto"

	"github.com/google/go-cmp/cmp"
)

// Text01Name is the name of the text writer in the tree.
const Text01Name = "text01"

// Text01Data is the data to write to the writer.
var Text01Data = []string{"Hello World", "Hello\nWorld", "Bonjour tout le monde"}

// CheckText01 checks the data exported by a text node.
func CheckText01(clt pbgrpc.TreeClient, path []string, i int) error {
	ctx := context.Background()
	nodes, err := client.PathToNodes(ctx, clt, path)
	if err != nil {
		return err
	}
	if diff := cmp.Diff(nodes[0].GetMime(), mime.PlainText); len(diff) > 0 {
		return fmt.Errorf("mime type error: %s", diff)
	}

	data, err := client.NodesData(ctx, clt, nodes)
	if err != nil {
		return err
	}
	text, err := client.ToRaw(data[0])
	if err != nil {
		return err
	}
	want := Text01Data[i]
	if string(text) != want {
		return fmt.Errorf("text error in test %d: got %s, want %s", i, string(text), want)
	}
	return nil
}
