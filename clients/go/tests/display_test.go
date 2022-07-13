package scope_test

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"multiscope/clients/go/clienttesting"
	"multiscope/clients/go/remote"
	"multiscope/internal/grpc/client"
	pb "multiscope/protos/root_go_proto"

	"github.com/google/go-cmp/cmp"
)

func checkDisplayed(clt *remote.Client, want []string) error {
	treeClt := clt.TreeClient()
	ctx := context.Background()
	nodes, err := client.PathToNodes(ctx, treeClt, []string{})
	if err != nil {
		return err
	}
	data, err := client.NodesData(ctx, treeClt, nodes)
	if err != nil {
		return err
	}
	root := pb.RootInfo{}
	if err := client.ToProto(data[0], &root); err != nil {
		return err
	}
	got := []string{}
	if root.Layout == nil {
		root.Layout = &pb.Layout{}
	}
	for _, path := range root.Layout.Displayed {
		got = append(got, strings.Join(path.GetPath(), "/"))
	}
	if !cmp.Equal(got, want) {
		return fmt.Errorf("displayed path error: got %v but want %v", got, want)
	}
	return nil
}

func TestOnDisplayByDefault(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}
	want := []string{}
	if err := checkDisplayed(clt, want); err != nil {
		t.Errorf("error at initialization: %v", err)
	}
	ticker, err := remote.NewTicker(clt, "ticker", nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = remote.NewScalarWriter(clt, "displayed", ticker.Path())
	if err != nil {
		t.Fatal(err)
	}
	want = []string{"ticker", "ticker/displayed"}
	if err := checkDisplayed(clt, want); err != nil {
		t.Errorf("error at initialization: %v", err)
	}
}
