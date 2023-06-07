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
	ctx := context.Background()
	nodes, err := client.PathToNodes(ctx, clt, []string{})
	if err != nil {
		return err
	}
	data, err := client.NodesData(ctx, clt, nodes)
	if err != nil {
		return err
	}
	root := pb.RootInfo{}
	if err := client.ToProto(data[0], &root); err != nil {
		return err
	}
	got := []string{}
	if root.Layout == nil {
		root.Layout = &pb.Layout{
			Layout: &pb.Layout_List{
				List: &pb.LayoutList{},
			},
		}
	}
	for _, path := range root.Layout.GetList().Displayed {
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
	if err = checkDisplayed(clt, want); err != nil {
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
