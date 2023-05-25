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

// tree_test contains tests for service.go
package tree_test

import (
	"context"
	"testing"

	"multiscope/internal/grpc/grpctesting"
	"multiscope/internal/server/root"
	"multiscope/internal/server/writers/base"
	"multiscope/internal/server/writers/base/basetesting"
	basepb "multiscope/protos/base_go_proto"
	basepbgrpc "multiscope/protos/base_go_proto"
	pb "multiscope/protos/tree_go_proto"
)

func TestProtoWriter(t *testing.T) {
	state := grpctesting.NewState(root.NewRoot(), nil, nil)
	conn, clt, err := grpctesting.SetupTest(state, base.RegisterService)
	if err != nil {
		t.Fatalf("testing.Start(base.RegisterService): %v", err)
	}
	writerClt := basepbgrpc.NewBaseWritersClient(conn)

	ctx := context.Background()

	path := &pb.NodePath{
		Path: []string{basetesting.Proto01Name},
	}

	examplePb, err := basetesting.TimestampToAny(basetesting.Proto01Data[0])
	if err != nil {
		t.Fatal(err)
	}

	pwReq := &basepb.NewProtoWriterRequest{
		Path:  path,
		Proto: examplePb}
	rep, err := writerClt.NewProtoWriter(ctx, pwReq)
	if err != nil {
		t.Fatalf("writerClt.NewProtoWriter(%v, %v): %v", ctx, pwReq, err)
	}
	writer := rep.GetWriter()
	for i, want := range basetesting.Proto01Data {
		any, err := basetesting.TimestampToAny(want)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := writerClt.WriteProto(ctx, &basepb.WriteProtoRequest{
			Writer: writer,
			Proto:  any,
		}); err != nil {
			t.Error(err)
			break
		}
		path := []string{basetesting.Proto01Name}
		if err := basetesting.CheckProto01(ctx, clt, path, i); err != nil {
			t.Fatalf("basetesting.CheckProto01(%v, %v, %v, %d): %v", ctx, clt, path, i, err)
		}
	}
}

func TestRawWriter(t *testing.T) {
	state := grpctesting.NewState(root.NewRoot(), nil, nil)
	conn, clt, err := grpctesting.SetupTest(state, base.RegisterService)
	if err != nil {
		t.Fatalf("testing.Start(tree.RegisterService): %v", err)
	}
	writerClt := basepbgrpc.NewBaseWritersClient(conn)

	ctx := context.Background()

	path := &pb.NodePath{
		Path: []string{basetesting.Raw01Name},
	}

	rwReq := &basepb.NewRawWriterRequest{
		Path: path,
		Mime: "TEST_BYTES"}
	rep, err := writerClt.NewRawWriter(ctx, rwReq)
	if err != nil {
		t.Fatalf("writerClt.NewRawWriter(%v, %v): %v", ctx, rwReq, err)
	}
	writer := rep.GetWriter()
	for i, want := range basetesting.Raw01Data {
		wReq := &basepb.WriteRawRequest{
			Writer: writer,
			Data:   want,
		}
		if _, err := writerClt.WriteRaw(ctx, wReq); err != nil {
			t.Fatalf("writerClt.WriteRaw(%v, %v): %v", ctx, wReq, err)
		}
		path := []string{basetesting.Raw01Name}
		if err := basetesting.CheckRaw01(ctx, clt, path, i); err != nil {
			t.Fatalf("basetesting.CheckRaw01(%v, %v, %v, %d): %v", ctx, clt, path, i, err)
		}
	}
}
