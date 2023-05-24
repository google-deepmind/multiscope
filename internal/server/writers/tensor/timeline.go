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

package tensor

import (
	"multiscope/internal/mime"
	"multiscope/internal/server/core"
	pb "multiscope/protos/tensor_go_proto"
	treepb "multiscope/protos/tree_go_proto"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
)

type tlNode struct {
	w    *Writer
	last *pb.Tensor
}

func newAdapter(w *Writer) *tlNode {
	return &tlNode{w: w}
}

func (adpt *tlNode) update(pbt *pb.Tensor) {
	adpt.last = pbt
}

func (adpt *tlNode) MIME() string {
	return mime.ProtoToMIME(adpt.last)
}

func (adpt *tlNode) MarshalData(data *treepb.NodeData, path []string, lastTick uint32) {
	err := adpt.w.forceUpdate()
	if err != nil {
		data.Error = err.Error()
		return
	}

	dst := &anypb.Any{}
	if err := anypb.MarshalFrom(dst, adpt.last, proto.MarshalOptions{}); err != nil {
		data.Error = err.Error()
		return
	}
	data.Data = &treepb.NodeData_Pb{Pb: dst}
}

// Child returns a child node given its ID.
func (adpt *tlNode) Child(name string) (core.Node, error) {
	return adpt.w.Child(name)
}

// Children returns the (sorted) list of names of all children of this node.
func (adpt *tlNode) Children() ([]string, error) {
	return adpt.w.Children()
}
