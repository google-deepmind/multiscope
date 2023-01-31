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
