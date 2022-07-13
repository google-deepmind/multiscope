// Package root implements the root of a Multiscope tree.
package root

import (
	"sync"

	"multiscope/internal/server/core"
	"multiscope/internal/server/writers/base"
	rootpb "multiscope/protos/root_go_proto"
	pb "multiscope/protos/tree_go_proto"
)

// Root is the root of the Multiscope tree.
type Root struct {
	mut sync.Mutex
	core.Root
	writer *base.ProtoWriter
	info   *rootpb.RootInfo
}

var _ core.Root = (*Root)(nil)

// NewRoot returns a new Multiscope root node.
func NewRoot() *Root {
	r := &Root{
		Root: base.NewRoot(),
		info: &rootpb.RootInfo{},
	}
	r.writer = base.NewProtoWriter(r.info)
	return r
}

// Path return the path to the root node.
func (r *Root) Path() *core.Path {
	return core.NewPath(r)
}

// MarshalData serializes the data of the root node.
func (r *Root) MarshalData(data *pb.NodeData, path []string, lastTick uint32) {
	if len(path) == 0 {
		r.writer.MarshalData(data, path, lastTick)
	}
	r.Root.MarshalData(data, path, lastTick)
}

func (r *Root) setLayout(layout *rootpb.Layout) error {
	r.mut.Lock()
	defer r.mut.Unlock()
	r.info.Layout = layout
	return r.writer.Write(r.info)
}
