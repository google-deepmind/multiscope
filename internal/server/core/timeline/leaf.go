package timeline

import (
	"multiscope/internal/server/core"
	treepb "multiscope/protos/tree_go_proto"

	"google.golang.org/protobuf/proto"
)

type constLeaf struct {
	data *treepb.NodeData
}

func newConstLeaf(node core.Node) *constLeaf {
	leaf := &constLeaf{data: &treepb.NodeData{}}
	node.MarshalData(leaf.data, nil, 0)
	return leaf
}

func (m *constLeaf) MarshalData(data *treepb.NodeData, path []string, lastTick uint32) {
	if lastTick == m.data.Tick {
		return
	}
	data.Data = m.data.Data
	data.Mime = m.data.Mime
	data.Error = m.data.Error
	data.Tick = m.data.Tick
}

func (m *constLeaf) StorageSize() uint64 {
	return uint64(proto.Size(m.data))
}

// ToErrorMarshaler stores an error in a marshaler.
func ToErrorMarshaler(err error) Marshaler {
	return &constLeaf{
		data: &treepb.NodeData{
			Error: err.Error(),
		},
	}
}
