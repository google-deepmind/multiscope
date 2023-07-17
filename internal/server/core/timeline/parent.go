package timeline

import (
	"fmt"
	"multiscope/internal/server/core"
	treepb "multiscope/protos/tree_go_proto"

	"golang.org/x/exp/maps"
)

type constParent struct {
	this *constLeaf

	fetchChildrenError error
	children           map[string]Marshaler
	childrenStorage    uint64
}

// NewParent returns a tree fo serialized nodes that can be stored in a timeline.
func NewParent(storeRoot bool, node core.ParentNode) Marshaler {
	m := &constParent{}
	if storeRoot {
		m.this = newConstLeaf(node)
	}
	childrenNames, err := node.Children()
	if err != nil {
		m.fetchChildrenError = err
		return m
	}
	m.children = make(map[string]Marshaler)
	for _, childName := range childrenNames {
		childNode, err := node.Child(childName)
		var childMarshaler Marshaler
		if err != nil {
			childMarshaler = ToErrorMarshaler(err)
		} else {
			childMarshaler = ToMarshaler(childNode)
		}
		m.children[childName] = childMarshaler
		m.childrenStorage += childMarshaler.StorageSize()
	}
	return m
}

func (m *constParent) MarshalData(data *treepb.NodeData, path []string, lastTick uint32) {
	if len(path) == 0 {
		m.this.MarshalData(data, path, lastTick)
		return
	}
	if m.fetchChildrenError != nil {
		data.Error = m.fetchChildrenError.Error()
		return
	}
	childName := path[0]
	child := m.children[childName]
	if child == nil {
		data.Error = fmt.Sprintf("child %q in path %v cannot be found. Available children are: %v", childName, path, maps.Keys(m.children))
		return
	}
	child.MarshalData(data, path[1:], lastTick)
}

func (m *constParent) StorageSize() uint64 {
	size := m.childrenStorage
	if m.this != nil {
		size += m.this.StorageSize()
	}
	return size
}
