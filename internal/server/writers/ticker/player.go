package ticker

import (
	"multiscope/internal/mime"
	"multiscope/internal/server/core"
	"multiscope/internal/server/treeservice"
	"multiscope/internal/server/writers/base"
	tickerpb "multiscope/protos/ticker_go_proto"
	treepb "multiscope/protos/tree_go_proto"
)

// Player is a group node that displays timing data.
type Player struct {
	*base.Group
	writer *base.ProtoWriter
}

var (
	_ core.Node       = (*Player)(nil)
	_ core.Parent     = (*Player)(nil)
	_ core.ChildAdder = (*Player)(nil)
)

// NewPlayer returns a new writer to stream data tables.
func NewPlayer() *Player {
	return &Player{
		Group:  base.NewGroup(mime.MultiscopePlayer),
		writer: base.NewProtoWriter(&tickerpb.PlayerData{}),
	}
}

func (t *Player) addToTree(state treeservice.State, path *treepb.NodePath) (*core.Path, error) {
	return core.SetNodeAt(state.Root(), path, t)
}

// StoreFrame goes through the tree to store the current state of the nodes into a storage.
func (t *Player) StoreFrame(data *tickerpb.PlayerData) error {
	return t.writer.Write(data)
}

// MIME returns the mime type of this node.
func (t *Player) MIME() string {
	return t.writer.MIME()
}

// MarshalData writes the node data into a NodeData protocol buffer.
func (t *Player) MarshalData(data *treepb.NodeData, path []string, lastTick uint32) {
	if len(path) == 0 {
		t.writer.MarshalData(data, path, lastTick)
		return
	}
	t.Group.MarshalData(data, path, lastTick)
}
