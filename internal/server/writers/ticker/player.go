package ticker

import (
	"fmt"
	"multiscope/internal/mime"
	"multiscope/internal/server/core"
	"multiscope/internal/server/events"
	"multiscope/internal/server/treeservice"
	"multiscope/internal/server/writers/base"
	"multiscope/internal/server/writers/ticker/timeline"
	pb "multiscope/protos/ticker_go_proto"
	tickerpb "multiscope/protos/ticker_go_proto"
	treepb "multiscope/protos/tree_go_proto"

	"github.com/pkg/errors"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
)

// Player is a group node that displays timing data.
type Player struct {
	*base.Group

	tline *timeline.Timeline
}

var (
	_ core.Node       = (*Player)(nil)
	_ core.Parent     = (*Player)(nil)
	_ core.ChildAdder = (*Player)(nil)
)

// NewPlayer returns a new writer to stream data tables.
func NewPlayer() *Player {
	p := &Player{
		Group: base.NewGroup(mime.MultiscopePlayer),
	}
	p.tline = timeline.New(p.Group)
	return p
}

func (p *Player) addToTree(state treeservice.State, path *treepb.NodePath) (*core.Path, error) {
	playerPath, err := core.SetNodeAt(state.Root(), path, p)
	if err != nil {
		return nil, err
	}
	state.Events().Register(playerPath.Path(), func() events.Callback {
		return events.CallbackF(p.processEvents)
	})
	return playerPath, nil
}

func (p *Player) processEvents(ev *treepb.Event) (bool, error) {
	action := &pb.PlayerAction{}
	if err := anypb.UnmarshalTo(ev.Payload, action, proto.UnmarshalOptions{}); err != nil {
		return false, err
	}
	switch act := action.Action.(type) {
	case *pb.PlayerAction_TickView:
		return p.tline.SetTickView(act.TickView)
	default:
		return false, errors.Errorf("player action %T not implemented", act)
	}
}

// StoreFrame goes through the tree to store the current state of the nodes into a storage.
func (p *Player) StoreFrame(data *tickerpb.PlayerData) error {
	return p.tline.Store()
}

// MIME returns the mime type of this node.
func (p *Player) MIME() string {
	return mime.ProtoToMIME(&pb.PlayerInfo{})
}

// MarshalData retrieves the NodeData of a child based on path.
func (p *Player) MarshalData(d *treepb.NodeData, path []string, lastTick uint32) {
	if len(path) > 0 {
		p.tline.MarshalData(d, path)
		return
	}
	data := &pb.PlayerInfo{
		Timeline: p.tline.MarshalDisplay(),
	}
	anyPB, err := anypb.New(data)
	if err != nil {
		d.Error = fmt.Sprintf("cannot serialize the ticker proto: %v", err)
		return
	}
	d.Data = &treepb.NodeData_Pb{Pb: anyPB}
}

// Close the player and release the storage memory.
func (p *Player) Close() error {
	return p.tline.Close()
}
