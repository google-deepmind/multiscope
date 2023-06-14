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

package ticker

import (
	"fmt"
	"multiscope/internal/mime"
	"multiscope/internal/server/core"
	"multiscope/internal/server/events"
	"multiscope/internal/server/treeservice"
	"multiscope/internal/server/writers/base"
	"multiscope/internal/server/writers/ticker/timeline"
	tickerpb "multiscope/protos/ticker_go_proto"
	treepb "multiscope/protos/tree_go_proto"

	"github.com/pkg/errors"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
)

type (
	playerControl interface {
		setPeriod(period *tickerpb.SetPeriod) error
		processCommand(cmd tickerpb.Command) error
		mainNextStep()
		pause()
		close()
	}

	// Player is a group node that displays timing data.
	Player struct {
		*base.Group
		tline   *timeline.Timeline
		control playerControl
		queue   *events.Queue
	}
)

var (
	_ core.Node        = (*Player)(nil)
	_ core.Parent      = (*Player)(nil)
	_ core.ChildAdder  = (*Player)(nil)
	_ core.NodeReseter = (*Player)(nil)
)

// NewPlayer returns a new writer to stream data tables.
func NewPlayer(ignorePause bool) *Player {
	p := &Player{
		Group: base.NewGroup(mime.MultiscopePlayer),
	}
	p.tline = timeline.New(p.Group)
	if ignorePause {
		p.control = newNoPauseControl(p)
	} else {
		p.control = newSyncControl(p)
	}
	return p
}

func (p *Player) addToTree(state *treeservice.State, path *treepb.NodePath) (*core.Path, error) {
	playerPath, err := core.SetNodeAt(state.Root(), path, p)
	if err != nil {
		return nil, err
	}
	p.queue = state.Events().NewQueueForPath(playerPath.Path(), p.processEvents)
	return playerPath, nil
}

func (p *Player) processEvents(ev *treepb.Event) error {
	action := &tickerpb.PlayerAction{}
	if err := anypb.UnmarshalTo(ev.Payload, action, proto.UnmarshalOptions{}); err != nil {
		return err
	}
	var err error
	switch act := action.Action.(type) {
	case *tickerpb.PlayerAction_TickView:
		err = p.tline.SetTickView(act.TickView)
		p.control.pause()
	case *tickerpb.PlayerAction_Command:
		switch act.Command {
		case tickerpb.Command_CMD_STEP:
			err = p.tline.SetTickView(&tickerpb.SetTickView{
				TickCommand: &tickerpb.SetTickView_Offset{Offset: 1},
			})
			p.control.pause()
		case tickerpb.Command_CMD_STEPBACK:
			err = p.tline.SetTickView(&tickerpb.SetTickView{
				TickCommand: &tickerpb.SetTickView_Offset{Offset: -1},
			})
			p.control.pause()
		default:
			err = p.control.processCommand(act.Command)
		}
	case *tickerpb.PlayerAction_SetPeriod:
		err = p.control.setPeriod(act.SetPeriod)
	default:
		err = errors.Errorf("player action %T not implemented", act)
	}
	return err
}

// ResetNode resets the node.
func (p *Player) ResetNode() error {
	p.control.mainNextStep()
	return p.tline.Reset()
}

// StoreFrame goes through the tree to store the current state of the nodes into a storage.
func (p *Player) StoreFrame(data *tickerpb.PlayerData) error {
	p.control.mainNextStep()
	return p.tline.Store()
}

// MIME returns the mime type of this node.
func (p *Player) MIME() string {
	return mime.ProtoToMIME(&tickerpb.PlayerInfo{})
}

// MarshalData retrieves the NodeData of a child based on path.
func (p *Player) MarshalData(d *treepb.NodeData, path []string, lastTick uint32) {
	if len(path) > 0 {
		p.tline.MarshalData(d, path)
		return
	}
	data := &tickerpb.PlayerInfo{
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
	p.queue.Delete()
	p.control.close()
	return p.tline.Close()
}
