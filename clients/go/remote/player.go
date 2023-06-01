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

package remote

import (
	"context"
	pb "multiscope/protos/ticker_go_proto"
	pbgrpc "multiscope/protos/ticker_go_proto"

	"github.com/pkg/errors"
)

// Player is a folder in the tree.
type Player struct {
	*ClientNode
	clt    pbgrpc.TickersClient
	player *pb.Player

	tick int
}

// NewPlayer creates a new group in the tree.
func NewPlayer(clt *Client, name string, ignorePause bool, parent Path) (*Player, error) {
	p := &Player{
		clt: pbgrpc.NewTickersClient(clt.Connection()),
	}
	ctx := context.Background()
	path := clt.toChildPath(name, parent)
	rep, err := p.clt.NewPlayer(ctx, &pb.NewPlayerRequest{
		TreeId:      clt.TreeID(),
		Path:        path.NodePath(),
		IgnorePause: ignorePause,
	})
	if err != nil {
		return nil, errors.Errorf("cannot create a new Player: %v", err)
	}
	p.player = rep.GetPlayer()
	if p.player == nil {
		return nil, errors.Errorf("server has returned a nil ticker")
	}
	p.ClientNode = NewClientNode(clt, toPath(p.player))
	if err := clt.Display().DisplayIfDefault(p.Path()); err != nil {
		return nil, err
	}
	return p, nil
}

// StoreFrame stores all the data from nodes under a player.
func (p *Player) StoreFrame() error {
	p.tick++
	ctx := context.Background()
	_, err := p.clt.StoreFrame(ctx, &pb.StoreFrameRequest{
		Player: p.player,
		Data: &pb.PlayerData{
			Tick: int64(p.tick),
		},
	})
	if err != nil {
		return errors.Errorf("cannot store frame: %v", err)
	}
	return err
}

// CurrentTick returns the current frame number.
func (p *Player) CurrentTick() int {
	return p.tick
}
