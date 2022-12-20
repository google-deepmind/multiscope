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
func NewPlayer(clt *Client, name string, parent Path) (*Player, error) {
	p := &Player{
		clt: pbgrpc.NewTickersClient(clt.Connection()),
	}
	ctx := context.Background()
	path := clt.toChildPath(name, parent)
	rep, err := p.clt.NewPlayer(ctx, &pb.NewPlayerRequest{
		Path: path.NodePath(),
	})
	if err != nil {
		return nil, err
	}
	p.player = rep.GetPlayer()
	if p.player == nil {
		return nil, errors.New("server has returned a nil ticker")
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
	return err
}

// CurrentTick returns the current frame number.
func (p *Player) CurrentTick() int {
	return p.tick
}
