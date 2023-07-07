package ticker

import (
	"multiscope/internal/server/writers/ticker/timedb"
	treepb "multiscope/protos/tree_go_proto"
)

// tlPlayer is a player than can be stored in a timedb.
type tlPlayer struct {
	player *Player
	db     *timedb.TimeDB
}

func newTLPlayer(p *Player) *tlPlayer {
	return &tlPlayer{player: p, db: p.db.Clone()}
}

func (p *tlPlayer) MarshalData(d *treepb.NodeData, path []string, lastTick uint32) {
	if len(path) == 0 {
		p.player.marshalDisplay(d, lastTick, p.db)
		return
	}
	p.player.marshalPastData(d, path, lastTick, p.db)
}

func (p *tlPlayer) StorageSize() uint64 {
	return p.db.StorageSize()
}
