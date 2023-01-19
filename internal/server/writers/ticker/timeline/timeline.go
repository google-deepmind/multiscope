// Package timeline implements the timeline of the player.
package timeline

import (
	"fmt"
	"math"
	"sync"

	"multiscope/internal/server/core"
	"multiscope/internal/server/writers/ticker/storage"
	"multiscope/internal/server/writers/ticker/timedb"
	pb "multiscope/protos/ticker_go_proto"
	treepb "multiscope/protos/tree_go_proto"

	"github.com/pkg/errors"
	"google.golang.org/protobuf/proto"
)

// Timeline stores data for a sub-tree for all ticks.
type Timeline struct {
	mux sync.Mutex

	currentTick uint64
	displayTick int64
	db          *timedb.TimeDB
}

// New returns a new time given a node in the tree.
func New(node core.Node) *Timeline {
	return &Timeline{
		db:          timedb.New(node),
		currentTick: 0,
		displayTick: math.MaxInt64,
	}
}

// MarshalDisplay sets the timeline information.
func (tl *Timeline) MarshalDisplay() *pb.TimeLine {
	tl.mux.Lock()
	defer tl.mux.Unlock()

	if tl.db.NumRecords() == 0 {
		return nil
	}
	currentStorage := tl.db.StorageSize()
	maxStorage := storage.Global().Available()
	storageCapacity := fmt.Sprintf("%.e/%.e (%d%%)",
		float64(currentStorage),
		float64(maxStorage),
		int(float64(currentStorage)/float64(maxStorage)*100))
	return &pb.TimeLine{
		DisplayTick:     uint64(tl.adjustDisplayTick(tl.displayTick)),
		OldestTick:      tl.db.Oldest(),
		HistoryLength:   uint64(tl.db.NumRecords()),
		StorageCapacity: storageCapacity,
	}
}

func (tl *Timeline) setDisplayTick(displayTick uint64) {
	tl.mux.Lock()
	defer tl.mux.Unlock()

	tl.displayTick = int64(displayTick)
}

func (tl *Timeline) offsetDisplayTick(offset int64) {
	tl.mux.Lock()
	defer tl.mux.Unlock()

	tl.displayTick = tl.adjustDisplayTick(tl.displayTick)
	tl.displayTick += offset
}

// SetTickView set the view to display.
func (tl *Timeline) SetTickView(view *pb.SetTickView) error {
	switch cmd := view.TickCommand.(type) {
	case *pb.SetTickView_ToDisplay:
		tl.setDisplayTick(cmd.ToDisplay)
	case *pb.SetTickView_Offset:
		tl.offsetDisplayTick(cmd.Offset)
	default:
		return errors.Errorf("tick command not supported: %T", cmd)
	}
	return nil
}

// Store the data for all children.
func (tl *Timeline) Store() error {
	tl.db.Store(tl.currentTick)

	tl.mux.Lock()
	defer tl.mux.Unlock()
	tl.currentTick++

	return nil
}

func (tl *Timeline) adjustDisplayTick(tick int64) (out int64) {
	defer func() {
		if out < 0 {
			out = 0
		}
	}()

	oldest := int64(tl.db.Oldest())
	if tick < int64(tl.db.Oldest()) {
		// Adjust to the oldest tick.
		return oldest
	}
	if tick < int64(tl.currentTick) {
		// Valid value between the oldest tick and the current tick.
		return tick
	}
	if tl.currentTick == 0 {
		// No data has been stored yet.
		return 0
	}
	// Adjust to the latest tick.
	return int64(tl.currentTick) - 1
}

// MarshalData serializes the data given the current tick being displayed.
func (tl *Timeline) MarshalData(data *treepb.NodeData, path []string) {
	tl.mux.Lock()
	defer tl.mux.Unlock()

	displayTick := uint64(tl.adjustDisplayTick(tl.displayTick))
	rec := tl.db.Fetch(displayTick)
	if rec == nil {
		data.Error = fmt.Sprintf("data for tick %d does not exist", displayTick)
		return
	}
	for _, p := range path {
		child := rec.Child(p)
		if child == nil {
			data.Error = fmt.Sprintf("child %q in path %v cannot be found in the timeline. Available children are: %v", p, path, rec.Children())
			return
		}
		rec = child
	}
	proto.Merge(data, rec.Data())
}

// CurrentTick returns the next frame to store.
func (tl *Timeline) CurrentTick() uint64 {
	tl.mux.Lock()
	defer tl.mux.Unlock()

	return tl.currentTick
}

// IsLastTickDisplayed returns true if the last tick is being displayed.
func (tl *Timeline) IsLastTickDisplayed() bool {
	tl.mux.Lock()
	defer tl.mux.Unlock()

	return tl.displayTick >= int64(tl.currentTick)
}

// Close the timeline and release all the memory.
func (tl *Timeline) Close() error {
	tl.db.Close()
	return nil
}
