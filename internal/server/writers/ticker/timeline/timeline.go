// Package timeline implements the timeline of the player.
package timeline

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"multiscope/internal/server/core"
	"multiscope/internal/server/writers/ticker/storage"
	pb "multiscope/protos/ticker_go_proto"
	treepb "multiscope/protos/tree_go_proto"

	"github.com/pkg/errors"
	"go.uber.org/multierr"
	"google.golang.org/protobuf/proto"
)

const maxNumberOfSteps = 1e6

// We store the data in a tree such that we can record the data or error
// for all levels.
// For instance, for the path /a/b/c/, we may have an error at level /a
// (e.g. cannot get the children).
// Storing a recursive data structure, as opposed to a flat structure, makes
// it easy to retrieve the error at level /a if the path /a/b/c is requested.
type nodeRecord struct {
	data   *treepb.NodeData
	toData map[string]*nodeRecord
	err    error
}

type tickData struct {
	size uint64
	root *nodeRecord
}

// Timeline stores data for a sub-tree for all ticks.
type Timeline struct {
	mux          sync.Mutex
	root         core.Node
	currentTick  uint64
	displayTick  uint64
	oldestTick   uint64
	tickToRecord map[uint64]*tickData
	storage      uint64
}

// New returns a new time given a node in the tree.
func New(root core.Node) *Timeline {
	storage.Global().Register()
	return &Timeline{
		root:         root,
		tickToRecord: make(map[uint64]*tickData),
		currentTick:  0,
		displayTick:  math.MaxInt64,
	}
}

// MarshalDisplay sets the timeline information.
func (tl *Timeline) MarshalDisplay() *pb.TimeLine {
	tl.mux.Lock()
	defer tl.mux.Unlock()

	if len(tl.tickToRecord) == 0 {
		return nil
	}
	maxStorage := storage.Global().Available()
	storageCapacity := fmt.Sprintf("%.e/%.e (%d%%)",
		float64(tl.storage),
		float64(maxStorage),
		int(float64(tl.storage)/float64(maxStorage)*100))
	return &pb.TimeLine{
		DisplayTick:     tl.adjustDisplayTick(tl.displayTick),
		OldestTick:      tl.oldestTick,
		HistoryLength:   uint64(len(tl.tickToRecord)),
		StorageCapacity: storageCapacity,
	}
}

func (tl *Timeline) setDisplayTick(displayTick uint64) {
	tl.mux.Lock()
	defer tl.mux.Unlock()

	tl.displayTick = displayTick
}

func (tl *Timeline) offsetDisplayTick(offset int64) {
	tl.mux.Lock()
	defer tl.mux.Unlock()

	tl.displayTick = uint64(int64(tl.displayTick) + offset)
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

type tickContext struct {
	parentPath []string
	current    core.Node
	tckData    *tickData
}

func (tl *Timeline) recursiveTick(ctx *tickContext) *nodeRecord {
	record := &nodeRecord{}
	data := &treepb.NodeData{}
	ctx.current.MarshalData(data, nil, 0)
	if data.Data != nil {
		record.data = data
		ctx.tckData.size += uint64(proto.Size(data))
	}
	parent, ok := ctx.current.(core.Parent)
	if !ok {
		return record
	}
	children, err := parent.Children()
	if err != nil {
		record.err = err
		return record
	}
	for _, childName := range children {
		child, childErr := parent.Child(childName)
		if childErr != nil {
			err = multierr.Append(err, childErr)
			continue
		}
		childPath := append([]string{}, ctx.parentPath...)
		childPath = append(childPath, childName)
		if record.toData == nil {
			record.toData = make(map[string]*nodeRecord)
		}
		record.toData[childName] = tl.recursiveTick(&tickContext{
			parentPath: childPath,
			current:    child,
			tckData:    ctx.tckData,
		})
	}
	record.err = err
	return record
}

func (tl *Timeline) cleanupStorage() {
	const targetRatio = .9
	targetStorage := uint64(float32(storage.Global().Available()) * targetRatio)
	targetNumberOfSteps := int(float32(len(tl.tickToRecord)) * targetRatio)
	for tl.storage > targetStorage || len(tl.tickToRecord) > targetNumberOfSteps {
		oldestRecord := tl.tickToRecord[tl.oldestTick]
		tl.storage -= oldestRecord.size
		delete(tl.tickToRecord, tl.oldestTick)
		tl.oldestTick++
	}
	runtime.GC()
}

// Store the data for all children.
func (tl *Timeline) Store() error {
	// Save the data for all children.
	tckData := &tickData{}
	tckData.root = tl.recursiveTick(&tickContext{
		parentPath: []string{},
		current:    tl.root,
		tckData:    tckData,
	})

	// Store the data in the timeline.
	tl.mux.Lock()
	defer tl.mux.Unlock()
	defer func() {
		tl.currentTick++
	}()
	tl.tickToRecord[tl.currentTick] = tckData
	tl.storage += tckData.size

	// Clean-up the timeline if necessary.
	if tl.storage < storage.Global().Available() && len(tl.tickToRecord) < maxNumberOfSteps {
		return nil
	}
	tl.cleanupStorage()
	return nil
}

func (tl *Timeline) adjustDisplayTick(tick uint64) uint64 {
	if tick < tl.oldestTick {
		// Adjust to the oldest tick.
		return tl.oldestTick
	}
	if tick < tl.currentTick {
		// Valid value between the oldest tick and the current tick.
		return tick
	}
	if tl.currentTick == 0 {
		// No data has been stored yet.
		return 0
	}
	// Adjust to the latest tick.
	return tl.currentTick - 1
}

func keys(d map[string]*nodeRecord) []string {
	r := []string{}
	for k := range d {
		r = append(r, k)
	}
	return r
}

// MarshalData serializes the data given the current tick being displayed.
func (tl *Timeline) MarshalData(data *treepb.NodeData, path []string) {
	tl.mux.Lock()
	defer tl.mux.Unlock()

	displayTick := tl.adjustDisplayTick(tl.displayTick)
	tckData := tl.tickToRecord[displayTick]
	if tckData == nil {
		data.Error = fmt.Sprintf("data for tick %d does not exist", displayTick)
		return
	}
	current := tckData.root
	if current.err != nil {
		data.Error = current.err.Error()
		return
	}
	for _, p := range path {
		child, ok := current.toData[p]
		if !ok {
			data.Error = fmt.Sprintf("child %q in path %v cannot be found in the timeline. Available children are: %v", p, path, keys(current.toData))
			return
		}
		if child.err != nil {
			data.Error = child.err.Error()
			return
		}
		current = child
	}
	proto.Merge(data, current.data)
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

	return tl.displayTick >= tl.currentTick
}

// Close the timeline and release all the memory.
func (tl *Timeline) Close() error {
	for k := range tl.tickToRecord {
		delete(tl.tickToRecord, k)
	}
	storage.Global().Unregister()
	return nil
}
