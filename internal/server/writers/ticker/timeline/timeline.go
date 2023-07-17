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

// Package timeline implements the timeline of the player.
package timeline

import (
	"fmt"
	"math"
	"sync"

	"multiscope/internal/server/core/timeline"
	"multiscope/internal/server/writers/ticker/storage"
	"multiscope/internal/server/writers/ticker/timedb"
	pb "multiscope/protos/ticker_go_proto"

	"github.com/pkg/errors"
)

// Timeline stores data for a sub-tree for all ticks.
type Timeline struct {
	mux sync.Mutex

	currentTick int64
	displayTick int64
}

// New returns a new time given a node in the tree.
func New() *Timeline {
	return &Timeline{
		currentTick: 0,
		displayTick: math.MaxInt64,
	}
}

// MarshalDisplay sets the timeline information.
func (tl *Timeline) MarshalDisplay(db *timedb.TimeDB) *pb.TimeLine {
	tl.mux.Lock()
	defer tl.mux.Unlock()

	if db.NumRecords() == 0 {
		return nil
	}
	currentStorage := db.StorageSize()
	maxStorage := storage.Global().Available()
	storageCapacity := fmt.Sprintf("%.e/%.e (%d%%)",
		float64(currentStorage),
		float64(maxStorage),
		int(float64(currentStorage)/float64(maxStorage)*100))
	return &pb.TimeLine{
		DisplayTick:     tl.adjustDisplayTick(db, tl.displayTick),
		OldestTick:      db.Oldest(),
		HistoryLength:   int64(db.NumRecords()),
		StorageCapacity: storageCapacity,
	}
}

func (tl *Timeline) setDisplayTick(db *timedb.TimeDB, displayTick int64) {
	tl.mux.Lock()
	defer tl.mux.Unlock()

	tl.displayTick = displayTick
}

func (tl *Timeline) offsetDisplayTick(db *timedb.TimeDB, offset int64) {
	tl.mux.Lock()
	defer tl.mux.Unlock()

	tl.displayTick = tl.adjustDisplayTick(db, tl.displayTick)
	tl.displayTick += offset
}

// SetTickView set the view to display.
func (tl *Timeline) SetTickView(db *timedb.TimeDB, view *pb.SetTickView) error {
	switch cmd := view.TickCommand.(type) {
	case *pb.SetTickView_ToDisplay:
		tl.setDisplayTick(db, cmd.ToDisplay)
	case *pb.SetTickView_Offset:
		tl.offsetDisplayTick(db, cmd.Offset)
	default:
		return errors.Errorf("tick command not supported: %T", cmd)
	}
	return nil
}

// Reset the timeline by removing all data.
func (tl *Timeline) Reset() error {
	tl.mux.Lock()
	defer tl.mux.Unlock()
	tl.currentTick = 0
	tl.displayTick = math.MaxInt64

	return nil
}

// Store the data for all children.
func (tl *Timeline) Store(db *timedb.TimeDB, root timeline.Marshaler) error {
	db.Store(tl.currentTick, root)

	tl.mux.Lock()
	defer tl.mux.Unlock()
	tl.currentTick++

	return nil
}

func (tl *Timeline) adjustDisplayTick(db *timedb.TimeDB, tick int64) (out int64) {
	defer func() {
		if out < 0 {
			out = 0
		}
	}()

	oldest := db.Oldest()
	if tick < db.Oldest() {
		// Adjust to the oldest tick.
		return oldest
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

// DisplayTick a record at the current tick being displayed.
func (tl *Timeline) DisplayTick(db *timedb.TimeDB) int64 {
	tl.mux.Lock()
	defer tl.mux.Unlock()

	return tl.adjustDisplayTick(db, tl.displayTick)
}

// CurrentTick returns the next frame to store.
func (tl *Timeline) CurrentTick() int64 {
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
