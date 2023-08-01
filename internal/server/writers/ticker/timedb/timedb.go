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

// Package timedb provides a database to save the current state of tree indexed by a time step.
package timedb

import (
	"fmt"
	"multiscope/internal/server/core"
	"multiscope/internal/server/core/timeline"
	"multiscope/internal/server/writers/ticker/storage"
	treepb "multiscope/protos/tree_go_proto"
	"runtime"
	"sync"

	"go.uber.org/atomic"
)

const maxNumberOfSteps = 1e6

type (
	storedRecord struct {
		rec  timeline.Marshaler
		size uint64
	}

	// TimeDB saves data indexed by time step.
	TimeDB struct {
		mut sync.RWMutex

		node core.Node

		storage      uint64
		tickToRecord map[int64]storedRecord
		oldestTick   int64

		disableCleanup atomic.Bool
	}
)

// New returns a new database to store data indexed by time.
func New(node core.Node) *TimeDB {
	storage.Global().Register()
	return &TimeDB{
		node:         node,
		tickToRecord: make(map[int64]storedRecord),
	}
}

// Clone returns a static clone of the DB, meaning that it is not
// valid to store additional records in a clone.
func (db *TimeDB) Clone() *TimeDB {
	db.mut.RLock()
	defer db.mut.RUnlock()

	cl := &TimeDB{
		storage:      db.storage,
		tickToRecord: make(map[int64]storedRecord),
		oldestTick:   db.oldestTick,
	}
	for tick, rec := range db.tickToRecord {
		cl.tickToRecord[tick] = rec
	}
	return cl
}

// Oldest returns the oldest tick.
func (db *TimeDB) Oldest() int64 {
	db.mut.RLock()
	defer db.mut.RUnlock()

	return db.oldestTick
}

// MarshalData for a given tick to display.
func (db *TimeDB) MarshalData(toDisplay int64, data *treepb.NodeData, path []string, lastTick uint32) {
	rec := db.fetch(toDisplay)
	if rec == nil {
		data.Error = fmt.Sprintf("tick %d has not been stored", toDisplay)
		return
	}
	rec.MarshalData(data, path, lastTick)
}

func (db *TimeDB) fetch(tick int64) timeline.Marshaler {
	db.mut.RLock()
	defer db.mut.RUnlock()

	return db.tickToRecord[tick].rec
}

// Reset the time database.
func (db *TimeDB) Reset() {
	db.mut.Lock()
	defer db.mut.Unlock()

	for tick := range db.tickToRecord {
		delete(db.tickToRecord, tick)
	}
	db.storage = 0
}

// Store a record given a tick index.
func (db *TimeDB) Store(tick int64, rec timeline.Marshaler) {
	size := rec.StorageSize()

	db.mut.Lock()
	defer db.mut.Unlock()

	db.tickToRecord[tick] = storedRecord{rec: rec, size: size}
	db.storage += size

	// Clean-up the db if necessary.
	if len(db.tickToRecord) >= maxNumberOfSteps {
		db.cleanup()
	}
	if db.disableCleanup.Load() {
		return
	}
	if db.storage < storage.Global().Available() {
		return
	}
	db.cleanup()
}

// NumRecords returns the number of records being stored.
func (db *TimeDB) NumRecords() int {
	db.mut.RLock()
	defer db.mut.RUnlock()

	return len(db.tickToRecord)
}

// StorageSize returns the current memory size used by the database.
func (db *TimeDB) StorageSize() uint64 {
	db.mut.RLock()
	defer db.mut.RUnlock()

	return db.storage
}

// DisableCleanup disables the cleanup done because of the lack of storage.
func (db *TimeDB) DisableCleanup() {
	db.disableCleanup.Store(true)
}

// CleanupDisabled returns true if cleanup has been disabled.
func (db *TimeDB) CleanupDisabled() bool {
	return db.disableCleanup.Load()
}

// Cleanup removes old records that do not fit in storage.
func (db *TimeDB) cleanup() {
	const targetRatio = .9
	targetStorage := uint64(float32(storage.Global().Available()) * targetRatio)
	targetNumberOfSteps := int(float32(len(db.tickToRecord)) * targetRatio)
	for db.storage > targetStorage || len(db.tickToRecord) > targetNumberOfSteps {
		oldestRecord := db.tickToRecord[db.oldestTick]
		db.storage -= oldestRecord.size
		delete(db.tickToRecord, db.oldestTick)
		db.oldestTick++
	}
	runtime.GC()
}

// Close the timeline and release all the memory.
func (db *TimeDB) Close() error {
	db.mut.Lock()
	defer db.mut.Unlock()

	for k := range db.tickToRecord {
		delete(db.tickToRecord, k)
	}
	storage.Global().Unregister()
	return nil
}
