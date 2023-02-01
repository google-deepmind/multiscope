// Package timedb provides a database to save the current state of tree indexed by a time step.
package timedb

import (
	"multiscope/internal/server/core"
	"multiscope/internal/server/writers/ticker/storage"
	"runtime"
	"sync"
)

const maxNumberOfSteps = 1e6

type (
	storedRecord struct {
		rec  *Record
		size uint64
	}

	// TimeDB saves data indexed by time step.
	TimeDB struct {
		mut sync.RWMutex

		node core.Node

		storage      uint64
		tickToRecord map[int64]storedRecord
		oldestTick   int64
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

// Oldest returns the oldest tick.
func (db *TimeDB) Oldest() int64 {
	db.mut.RLock()
	defer db.mut.RUnlock()

	return db.oldestTick
}

// Fetch the root record for a given tick.
// The result can be nil if no record has been stored for the tick.
func (db *TimeDB) Fetch(tick int64) *Record {
	db.mut.RLock()
	defer db.mut.RUnlock()

	return db.tickToRecord[tick].rec
}

// Store a record given a tick index.
func (db *TimeDB) Store(tick int64) {
	rec := db.newContext().buildRecord()
	size := rec.computeSize()

	db.mut.Lock()
	defer db.mut.Unlock()

	db.tickToRecord[tick] = storedRecord{rec: rec, size: size}
	db.storage += size

	// Clean-up the db if necessary.
	if db.storage < storage.Global().Available() && len(db.tickToRecord) < maxNumberOfSteps {
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
