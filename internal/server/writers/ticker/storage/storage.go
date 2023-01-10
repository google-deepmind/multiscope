// Package storage provides a global registry for storage for the server.
package storage

import (
	"log"
	"sync/atomic"
)

// Storage maintains a global memory resource to be shared amongst multiple storages.
type Storage struct {
	totalMaxStorageSize uint64
	nStorage            int64
}

var storage = &Storage{}

func init() {
	storage.SetSize(2e9)
}

// Global returns the global storage to use within a process.
func Global() *Storage {
	return storage
}

// SetSize sets the maximum size of the storage for all timelines.
func (s *Storage) SetSize(size uint64) {
	atomic.StoreUint64(&s.totalMaxStorageSize, size)
}

// Register registers a new storage to allocate its share of the total storage.
func (s *Storage) Register() {
	atomic.AddInt64(&s.nStorage, 1)
}

// Unregister removes a storage from the shared storage.
func (s *Storage) Unregister() {
	nStorage := atomic.AddInt64(&s.nStorage, -1)
	// We panic if nStorage becomes negative, that is Unregister was called more than Register.
	if nStorage < 0 {
		log.Panic("Storage.Unregister() has been called more than Storage.Register()")
	}
}

// Available returns how much storage is available for a given timeline.
func (s *Storage) Available() uint64 {
	nStorage := atomic.LoadInt64(&s.nStorage)
	totalMaxStorageSize := atomic.LoadUint64(&s.totalMaxStorageSize)
	if nStorage <= 1 {
		return totalMaxStorageSize
	}
	return totalMaxStorageSize / uint64(nStorage)
}
