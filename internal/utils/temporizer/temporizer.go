// Copyright 2023 Google LLC
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

// Package temporizer implements a temporizer to drop too frequent calls.
package temporizer

import (
	"sync"
	"time"
)

// Temporizer drops set calls if they are coming too frequently.
type Temporizer[T comparable] struct {
	mut      sync.Mutex
	callback func(v T)

	lastCall        time.Time
	lastValue       T
	currentDuration time.Duration

	pending         T
	pendingDuration time.Duration
	pendingActive   bool
}

// NewTemporizer returns a Temporizer given a setter function to call only once in a while.
// The structure is thread-safe, meaning that Set can be called from multiple Go routine.
func NewTemporizer[T comparable](cb func(v T)) *Temporizer[T] {
	return &Temporizer[T]{
		callback: cb,
		lastCall: time.Now(),
	}
}

func (tp *Temporizer[T]) callCallback(v T, dr time.Duration) {
	tp.callback(v)
	tp.lastCall = time.Now()
	tp.lastValue = v
	tp.currentDuration = dr
}

func (tp *Temporizer[T]) sleepTime() time.Duration {
	tp.mut.Lock()
	defer tp.mut.Unlock()

	return tp.currentDuration - time.Since(tp.lastCall)
}

func (tp *Temporizer[T]) backgroundRoutine() {
	sleepTime := tp.sleepTime()
	for tp.sleepTime() > 0 {
		time.Sleep(sleepTime)
		sleepTime = tp.sleepTime()
	}

	tp.mut.Lock()
	defer tp.mut.Unlock()

	tp.callCallback(tp.pending, tp.pendingDuration)
	tp.pendingActive = false
}

// Set a new value. The callback will be called only if it has not been called for a while.
func (tp *Temporizer[T]) Set(v T, dr time.Duration) {
	tp.mut.Lock()
	defer tp.mut.Unlock()

	if v == tp.lastValue {
		return
	}
	if time.Since(tp.lastCall) > tp.currentDuration {
		// Last call was a long time ago:
		// we can just call the callback.
		tp.callCallback(v, dr)
		return
	}
	// This is too soon. Update the pending value.
	tp.pending = v
	if dr > tp.pendingDuration {
		tp.pendingDuration = dr
	}
	if tp.pendingActive {
		// There is already a Go routine that will call
		// the callback. We are done.
		return
	}
	// No pending Go routine: starts one.
	tp.pendingActive = true
	go tp.backgroundRoutine()
}

// SetDuration sets the duration for the next update value.
func (tp *Temporizer[T]) SetDuration(dr time.Duration) {
	tp.mut.Lock()
	defer tp.mut.Unlock()

	tp.currentDuration = dr
	tp.lastCall = time.Now()
}
