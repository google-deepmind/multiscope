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

// Package control processes events to pause, step, and run a process.
package control

import (
	"multiscope/internal/period"
	pb "multiscope/protos/ticker_go_proto"
	"sync/atomic"
	"time"

	"github.com/pkg/errors"
)

// Control a process by processing events from the UI.
type Control struct {
	outPeriod period.Period
	pause     atomic.Bool
	period    atomic.Int64

	pauseNextStep chan bool
}

const channelBuffer = 10

// New returns a new instance of control.
func New() *Control {
	return &Control{
		pauseNextStep: make(chan bool, channelBuffer),
	}
}

// ProcessCommand processes a command.
// This function is typically called from the Go routine managing an event.
func (c *Control) ProcessCommand(cmd pb.Command) error {
	switch cmd.Number() {
	case pb.Command_CMD_STEP.Number():
		c.pauseNextStep <- true
	case pb.Command_CMD_PAUSE.Number():
		c.pause.Store(true)
	case pb.Command_CMD_RUN.Number():
		c.pauseNextStep <- false
	default:
		return errors.Errorf("command not supported: %q", cmd.String())
	}
	return nil
}

// IsOnPause returns true if the controller is on pause.
func (c *Control) IsOnPause() bool {
	return c.pause.Load()
}

// Pause the main user thread.
func (c *Control) Pause() {
	c.pause.Store(true)
}

func (c *Control) drainPauseNextStep() {
	for {
		select {
		case pause := <-c.pauseNextStep:
			c.pause.Store(pause)
		default:
			return
		}
	}
}

// WaitNextStep waits the next step if the control is on pause.
// If not, then the function returns immedialtly.
func (c *Control) WaitNextStep() {
	// Pause if we have to.
	if c.pause.Load() {
		c.pause.Store(<-c.pauseNextStep)
	}
	c.drainPauseNextStep()

	outPeriod := c.outPeriod.Sample()
	wantPeriod := time.Duration(c.period.Load())
	if outPeriod < wantPeriod {
		time.Sleep(wantPeriod - outPeriod)
	}
	c.outPeriod.Start()
}

// SetPeriod sets the period for waiting.
func (c *Control) SetPeriod(period time.Duration) {
	c.period.Store(int64(period))
}
