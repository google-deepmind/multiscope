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

	wait chan bool
}

const channelBuffer = 10

// New returns a new instance of control.
func New() *Control {
	return &Control{
		wait: make(chan bool, channelBuffer),
	}
}

// ProcessCommand processes a command.
// This function is typically called from the Go routine managing an event.
func (c *Control) ProcessCommand(cmd pb.Command) error {
	switch cmd.Number() {
	case pb.Command_STEP.Number():
		c.pause.Store(true)
		c.wait <- true
	case pb.Command_PAUSE.Number():
		c.pause.Store(true)
	case pb.Command_RUN.Number():
		if c.pause.Load() {
			c.pause.Store(false)
			c.wait <- true
		}
	default:
		return errors.Errorf("command not supported: %q", cmd.String())
	}
	return nil
}

// Pause the main user thread.
func (c *Control) Pause() {
	c.pause.Store(true)
}

// WaitNextStep waits the next step if the control is on pause.
// If not, then the function returns immedialtly.
func (c *Control) WaitNextStep() {
	if c.pause.Load() {
		<-c.wait
	}

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
