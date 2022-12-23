package ticker

import (
	"log"
	"multiscope/internal/control"
	pb "multiscope/protos/ticker_go_proto"
	"time"

	"go.uber.org/atomic"
)

type (
	playerNoPauseControl struct {
		player *Player

		tline *control.Control

		toTLine   chan bool
		terminate atomic.Bool
	}
)

func newNoPauseControl(player *Player) playerControl {
	c := &playerNoPauseControl{
		player: player,
		tline:  control.New(),

		toTLine: make(chan bool),
	}
	go c.playerLoop()
	return c
}

func (c *playerNoPauseControl) playerLoop() {
	for !c.terminate.Load() {
		c.tline.WaitNextStep()
		if c.player.tline.IsLastTickDisplayed() {
			// The player is synchronized on the main loop:
			// slow down to avoid 100% usage.
			select {
			case <-c.toTLine:
			case <-time.After(500 * time.Millisecond):
			}
		} else {
			if err := c.player.tline.SetTickView(&pb.SetTickView{TickCommand: &pb.SetTickView_Offset{
				Offset: 1,
			}}); err != nil {
				log.Printf("timeline Go routine cannot set the ticker view: %v", err)
			}
		}
	}
}

func (c *playerNoPauseControl) setPeriod(period *pb.SetPeriod) error {
	c.tline.SetPeriod(time.Duration(period.PeriodMs) * time.Millisecond)
	return nil
}

func (c *playerNoPauseControl) processCommand(cmd pb.Command) error {
	if cmd != pb.Command_RUN {
		c.player.tline.SetTickView(&pb.SetTickView{
			TickCommand: &pb.SetTickView_ToDisplay{
				ToDisplay: c.player.tline.CurrentTick() - 1,
			},
		})
	}
	return c.tline.ProcessCommand(cmd)
}

func (c *playerNoPauseControl) pause() {
	c.tline.Pause()
}

func (c *playerNoPauseControl) mainNextStep() {
	select {
	case c.toTLine <- true:
	default:
	}
}

func (c *playerNoPauseControl) close() {
	c.terminate.Store(true)
}
