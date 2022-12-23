package ticker

import (
	"log"
	"multiscope/internal/control"
	pb "multiscope/protos/ticker_go_proto"
	"sync"
	"time"

	"go.uber.org/atomic"
)

type (
	playerSyncControl struct {
		terminate atomic.Bool
		player    *Player

		tline *control.Control

		toMain       chan bool
		mainStepDone sync.WaitGroup
	}
)

func newSyncControl(player *Player) playerControl {
	c := &playerSyncControl{
		player: player,
		tline:  control.New(),
		toMain: make(chan bool),
	}
	c.mainStepDone.Add(1)
	go c.playerLoop()
	return c
}

func (c *playerSyncControl) playerLoop() {
	for !c.terminate.Load() {
		c.tline.WaitNextStep()
		if c.player.tline.IsLastTickDisplayed() {
			if !c.tline.IsOnPause() {
				c.mainStepDone.Add(1)
				c.toMain <- true
				c.mainStepDone.Wait()
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

func (c *playerSyncControl) pause() {
	c.tline.Pause()
}

func (c *playerSyncControl) setPeriod(period *pb.SetPeriod) error {
	c.tline.SetPeriod(time.Duration(period.PeriodMs) * time.Millisecond)
	return nil
}

func (c *playerSyncControl) processCommand(cmd pb.Command) error {
	return c.tline.ProcessCommand(cmd)
}

func (c *playerSyncControl) mainNextStep() {
	c.mainStepDone.Done()
	<-c.toMain
}

func (c *playerSyncControl) close() {
	c.terminate.Store(true)
}
