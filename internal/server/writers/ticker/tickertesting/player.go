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

package tickertesting

import (
	"context"
	stderr "errors"
	"fmt"
	"multiscope/internal/grpc/client"
	pb "multiscope/protos/ticker_go_proto"
	treepb "multiscope/protos/tree_go_proto"
	"time"

	"github.com/pkg/errors"
)

type displayNotModifiedError struct {
	err error
}

func (e displayNotModifiedError) Unwrap() error { return e.err }

func (e displayNotModifiedError) Error() string { return e.err.Error() }

func errTickerDisplayNotModified() error {
	return displayNotModifiedError{
		err: errors.New("the ticker action did not modify the display"),
	}
}

func queryDisplayTick(clt client.Client, tickerPath []string) (int64, error) {
	ctx := context.Background()
	data, err := client.NodesData(ctx, clt, []*treepb.Node{{
		Path: &treepb.NodePath{
			Path: tickerPath,
		}}})
	if err != nil {
		return 0, err
	}
	display := &pb.PlayerInfo{}
	if err := client.ToProto(data[0], display); err != nil {
		return 0, err
	}
	if display.Timeline == nil {
		return 0, errors.Errorf("the ticker did not return a timeline")
	}
	return display.Timeline.DisplayTick, nil
}

func sendEventWaitForUpdate(clt client.Client, tickerPath []string, action *pb.PlayerAction) error {
	old, err := queryDisplayTick(clt, tickerPath)
	if err != nil {
		return err
	}
	ctx := context.Background()
	if err := client.SendEvent(ctx, clt, tickerPath, action); err != nil {
		return err
	}
	current := old
	deadline := time.Now().Add(10 * time.Second)
	for old == current {
		time.Sleep(10 * time.Millisecond)
		current, err = queryDisplayTick(clt, tickerPath)
		if err != nil {
			return err
		}
		if time.Now().After(deadline) {
			return errTickerDisplayNotModified()
		}
	}
	return nil
}

// SendSetDisplayTick sends an event to a ticker to set the tick being displayed.
func SendSetDisplayTick(clt client.Client, tickerPath []string, tick int64) error {
	return sendEventWaitForUpdate(clt, tickerPath, &pb.PlayerAction{
		Action: &pb.PlayerAction_TickView{
			TickView: &pb.SetTickView{
				TickCommand: &pb.SetTickView_ToDisplay{
					ToDisplay: tick,
				},
			},
		},
	})
}

func sendOffsetDisplayTick(clt client.Client, tickerPath []string, delta int64) error {
	return sendEventWaitForUpdate(clt, tickerPath, &pb.PlayerAction{
		Action: &pb.PlayerAction_TickView{
			TickView: &pb.SetTickView{
				TickCommand: &pb.SetTickView_Offset{
					Offset: delta,
				},
			},
		},
	})
}

func checkDisplayedValues(clt client.Client, nodes []*treepb.Node, tick int) error {
	ctx := context.Background()
	data, err := client.NodesData(ctx, clt, nodes)
	if err != nil {
		return err
	}
	htmlText, err := client.ToRaw(data[0])
	if err != nil {
		return err
	}
	htmlWant := fmt.Sprintf("html:%d", tick)
	if string(htmlText) != htmlWant {
		return fmt.Errorf("got %s, want %s", string(htmlText), htmlWant)
	}
	cssText, err := client.ToRaw(data[1])
	if err != nil {
		return err
	}
	cssWant := fmt.Sprintf("css:%d", tick)
	if string(cssText) != cssWant {
		return fmt.Errorf("got %s, want %s", string(cssText), cssWant)
	}
	return nil
}

// CheckPlayerTimeline01 checks the timeline by sending events to the player.
func CheckPlayerTimeline01(clt client.Client, tickerPath, writerPath []string) error {
	// Get the nodes given the path.
	nodes, err := client.PathToNodes(context.Background(), clt,
		append(append([]string{}, writerPath...), "html"),
		append(append([]string{}, writerPath...), "css"))
	if err != nil {
		return fmt.Errorf("cannot get paths to nodes: %w", err)
	}
	// Check the data using SetDisplayTick
	for i := 0; i < Ticker01NumTicks; i++ {
		displayTick := int64(i)
		if err := SendSetDisplayTick(clt, tickerPath, displayTick); err != nil {
			return fmt.Errorf("cannot set the clock display to %d: %w", displayTick, err)
		}
		if err := checkDisplayedValues(clt, nodes, i); err != nil {
			return fmt.Errorf("the wrong data is being displayed: %w", err)
		}
	}

	// Check the data using OffsetDisplayTick.
	if err := SendSetDisplayTick(clt, tickerPath, 0); err != nil {
		return fmt.Errorf("cannot set display tick: %w", err)
	}
	for i := 0; i < Ticker01NumTicks; i++ {
		if err := checkDisplayedValues(clt, nodes, i); err != nil {
			return fmt.Errorf("OffsetDisplayTick error: %w", err)
		}
		err := sendOffsetDisplayTick(clt, tickerPath, 1)
		if i < Ticker01NumTicks-1 && err != nil {
			return fmt.Errorf("cannot send offset at iteration %d: %w", i, err)
		}
		if i == Ticker01NumTicks && !stderr.As(err, &displayNotModifiedError{}) {
			return fmt.Errorf("this is the last tick: got error %v, want error %T", err, &displayNotModifiedError{})
		}
	}
	if err := checkDisplayedValues(clt, nodes, Ticker01NumTicks-1); err != nil {
		return fmt.Errorf("OffsetDisplayTick error: %w", err)
	}
	if err := SendSetDisplayTick(clt, tickerPath, 0); err != nil {
		return fmt.Errorf("SetDisplayTick error: %w", err)
	}
	// Check what we get when offsetting outside the range.
	if err := sendOffsetDisplayTick(clt, tickerPath, 100); err != nil {
		return fmt.Errorf("OffsetDisplayTick error: %w", err)
	}
	if err := checkDisplayedValues(clt, nodes, 9); err != nil {
		return fmt.Errorf("outside range positive displayTick: %w", err)
	}
	// We always use the last step if displayTick is lower than 0.
	if err := sendOffsetDisplayTick(clt, tickerPath, -100); err != nil {
		return fmt.Errorf("cannot send an offset of -100: %w", err)
	}
	if err := checkDisplayedValues(clt, nodes, 0); err != nil {
		return fmt.Errorf("outside range negative displayTick: %w", err)
	}
	// Put back the player in running mode for further testing.
	if err := client.SendEvent(context.Background(), clt, tickerPath, &pb.PlayerAction{
		Action: &pb.PlayerAction_Command{Command: pb.Command_CMD_RUN},
	}); err != nil {
		return errors.Errorf("cannot send a run event to the player: %v", err)
	}
	return nil
}

// CheckPlayerTimelineAfterReset checks the timeline it has been reset.
func CheckPlayerTimelineAfterReset(clt client.Client, tickerPath, writerPath []string) error {
	// Get the nodes given the path.
	nodes, err := client.PathToNodes(context.Background(), clt,
		writerPath,
		append(append([]string{}, writerPath...), "html"),
		append(append([]string{}, writerPath...), "css"))
	if err != nil {
		return fmt.Errorf("cannot get paths to nodes: %v", err)
	}
	ctx := context.Background()
	data, err := client.NodesData(ctx, clt, nodes)
	if err != nil {
		return err
	}
	if _, err := client.ToRaw(data[1]); err == nil {
		return errors.Errorf("want a node error but got nil")
	}
	return nil
}
