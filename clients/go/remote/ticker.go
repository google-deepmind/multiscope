package remote

import (
	"context"
	"time"

	"multiscope/internal/control"
	"multiscope/internal/period"
	"multiscope/internal/server/events"
	pb "multiscope/protos/ticker_go_proto"
	pbgrpc "multiscope/protos/ticker_go_proto"
	treepb "multiscope/protos/tree_go_proto"

	"github.com/pkg/errors"
	"go.uber.org/multierr"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
	"google.golang.org/protobuf/types/known/durationpb"
)

type (
	// Caller gets called by the Ticker when it ticks.
	Caller func() error

	// Subscriber registers callers so that they can be called back to push their data to Multiscope.
	Subscriber interface {
		Subscribe(c Caller)
	}

	// Ticker is a directory node able to perform additional synchronization on its children.
	Ticker struct {
		*ClientNode
		clt    pbgrpc.TickersClient
		ticker *pb.Ticker

		tick    int64
		callers []Caller

		totalPeriod      period.Period
		experimentPeriod period.Period
		callbackPeriod   period.Period

		control *control.Control
	}
)

const measureStepSize = .99

// NewTicker creates a new ticker node in tree.
func NewTicker(clt *Client, name string, parent Path) (*Ticker, error) {
	t := &Ticker{
		clt:  pbgrpc.NewTickersClient(clt.Connection()),
		tick: -1,

		control: control.New(),
	}
	t.totalPeriod.HistoryTrace = measureStepSize
	t.experimentPeriod.HistoryTrace = measureStepSize
	t.callbackPeriod.HistoryTrace = measureStepSize
	ctx := context.Background()
	path := clt.toChildPath(name, parent)
	rep, err := t.clt.NewTicker(ctx, &pb.NewTickerRequest{
		Path: path.NodePath(),
	})
	if err != nil {
		return nil, err
	}
	t.ticker = rep.GetTicker()
	if t.ticker == nil {
		return nil, errors.New("server has returned a nil ticker")
	}
	t.ClientNode = NewClientNode(clt, toPath(t.ticker))
	if err := clt.Display().DisplayIfDefault(t.Path()); err != nil {
		return nil, err
	}
	if _, err := clt.EventsManager().Register(t.Path(), func() events.Callback {
		return events.CallbackF(t.processEvent)
	}); err != nil {
		return nil, err
	}

	// Subscribe builtin callbacks.
	t.Subscribe(t.writeData)
	return t, nil
}

var tickerActionURL = string(proto.MessageName(&pb.TickerAction{}))

func (t *Ticker) processEvent(event *treepb.Event) (bool, error) {
	payload := event.GetPayload()
	if payload == nil {
		return false, nil
	}
	if events.CoreURL(payload.GetTypeUrl()) != tickerActionURL {
		return false, nil
	}
	action := pb.TickerAction{}
	if err := anypb.UnmarshalTo(payload, &action, proto.UnmarshalOptions{}); err != nil {
		return false, err
	}
	var err error
	switch a := action.Action.(type) {
	case *pb.TickerAction_SetPeriod:
		period := time.Duration(a.SetPeriod.PeriodMs) * time.Millisecond
		t.control.SetPeriod(period)
	case *pb.TickerAction_Command:
		err = t.control.ProcessCommand(a.Command)
	default:
		err = errors.Errorf("command not supported: %q", a)
	}
	return true, err
}

func (t *Ticker) callCallers() error {
	t.callbackPeriod.Start()
	defer t.callbackPeriod.Sample()

	var err error
	for _, caller := range t.callers {
		err = multierr.Append(err, caller())
	}
	return err
}

func (t *Ticker) writeData() error {
	if !t.ShouldWrite() {
		return nil
	}
	total := t.totalPeriod.Average()
	exp := t.experimentPeriod.Average()
	callback := t.callbackPeriod.Average()
	idle := total - exp - callback
	_, err := t.clt.WriteTicker(context.Background(), &pb.WriteTickerRequest{
		Ticker: t.ticker,
		Data: &pb.TickerData{
			Tick: t.tick,
			Periods: &pb.TickerData_Periods{
				Total:      durationpb.New(total),
				Experiment: durationpb.New(exp),
				Callbacks:  durationpb.New(callback),
				Idle:       durationpb.New(idle),
			},
		},
	})
	return err
}

// Pause the ticker at the next step.
func (t *Ticker) Pause() {
	t.control.Pause()
}

// Tick the remote ticker.
func (t *Ticker) Tick() error {
	t.experimentPeriod.Sample()
	defer t.experimentPeriod.Start()

	t.control.WaitNextStep()

	t.tick++
	t.totalPeriod.Sample()

	err := t.callCallers()
	return err
}

// Subscribe caller to be called everytime Tick is called.
func (t *Ticker) Subscribe(c Caller) {
	t.callers = append(t.callers, c)
}

// CurrentTick returns the current tick of the clock.
func (t *Ticker) CurrentTick() int64 {
	return t.tick
}

// SetPeriod sets the period of the ticker.
func (t *Ticker) SetPeriod(p time.Duration) error {
	t.control.SetPeriod(p)
	return nil
}
