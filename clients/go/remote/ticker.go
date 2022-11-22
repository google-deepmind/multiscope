package remote

import (
	"context"
	"sync"
	"time"

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
	pauseProtected struct {
		mut   sync.Mutex
		pause bool
	}

	// Caller gets called by the Ticker when it ticks.
	Caller func() error

	// Ticker is a directory node able to perform additional synchronization on its children.
	Ticker struct {
		*ClientNode
		clt    pbgrpc.TickersClient
		ticker *pb.Ticker

		period  time.Duration
		tick    int64
		callers []Caller

		totalPeriod      period.Period
		experimentPeriod period.Period
		callbackPeriod   period.Period

		toTick        chan func() error
		wait          chan bool
		pauseNextStep pauseProtected
	}
)

func (p *pauseProtected) set(pause bool) {
	p.mut.Lock()
	defer p.mut.Unlock()
	p.pause = pause
}

func (p *pauseProtected) get() bool {
	p.mut.Lock()
	defer p.mut.Unlock()
	return p.pause
}

const measureStepSize = .99

// NewTicker creates a new ticker node in tree.
func NewTicker(clt *Client, name string, parent Path) (*Ticker, error) {
	const channelBuffer = 10
	t := &Ticker{
		clt:    pbgrpc.NewTickersClient(clt.Connection()),
		tick:   -1,
		toTick: make(chan func() error, channelBuffer),
		wait:   make(chan bool, channelBuffer),
	}
	t.totalPeriod.HistoryTrace = measureStepSize
	t.experimentPeriod.HistoryTrace = measureStepSize
	t.callbackPeriod.HistoryTrace = measureStepSize
	ctx := context.Background()
	path := clt.toChildPath(name, parent)
	rep, err := t.clt.New(ctx, &pb.NewTickerRequest{
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
	t.Subscribe(t.processTickFunc)
	return t, nil
}

func (t *Ticker) processSetPeriod(msg *pb.TickerAction_SetPeriod) error {
	p := time.Duration(msg.GetPeriodMs())
	t.toTick <- func() error {
		return t.SetPeriod(p * time.Millisecond)
	}
	return nil
}

func (t *Ticker) processCommand(cmd pb.TickerAction_Command) error {
	switch cmd.Number() {
	case pb.TickerAction_STEP.Number():
		t.pauseNextStep.set(true)
		t.wait <- true
	case pb.TickerAction_PAUSE.Number():
		t.pauseNextStep.set(true)
	case pb.TickerAction_RUN.Number():
		t.pauseNextStep.set(false)
		t.wait <- true
	default:
		return errors.Errorf("command not supported: %q", cmd.String())
	}
	return nil
}

func (t *Ticker) processEvent(event *treepb.Event) (bool, error) {
	payload := event.GetPayload()
	if payload == nil {
		return false, nil
	}
	const typeURL = "multiscope.ticker.TickerAction"
	if events.CoreURL(payload.GetTypeUrl()) != typeURL {
		return false, nil
	}
	action := pb.TickerAction{}
	if err := anypb.UnmarshalTo(payload, &action, proto.UnmarshalOptions{}); err != nil {
		return false, err
	}
	var err error
	switch a := action.Action.(type) {
	case *pb.TickerAction_SetPeriod_:
		err = t.processSetPeriod(a.SetPeriod)
	case *pb.TickerAction_Command_:
		err = t.processCommand(a.Command)
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

func (t *Ticker) processTickFunc() (err error) {
	for {
		select {
		case f := <-t.toTick:
			err = multierr.Append(err, f())
		default:
			return
		}
	}
}

func (t *Ticker) writeData() error {
	if !t.ShouldWrite() {
		return nil
	}
	total := t.totalPeriod.Average()
	exp := t.experimentPeriod.Average()
	callback := t.callbackPeriod.Average()
	idle := total - exp - callback
	_, err := t.clt.Write(context.Background(), &pb.WriteRequest{
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
	t.pauseNextStep.set(true)
}

// Tick the remote ticker.
func (t *Ticker) Tick() error {
	expPeriod := t.experimentPeriod.Sample()
	defer t.experimentPeriod.Start()

	if expPeriod < t.period {
		time.Sleep(t.period - expPeriod)
	}

	if t.pauseNextStep.get() {
		<-t.wait
	}

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
	t.period = p
	return nil
}
