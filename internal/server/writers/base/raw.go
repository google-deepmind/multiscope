package base

import (
	"bytes"
	"fmt"
	"sync"

	"multiscope/internal/server/core"
	"multiscope/internal/server/treeservice"
	pb "multiscope/protos/tree_go_proto"

	"google.golang.org/protobuf/proto"
)

// RawWriter implements a leaf node to stream raw bytes.
type RawWriter struct {
	mux  sync.Mutex
	mime string
	data bytes.Buffer
	tick uint32
}

var _ core.Node = (*RawWriter)(nil)

// TODO(vikrantvarma): set size via API once implemented and delete this.
const (
	DefaultHeight = 300
	DefaultWidth  = 300
)

// NewRawWriter returns a new writer to stream raw bytes.
func NewRawWriter(mime string) *RawWriter {
	return &RawWriter{
		tick: 1,
		mime: mime,
	}
}

// Write sets the content of the leaf with p.
func (w *RawWriter) Write(p []byte) (n int, err error) {
	buf := w.LockBuffer()
	defer w.UnlockBuffer()

	buf.Reset()
	n, err = buf.Write(p)
	return
}

// HandleWrite handles a put data request for a RawWriter.
func (w *RawWriter) HandleWrite(msg proto.Message) error {
	data, ok := msg.(*pb.NodeData)
	if !ok {
		// Indicates a bug.
		return fmt.Errorf("internal error in RawWriter: data of unexpected type '%T'; expected NodeData", msg)
	}
	if data.GetRaw() == nil {
		return fmt.Errorf("`raw` must be set in NodeData for RawWriter")
	}
	_, err := w.Write(data.GetRaw())
	return err
}

// AddToTree adds the raw writer to tree in server state, at the given path.
func (w *RawWriter) AddToTree(state treeservice.State, path *pb.NodePath) (*core.Path, error) {
	return core.SetNodeAt(state.Root(), path, w)
}

// MIME returns the MIME type of the raw data.
func (w *RawWriter) MIME() string {
	return w.mime
}

// MarshalData writes the raw data into a NodeData protocol buffer.
func (w *RawWriter) MarshalData(d *pb.NodeData, path []string, lastTick uint32) {
	w.mux.Lock()
	defer w.mux.Unlock()

	d.Tick = w.tick
	d.Mime = w.mime
	if w.tick == lastTick {
		return
	}

	p := &pb.NodeData_Raw{}
	p.Raw = append([]byte{}, w.data.Bytes()...)
	d.Data = p
}

// LockBuffer locks the writer and returns the table to manipulate its data.
func (w *RawWriter) LockBuffer() *bytes.Buffer {
	w.mux.Lock()
	return &w.data
}

// UnlockBuffer unlocks the writer. The table returned by LockTable() should not be modified after this point.
func (w *RawWriter) UnlockBuffer() {
	w.tick++
	w.mux.Unlock()
}

func (w *RawWriter) String() string {
	return w.MIME()
}
