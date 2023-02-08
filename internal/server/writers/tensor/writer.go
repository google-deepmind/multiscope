package tensor

import (
	"fmt"
	"sync"

	"multiscope/internal/mime"
	"multiscope/internal/server/core"
	"multiscope/internal/server/treeservice"
	"multiscope/internal/server/writers/base"
	"multiscope/internal/server/writers/tensor/tnrdr"
	pb "multiscope/protos/tensor_go_proto"
	treepb "multiscope/protos/tree_go_proto"

	"go.uber.org/multierr"
)

type (
	updater interface {
		forwardActive(path *core.Path)

		reset() error

		update(updateIndex uint, t sTensor) error

		forceUpdate(updateIndex uint, t sTensor) error

		lastUpdateIndex() uint
	}

	// Writer writes tensors to Multiscope.
	Writer struct {
		*base.Group
		mux sync.Mutex

		m           tnrdr.Metrics
		tensor      sTensor
		updts       []updater
		updateIndex uint

		state    treeservice.State
		timeline *tlNode
	}
)

// Names of the children to display additional data about the tensors.
const (
	NodeNameImage        = "tensor"
	NodeNameBitPlane     = "bits"
	NodeNameInfo         = "info"
	NodeNameMinMax       = "minmax"
	NodeNameNorms        = "norms"
	NodeNameDistribution = "distribution"
	NodeNameHistory      = "history"
	NodeNameRGBImage     = "rgb"
)

var _ core.Node = (*Writer)(nil)

// NewWriter returns a new writer to write Tensors.
func NewWriter() (*Writer, error) {
	w := &Writer{
		Group: base.NewGroup(mime.MultiscopeTensorGroup),
	}
	w.timeline = newAdapter(w)
	w.updts = []updater{
		newMetrics(w),
		newImageUpdater(w),
		newBitPlaneUpdater(w),
		newRGBUpdater(w),
		newDistribution(w),
		newTensorUpdater(w),
		newHistory(w),
	}
	w.m.Reset()
	return w, nil
}

func (w *Writer) addToTree(state treeservice.State, path *treepb.NodePath) (*core.Path, error) {
	writerPath, err := core.SetNodeAt(state.Root(), path, w)
	if err != nil {
		return nil, err
	}
	w.ForwardActive(state, writerPath)
	return writerPath, nil
}

// ForwardActive forwards activation up from the children to the writer.
func (w *Writer) ForwardActive(state treeservice.State, path *core.Path) {
	w.state = state
	for _, updt := range w.updts {
		updt.forwardActive(path)
	}
}

func (w *Writer) write(pbt *pb.Tensor) error {
	tns, err := protoToTensor(pbt)
	if err != nil {
		return fmt.Errorf("cannot deserialize the tensor: %v", err)
	}
	return w.Write(tns)
}

func (w *Writer) reset() (err error) {
	w.mux.Lock()
	defer w.mux.Unlock()
	w.tensor = &sliceTensor[float32]{}
	w.m.Reset()
	for _, updt := range w.updts {
		err = multierr.Append(err, updt.reset())
	}
	return
}

// Write a tensor to the writer.
func (w *Writer) Write(tns sTensor) (err error) {
	w.mux.Lock()
	defer w.mux.Unlock()
	w.updateIndex++

	w.tensor = tns
	w.m.Update(w.tensor.ValuesF32())
	for _, updt := range w.updts {
		err = multierr.Append(err, updt.update(w.updateIndex, w.tensor))
	}
	return
}

func (w *Writer) forceUpdate() (err error) {
	for _, updt := range w.updts {
		if updt.lastUpdateIndex() != w.updateIndex {
			err = multierr.Append(err, updt.forceUpdate(w.updateIndex, w.tensor))
		}
	}
	return
}

// Timeline returns a node to serialize in the timeline.
func (w *Writer) Timeline() core.Node {
	return w.timeline
}

// MarshalData writes the tensor protobuf into a data node.
func (w *Writer) MarshalData(data *treepb.NodeData, path []string, lastTick uint32) {
	if w.tensor == nil {
		return
	}
	if len(path) == 0 {
		return
	}
	w.Group.MarshalData(data, path, lastTick)
}
