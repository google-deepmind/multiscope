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
	// Tensor defines a float64 tensor.
	Tensor interface {
		// Shape returns the shape of the Tensor. The product of the dimensions matches the length of the slice returned by Value.
		Shape() []int

		// Value returns a slice stored by the Tensor.
		//
		// The data may not be copied. The slice needs to be copied to own it (e.g. for storage).
		// Not completely sure this is necessary, but it seems safer and does not add any significant cost.
		Values() []float32
	}

	// WithUInt8 is a tensor to which the values can be accessed as uint8s.
	WithUInt8 interface {
		// ValueUInt8 returns the value of the tensor as uint8s.
		ValuesUInt8() []uint8
	}

	updaters interface {
		forwardActive(path *core.Path)

		reset() error

		update(t Tensor) error
	}

	// Writer writes tensors to Multiscope.
	Writer struct {
		*base.Group
		mux sync.Mutex

		m      tnrdr.Metrics
		tensor Tensor
		updts  []updaters

		state treeservice.State
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
	w.updts = []updaters{
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
	tns, err := ProtoToTensor(pbt)
	if err != nil {
		return fmt.Errorf("cannot deserialize the tensor: %v", err)
	}
	return w.Write(tns)
}

func (w *Writer) reset() (err error) {
	w.mux.Lock()
	defer w.mux.Unlock()
	w.tensor = &PBTensor{}
	w.m.Reset()
	for _, updt := range w.updts {
		err = multierr.Append(err, updt.reset())
	}
	return
}

// Write a tensor to the writer.
func (w *Writer) Write(tns Tensor) (err error) {
	w.mux.Lock()
	defer w.mux.Unlock()
	w.tensor = tns
	w.m.Update(w.tensor.Values())
	for _, updt := range w.updts {
		err = multierr.Append(err, updt.update(w.tensor))
	}
	return
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
