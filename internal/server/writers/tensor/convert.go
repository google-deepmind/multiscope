package tensor

import (
	"fmt"
	"unsafe"

	pb "multiscope/protos/tensor_go_proto"
)

var floatSize = int(unsafe.Sizeof(float32(1)))

// PBTensor is a deserialized tensor from a proto.
type PBTensor struct {
	shape []int
	value []float32
	// Keep a reference to the storage so that it is not garbage collected.
	storage *pb.Tensor
}

// Shape returns the shape of the tensor.
func (p *PBTensor) Shape() []int {
	return p.shape
}

// Values returns the value of the tensor.
func (p *PBTensor) Values() []float32 {
	return p.value
}

func intShape(dims []*pb.Shape_Dim) []int {
	shapeint := make([]int, len(dims))
	for i, dim := range dims {
		shapeint[i] = int(dim.Size)
	}
	return shapeint
}

// ProtoToTensor returns a new tensor with the memory mapped to the proto memory.
func ProtoToTensor(p *pb.Tensor) (*PBTensor, error) {
	if p.GetDtype() != pb.DataType_DT_FLOAT32 {
		return nil, fmt.Errorf("invalid dtype: got %v but wants %v", p.GetDtype(), pb.DataType_DT_FLOAT32)
	}
	storage := p.Content
	if len(storage)%floatSize != 0 {
		return nil, fmt.Errorf("invalid value length: %d is not divisible by the size of a float (%d)", len(storage), floatSize)
	}
	var shapeLength int
	shape := intShape(p.Shape.Dim)
	if len(shape) > 0 {
		shapeLength = 1
	}
	for _, dim := range shape {
		shapeLength *= dim
	}
	length := len(storage) / floatSize
	if shapeLength != length {
		return nil, fmt.Errorf("shape %v of length %d does not match the value buffer of length %d=%d/sizeof(float32)", shape, shapeLength, length, len(storage))
	}
	// Cast a slice of bytes into a slice of floats.
	// See https://stackoverflow.com/questions/48756732/what-does-1-30c-yourtype-do-exactly-in-cgo
	var value []float32
	if length > 0 {
		value = (*[1 << 30]float32)(unsafe.Pointer(&(storage[0])))[:length:length]
	}
	return &PBTensor{shape: shape, value: value, storage: p}, nil
}

// ToProto converts a tensor into a Tensor proto.
func ToProto(t Tensor) *pb.Tensor {
	vals := t.Values()
	storage := make([]byte, len(vals)*floatSize)
	// Cast a float32 slice into a byte slice.
	// See https://stackoverflow.com/questions/48756732/what-does-1-30c-yourtype-do-exactly-in-cgo
	if len(vals) > 0 {
		values := (*[1 << 30]byte)(unsafe.Pointer(&(vals[0])))[:len(storage):len(storage)]
		copy(storage, values)
	}
	shape := t.Shape()
	dims := make([]*pb.Shape_Dim, len(shape))
	for i, dim := range shape {
		dims[i] = &pb.Shape_Dim{
			Size: int64(dim),
		}
	}
	return &pb.Tensor{
		Shape: &pb.Shape{
			Dim: dims,
		},
		Content: storage,
		Dtype:   pb.DataType_DT_FLOAT32,
	}
}
