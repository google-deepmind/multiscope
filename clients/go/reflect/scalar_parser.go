package reflect

import (
	"fmt"
	"reflect"

	"multiscope/clients/go/remote"
)

type scalarParser struct{}

var floatTarget = reflect.TypeOf(float64(0))

func toFloat(v any) (float64, bool) {
	if !reflect.TypeOf(v).ConvertibleTo(floatTarget) {
		return 0, false
	}
	return reflect.ValueOf(v).Convert(floatTarget).Float(), true
}

func (scalarParser) CanParse(obj any) bool {
	_, ok := toFloat(obj)
	return ok
}

func (scalarParser) Parse(state *ParserState, name string, fObj TargetGetter) (remote.Node, error) {
	parent := state.Parent().Node()
	writer, err := remote.NewScalarWriter(parent.Client(), name, parent.Path())
	if err != nil {
		return nil, err
	}
	state.Root().(remote.Subscriber).Subscribe(func() error {
		if !writer.ShouldWrite() {
			return nil
		}
		o := fObj()
		f, ok := toFloat(o)
		if !ok {
			return fmt.Errorf("cannot convert %q with %v of type %T to float64", name, o, o)
		}
		return writer.WriteFloat64(map[string]float64{
			name: f,
		})
	})
	return writer, nil
}
