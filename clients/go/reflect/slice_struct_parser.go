package reflect

import (
	"fmt"
	"reflect"
	"strconv"

	"multiscope/clients/go/remote"
)

type sliceStructParser struct{}

func (sliceStructParser) CanParse(obj any) bool {
	tp := reflect.TypeOf(obj)
	if tp.Kind() != reflect.Slice {
		return false
	}
	tp = tp.Elem()
	if tp.Kind() == reflect.Interface {
		return true
	}
	if tp.Kind() != reflect.Ptr {
		return false
	}
	tp = tp.Elem()
	if tp.Kind() != reflect.Struct {
		return false
	}
	return true
}

func emptySlice(parent Parent, name string) (remote.Node, error) {
	w, err := remote.NewTextWriter(parent.Node().Client(), name, parent.Node().Path())
	if err != nil {
		return nil, err
	}
	w.Write("empty slice")
	return w, nil
}

func (sliceStructParser) Parse(state *ParserState, name string, fObj TargetGetter) (remote.Node, error) {
	obj := fObj()
	nodeName := fmt.Sprintf("%s:%T", name, obj)
	val := reflect.ValueOf(obj)
	parent := state.Parent()
	if val.Len() == 0 {
		return emptySlice(parent, nodeName)
	}
	grp, err := remote.NewGroup(parent.Node().Client(), nodeName, parent.Node().Path())
	if err != nil {
		return nil, err
	}
	state.Push(grp, obj)
	defer state.Pop()
	for i := 0; i < val.Len(); i++ {
		valIndex := i
		fObj := func() any {
			if valIndex >= val.Len() {
				return nil
			}
			el := val.Index(valIndex)
			if !el.CanInterface() {
				return nil
			}
			return el.Interface()
		}
		if _, err := state.Parse(strconv.Itoa(valIndex), fObj, nil); err != nil {
			return nil, err
		}
	}
	return grp, nil
}
