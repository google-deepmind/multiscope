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

package reflect

import (
	"fmt"
	"reflect"

	"multiscope/clients/go/remote"
)

const multiscopeTag = "multiscope_reflect"

type structParser struct{}

func (structParser) CanParse(obj any) bool {
	val := reflect.ValueOf(obj)
	if val.Kind() != reflect.Ptr {
		return false
	}
	val = val.Elem()
	if val.Kind() != reflect.Struct {
		return false
	}
	return true
}

func (structParser) Parse(state *ParserState, name string, fObj TargetGetter) (remote.Node, error) {
	obj := fObj()
	parent := state.Parent()
	nodeName := fmt.Sprintf("%s:%T", name, obj)
	grp, err := remote.NewGroup(parent.Node().Client(), nodeName, parent.Node().Path())
	if err != nil {
		return nil, err
	}
	state.Push(grp, obj)
	defer state.Pop()
	val := reflect.ValueOf(obj).Elem()
	if err := parseFields(state, val); err != nil {
		return nil, err
	}
	return grp, nil
}

var accessibleFloatKind = map[reflect.Kind]bool{
	reflect.Float32: true, reflect.Float64: true,
}

var accessibleIntKind = map[reflect.Kind]bool{
	reflect.Int:  true,
	reflect.Int8: true, reflect.Int16: true, reflect.Int32: true, reflect.Int64: true,
}

var accessibleUintKind = map[reflect.Kind]bool{
	reflect.Uint:  true,
	reflect.Uint8: true, reflect.Uint16: true, reflect.Uint32: true, reflect.Uint64: true,
}

func parseField(val reflect.Value, field int) (TargetGetter, error) {
	fieldType := val.Type().Field(field)
	if methodName := fieldType.Tag.Get(multiscopeTag); methodName != "" {
		return callExportMethod(val, methodName)
	}
	fieldVal := val.Field(field)
	if fieldVal.CanInterface() {
		return func() any {
			return val.Field(field).Interface()
		}, nil
	}
	if accessibleIntKind[fieldVal.Kind()] {
		return func() any {
			return val.Field(field).Int()
		}, nil
	}
	if accessibleUintKind[fieldVal.Kind()] {
		return func() any {
			return val.Field(field).Uint()
		}, nil
	}
	if accessibleFloatKind[fieldVal.Kind()] {
		return func() any {
			return val.Field(field).Float()
		}, nil
	}
	return nil, nil
}

func parseFields(state *ParserState, val reflect.Value) error {
	for i := 0; i < val.NumField(); i++ {
		fObj, err := parseField(val, i)
		fieldName := val.Type().Field(i).Name
		if err != nil {
			return fmt.Errorf("cannot access the value of the field %s for the type %v: %v", fieldName, val.Type(), err)
		}
		if fObj == nil {
			continue
		}
		if _, err := state.Parse(fieldName, fObj, nil); err != nil {
			return err
		}
	}
	return nil
}

func newFieldFloatAccess(state *ParserState, val reflect.Value, field int) (remote.Node, error) {
	fieldName := val.Type().Field(field).Name
	fObj := func() any {
		return val.Field(field).Float()
	}
	return state.Parse(fieldName, fObj, nil)
}

// callExportMethod calls a method, then returns the object returned by the method.
func callExportMethod(val reflect.Value, methodName string) (TargetGetter, error) {
	method := val.MethodByName(methodName)
	if !method.IsValid() && val.CanAddr() {
		method = val.Addr().MethodByName(methodName)
	}
	if !method.IsValid() {
		return nil, fmt.Errorf("method %s() cannot be found for type %s.%s", methodName, val.Type().PkgPath(), val.Type().Name())
	}
	methodType := method.Type()
	if methodType.NumIn() != 0 {
		return nil, fmt.Errorf("method %s should take no arguments (found %v)", methodName, methodType.NumIn())
	}
	if methodType.NumOut() != 1 {
		return nil, fmt.Errorf("method %s should return only a single value (found %v)", methodName, methodType.NumOut())
	}
	return func() any {
		return method.Call([]reflect.Value{})[0].Interface()
	}, nil
}
