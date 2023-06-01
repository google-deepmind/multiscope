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
	"reflect"

	"multiscope/clients/go/remote"

	"github.com/pkg/errors"
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
	var sub remote.Subscriber
	if err := state.Find(&sub); err != nil {
		return nil, errors.Errorf("cannot find a parent subscriber to register the writer")
	}
	sub.Subscribe(func() error {
		if !writer.ShouldWrite() {
			return nil
		}
		o := fObj()
		f, ok := toFloat(o)
		if !ok {
			return errors.Errorf("cannot convert %q with %v of type %T to float64", name, o, o)
		}
		return writer.WriteFloat64(map[string]float64{
			name: f,
		})
	})
	return writer, nil
}
