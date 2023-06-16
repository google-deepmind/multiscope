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

package timedb

import (
	"fmt"
	"multiscope/internal/server/core/timeline"
	treepb "multiscope/protos/tree_go_proto"

	"golang.org/x/exp/maps"
)

// Record saved in the database.
// We store the data in a tree such that we can record the data or error
// for all levels.
// For instance, for the path /a/b/c/, we may have an error at level /a
// (e.g. cannot get the children).
// Storing a recursive data structure, as opposed to a flat structure, makes
// it easy to retrieve the error at level /a if the path /a/b/c is requested.
type Record struct {
	data timeline.Marshaler

	fetchChildrenError error
	children           map[string]*Record
}

func newErrorRecord(path *treepb.NodePath, err error) *Record {
	return newRecord(timeline.ToErrorMarshaler(path, err))
}

func newRecord(data timeline.Marshaler) *Record {
	return &Record{data: data}
}

// MarshalData from the record.
func (rec *Record) MarshalData(data *treepb.NodeData, path []string, lastTick uint32) {
	if len(path) == 0 {
		rec.data.MarshalData(data, path, lastTick)
		return
	}
	if rec.fetchChildrenError != nil {
		data.Error = rec.fetchChildrenError.Error()
		return
	}
	childName := path[0]
	child := rec.children[childName]
	if child == nil {
		data.Error = fmt.Sprintf("child %q in path %v cannot be found in the timeline. Available children are: %v", childName, path, maps.Keys(rec.children))
		return
	}
	child.MarshalData(data, path[1:], lastTick)
}

func (rec *Record) addChild(name string, child *Record) {
	if rec.children == nil {
		rec.children = make(map[string]*Record)
	}
	rec.children[name] = child
}

func (rec *Record) setFetchChildrenError(err error) {
	rec.fetchChildrenError = err
}

func (rec *Record) computeSize() uint64 {
	size := rec.data.StorageSize()
	for _, child := range rec.children {
		size += child.computeSize()
	}
	return size
}
