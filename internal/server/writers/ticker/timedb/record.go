package timedb

import (
	treepb "multiscope/protos/tree_go_proto"

	"golang.org/x/exp/maps"
	"google.golang.org/protobuf/proto"
)

// Record saved in the database.
// We store the data in a tree such that we can record the data or error
// for all levels.
// For instance, for the path /a/b/c/, we may have an error at level /a
// (e.g. cannot get the children).
// Storing a recursive data structure, as opposed to a flat structure, makes
// it easy to retrieve the error at level /a if the path /a/b/c is requested.
type Record struct {
	data *treepb.NodeData

	fetchChildrenError error
	children           map[string]*Record
}

func newErrorRecord(err error) *Record {
	return newRecord(&treepb.NodeData{Error: err.Error()})
}

func newRecord(data *treepb.NodeData) *Record {
	return &Record{data: data}
}

// Data returns the serialized data for the record.
func (rec *Record) Data() *treepb.NodeData {
	return rec.data
}

// Child returns a child record given a name.
func (rec *Record) Child(name string) *Record {
	if rec.fetchChildrenError != nil {
		return newErrorRecord(rec.fetchChildrenError)
	}
	return rec.children[name]
}

// Children returns the list of children.
func (rec *Record) Children() []string {
	return maps.Keys(rec.children)
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
	size := uint64(proto.Size(rec.data))
	for _, child := range rec.children {
		size += child.computeSize()
	}
	return size
}
