package timedb

import (
	"multiscope/internal/server/core"
	treepb "multiscope/protos/tree_go_proto"
)

type context struct {
	current core.Node
}

// NewConstructor returns a new constructor to construct a new record.
func (db *TimeDB) newContext() *context {
	return &context{current: db.node}
}

func (ctx *context) buildChildRecord(parent core.Parent, childName string) *Record {
	child, childErr := parent.Child(childName)
	if childErr != nil {
		return newErrorRecord(childErr)
	}
	childCtx := &context{
		current: child,
	}
	return childCtx.buildRecord()
}

func (ctx *context) buildRecord() *Record {
	data := &treepb.NodeData{}
	ctx.current.MarshalData(data, nil, 0)

	rec := newRecord(data)
	parent, ok := ctx.current.(core.Parent)
	if !ok {
		return rec
	}
	children, err := parent.Children()
	if err != nil {
		rec.setFetchChildrenError(err)
		return rec
	}
	for _, childName := range children {
		childRec := ctx.buildChildRecord(parent, childName)
		rec.addChild(childName, childRec)
	}
	return rec
}
